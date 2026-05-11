"""
sd-forge-sdxl-gguf-brandulateai
================================

Adds support for loading SDXL .gguf checkpoints in Forge / Forge Neo.

Brandulate AI · https://patreon.com/brandulate · https://discord.gg/DZEenb5wGc
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_PATCH_APPLIED = False

# Diagnostic log file — every patch event is also appended here so users (or
# tooling) can read what happened without needing access to Forge's
# interactive terminal.
_LOG_PATH = Path(__file__).resolve().parent.parent / "gguf_loader.log"


def _diag(msg: str) -> None:
    """Print to stderr AND append to the diagnostic log file."""
    line = f"[{time.strftime('%H:%M:%S')}] [sdxl-gguf-loader] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _extract_shape_from_field(field: Any) -> Optional[List[int]]:
    """
    Extract the integer array stored in a GGUFReader array field.

    `writer.add_array(key, (320, 4, 3, 3))` produces a `ReaderField` whose
    `parts` is laid out like:
        parts[0]   key length      (uint64)
        parts[1]   key bytes       (uint8 array)
        parts[2]   value type      (uint32 = 9 = ARRAY)
        parts[3]   element type    (uint32 = 5 = INT32)
        parts[4]   array length    (uint64 = N)
        parts[5..] N entries, one numpy array per element

    `field.data` holds the indices into `parts` of those element entries
    (e.g. `[5, 6, 7, 8]` for a 4-element shape). Iterate those, unpack each
    single-element numpy array, collect into a Python list of ints.
    """
    try:
        if not (hasattr(field, "data") and hasattr(field, "parts")):
            return None
        idxs = list(field.data) if field.data else []
        if not idxs:
            return None
        vals: List[int] = []
        for idx in idxs:
            if idx < 0 or idx >= len(field.parts):
                continue
            arr = field.parts[idx]
            try:
                if hasattr(arr, "tolist"):
                    t = arr.tolist()
                    if isinstance(t, list):
                        for v in t:
                            vals.append(int(v))
                    elif isinstance(t, (int, float)):
                        vals.append(int(t))
                elif hasattr(arr, "__iter__"):
                    for v in arr:
                        vals.append(int(v))
                else:
                    vals.append(int(arr))
            except Exception:
                continue
        return vals if vals else None
    except Exception:
        return None


def _build_orig_shape_map(reader) -> Dict[str, List[int]]:
    """Walk `reader.fields` and pull out every `comfy.gguf.orig_shape.<key>`
    entry into a {tensor_name: shape_dims} dict."""
    out: Dict[str, List[int]] = {}
    prefix = "comfy.gguf.orig_shape."
    fields = getattr(reader, "fields", {}) or {}
    try:
        items = fields.items()
    except Exception:
        return out
    for key, field in items:
        if not isinstance(key, str):
            continue
        if not key.startswith(prefix):
            continue
        tensor_name = key[len(prefix):]
        shape = _extract_shape_from_field(field)
        if shape:
            out[tensor_name] = shape
    return out


def _patched_load_torch_file(orig_func):
    import gguf as _gguf
    import numpy as _np
    import torch as _torch

    def _wrapped(ckpt, *args, **kwargs):
        try:
            ckpt_label = ckpt if isinstance(ckpt, str) else f"<non-str: {type(ckpt).__name__}>"
            _diag(f"load: {ckpt_label}")
        except Exception:
            pass

        try:
            is_gguf = isinstance(ckpt, str) and ckpt.lower().endswith(".gguf")
        except Exception:
            is_gguf = False
        if not is_gguf:
            return orig_func(ckpt, *args, **kwargs)

        try:
            from backend.operations_gguf import ParameterGGUF, quants_mapping
        except Exception as e:
            _diag(f"falling back to default loader: {e}")
            return orig_func(ckpt, *args, **kwargs)

        reader = _gguf.GGUFReader(ckpt)
        sd = {}
        orig_shapes = _build_orig_shape_map(reader)
        quant_types = set(quants_mapping.keys())

        n_a = n_b = n_c = n_d = n_total = 0
        for tensor in reader.tensors:
            name = str(tensor.name)
            n_total += 1
            is_quantized = tensor.tensor_type in quant_types

            if is_quantized:
                param = ParameterGGUF(tensor)
                if name in orig_shapes:
                    shape_ints = [int(d) for d in orig_shapes[name]]
                    n_a += 1
                else:
                    shape_ints = [int(d) for d in param.real_shape]
                try:
                    param.real_shape = _torch.Size(shape_ints)
                except Exception as e:
                    _diag(f"shape set failed on {name}: {e}")

                if len(shape_ints) == 4:
                    try:
                        dequant_param = param.dequantize_as_pytorch_parameter()
                        dequant_data = dequant_param.data.to(dtype=_torch.float16).contiguous().view(shape_ints)
                        sd[name] = _torch.nn.Parameter(dequant_data, requires_grad=False)
                        n_c += 1
                    except Exception as e:
                        _diag(f"conv path failed for {name}: {e}")
                        sd[name] = param
                        n_b += 1
                else:
                    sd[name] = param
                    n_b += 1
            else:
                try:
                    np_data = _np.asarray(tensor.data)
                    plain = _torch.from_numpy(np_data.copy())
                    sd[name] = _torch.nn.Parameter(plain, requires_grad=False)
                    n_d += 1
                except Exception as e:
                    _diag(f"plain path failed for {name}: {e}")
                    param = ParameterGGUF(tensor)
                    shape_ints = [int(d) for d in param.real_shape]
                    try:
                        param.real_shape = _torch.Size(shape_ints)
                    except Exception:
                        pass
                    sd[name] = param

        fname = ckpt.replace("\\", "/").rsplit("/", 1)[-1]
        _diag(f"{fname}: {n_total} tensor(s) loaded.")

        if kwargs.get("return_metadata", False):
            return sd, None
        return sd

    return _wrapped


def apply():
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        if _LOG_PATH.exists():
            try:
                prev = _LOG_PATH.with_suffix(".log.prev")
                if prev.exists():
                    prev.unlink()
                _LOG_PATH.rename(prev)
            except Exception:
                pass
    except Exception:
        pass
    _diag(f"starting · python={sys.version_info.major}.{sys.version_info.minor}")

    try:
        import backend.utils as _bu
    except Exception as e:
        _diag(f"Forge backend not detected — skipping ({e})")
        return False
    if not hasattr(_bu, "load_torch_file"):
        _diag(f"Forge layout has changed, skipping.")
        return False
    try:
        _original = _bu.load_torch_file
        _wrapped = _patched_load_torch_file(_original)

        _bu.load_torch_file = _wrapped

        n_modules_patched = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is _bu:
                continue
            try:
                ltf = getattr(mod, "load_torch_file", None)
            except Exception:
                continue
            if ltf is _original:
                try:
                    setattr(mod, "load_torch_file", _wrapped)
                    n_modules_patched += 1
                except Exception:
                    pass
        _diag(f"linked {n_modules_patched} module(s).")
    except Exception as e:
        _diag(f"init failed: {e}")
        return False
    _PATCH_APPLIED = True
    _diag(f"Active.")
    return True


apply()
