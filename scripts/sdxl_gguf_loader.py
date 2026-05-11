"""
sd-forge-sdxl-gguf-brandulateai · sdxl_gguf_loader.py
======================================================

Tiny startup-time monkey-patch that teaches Forge / Forge Neo's GGUF reader
to load SDXL `.gguf` checkpoints correctly. Without this extension, Forge
recognizes Flux GGUFs but fails on SDXL GGUFs with a chain of cryptic errors
("Failed to recognize model type", "slice indices must have __index__",
"ParameterGGUF detach() returns Tensor", "size of tensor a (3) must match
tensor b (144)"). With this extension installed, SDXL GGUFs produced by
ComfyUI-GGUF's `convert.py` (and any other writer that follows the same
metadata conventions) load directly from Forge's standard checkpoint dropdown.

Maintained by Brandulate AI · https://github.com/brandulateai

Why this is needed
------------------
1. K-quant block alignment.
   The K-quant kernels in `gguf.quants.quantize()` require tensor data to be
   aligned into 256-element blocks. For convolution weights like SDXL's
   `model.diffusion_model.input_blocks.0.0.weight = (320, 4, 3, 3) = 11520`
   elements, the conv layout doesn't divide cleanly into 256, so the GGUF
   writer's "rearrangement trick" pre-reshapes the tensor to `(45, 256)`
   before quantizing. The original shape is written as a separate GGUF KV
   metadata entry named `comfy.gguf.orig_shape.<tensor-name>`.
   ComfyUI-GGUF's loader knows to read that metadata and restore the shape
   on load. Forge's `backend/operations_gguf.py::ParameterGGUF` does NOT —
   it just calls `torch.Size(reversed(tensor.shape))` on the GGUF file's
   shape header. So the reshaped `(45, 256)` survives all the way to
   `huggingface_guess`, which then can't match it to any model.

2. numpy.uint64 slice indices.
   `ParameterGGUF.__init__` builds `real_shape` from a numpy uint64 array.
   `huggingface_guess/utils.py:88` later does `weights.shape[0] // 3` and
   slices with the result. On many Python/numpy combos a numpy uint64
   slice index raises TypeError.

3. CLIP/VAE loaders not GGUF-aware.
   `backend/loader.py` only routes the UNet branch through
   `ForgeOperationsGGUF`. The CLIP and VAE branches use the default
   `ForgeOperations`, whose `Linear._load_from_state_dict` wraps tensors in
   `torch.nn.Parameter(...)`. nn.Parameter requires `detach()` to preserve
   subclass type; `ParameterGGUF.detach()` returns a plain Tensor → error.

4. No GGUF Conv2d.
   `ForgeOperationsGGUF` overrides only `Linear` and `Embedding`. SDXL has
   ~50 quantized 4D conv weights that fall through to `nn.Conv2d`'s default
   strict-shape `_load_from_state_dict`, which compares the model's
   `(out, in, kH, kW)` to the packed-byte storage shape `(N_blocks, 144)`
   and fails. Flux GGUFs don't hit this because Flux has no Conv2d layers.

The fix
-------
For each tensor read from a `.gguf` file, route one of three ways:

  Tensor profile                              -> What we hand to Forge
  ------------------------------------------------------------------------
  Non-quantized (F16/F32/BF16)                -> plain torch.nn.Parameter
    (typically all CLIP and VAE tensors)
  Quantized + 2D (linear/embedding weight)    -> ParameterGGUF with
                                                  real_shape restored from
                                                  orig_shape metadata
                                                  (~94% of UNet params)
  Quantized + 4D (conv kernel weight)         -> dequantize via
                                                  ParameterGGUF
                                                  .dequantize_as_pytorch_parameter()
                                                  → plain F16 Parameter

Non-`.gguf` paths fall through to the unmodified `load_torch_file`. Flux
GGUFs keep working exactly as before — for them, no tensor matches the
"quantized + 4D conv" case, so the conv-dequantize step is a no-op.

Diagnostic log
--------------
Every patch event is appended to `gguf_loader.log` next to this script so
you can confirm the patch fired and see per-file tensor breakdowns. The log
rotates to `.log.prev` on each Forge restart.

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
    """Wrap backend.utils.load_torch_file so SDXL GGUF loads work end-to-end."""
    import gguf as _gguf
    import numpy as _np
    import torch as _torch

    def _wrapped(ckpt, *args, **kwargs):
        try:
            ckpt_label = ckpt if isinstance(ckpt, str) else f"<non-str: {type(ckpt).__name__}>"
            _diag(f"_wrapped() called for: {ckpt_label}")
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
            _diag(f"Could not import ParameterGGUF, falling back to vanilla loader: {e}")
            return orig_func(ckpt, *args, **kwargs)

        _diag(f"GGUF detected — reading via GGUFReader")
        reader = _gguf.GGUFReader(ckpt)
        sd = {}
        orig_shapes = _build_orig_shape_map(reader)
        _diag(f"Found {len(orig_shapes)} comfy.gguf.orig_shape.* entries in this file")

        # Set of GGUF type enums that Forge's quants_mapping handles natively.
        # For tensors of these types we generally keep `ParameterGGUF` (so
        # ForgeOperationsGGUF.Linear can dequantize on-the-fly during forward).
        # Exceptions:
        #   - non-quantized tensors (F16/F32/BF16) → plain `torch.nn.Parameter`
        #     so the regular CLIP/VAE/Embedding `_load_from_state_dict` paths
        #     don't choke on a Parameter subclass whose `detach()` doesn't
        #     preserve type.
        #   - **4D conv weights** → dequantize at load time to plain
        #     `torch.nn.Parameter`. ForgeOperationsGGUF does NOT override
        #     Conv2d._load_from_state_dict, so quantized 4D weights would
        #     fail nn.Conv2d's strict shape check (packed `(N_blocks, 144)`
        #     vs declared `(out, in, kH, kW)`). Dequantizing them up front
        #     trades ~30 % UNet size for "actually works on SDXL".
        quant_types = set(quants_mapping.keys())

        n_patched = 0
        n_kept_gguf = 0
        n_unwrapped_plain = 0
        n_dequantized_conv = 0
        n_tensors_seen = 0
        for tensor in reader.tensors:
            name = str(tensor.name)
            n_tensors_seen += 1
            is_quantized = tensor.tensor_type in quant_types

            if is_quantized:
                # Quantized path — first wrap in ParameterGGUF, then decide
                # whether to keep quantized (Linear) or dequantize (Conv).
                param = ParameterGGUF(tensor)
                # Restore logical shape: orig_shape metadata wins; else coerce
                # the gguf-reader uint64 shape to plain Python ints (otherwise
                # huggingface_guess slicing errors out on np.uint64 indices).
                if name in orig_shapes:
                    shape_ints = [int(d) for d in orig_shapes[name]]
                    n_patched += 1
                else:
                    shape_ints = [int(d) for d in param.real_shape]
                try:
                    param.real_shape = _torch.Size(shape_ints)
                except Exception as e:
                    _diag(f"failed to set real_shape on {name}: {e}")

                # If this is a 4D weight (conv kernel), dequantize now. nn.Conv2d's
                # default `_load_from_state_dict` does a strict shape check that
                # ParameterGGUF can't satisfy without a custom Conv2d in
                # ForgeOperationsGGUF (Forge only ships Linear+Embedding GGUF
                # overrides). Dequantize → ship as plain F16 Parameter.
                #
                # Use `dequantize_as_pytorch_parameter()` rather than the
                # standalone `dequantize_tensor()` helper — the former calls
                # `gguf_cls.bake(self)` first, which the dequant kernels require
                # ("GGUF Tensor is not baked!" otherwise).
                if len(shape_ints) == 4:
                    try:
                        dequant_param = param.dequantize_as_pytorch_parameter()
                        dequant_data = dequant_param.data.to(dtype=_torch.float16).contiguous().view(shape_ints)
                        sd[name] = _torch.nn.Parameter(dequant_data, requires_grad=False)
                        n_dequantized_conv += 1
                    except Exception as e:
                        _diag(f"failed to dequantize 4D weight {name}: {e}; falling back to ParameterGGUF")
                        sd[name] = param
                        n_kept_gguf += 1
                else:
                    sd[name] = param
                    n_kept_gguf += 1
            else:
                # UNWRAP non-quantized to plain torch.nn.Parameter — this is the
                # fix for CLIP / VAE / non-quantized UNet tensors. The GGUF
                # reader stores these in their natural numpy dtype with the
                # natural logical shape, so a direct `torch.from_numpy()`
                # round-trip preserves both.
                try:
                    np_data = _np.asarray(tensor.data)
                    plain = _torch.from_numpy(np_data.copy())
                    sd[name] = _torch.nn.Parameter(plain, requires_grad=False)
                    n_unwrapped_plain += 1
                except Exception as e:
                    _diag(f"could not unwrap {name} (dtype={getattr(np_data, 'dtype', '?')}); "
                          f"falling back to ParameterGGUF wrap. err={e}")
                    param = ParameterGGUF(tensor)
                    shape_ints = [int(d) for d in param.real_shape]
                    try:
                        param.real_shape = _torch.Size(shape_ints)
                    except Exception:
                        pass
                    sd[name] = param

        fname = ckpt.replace("\\", "/").rsplit("/", 1)[-1]
        _diag(f"{fname}: shape-restored {n_patched}/{len(orig_shapes)} reshape-aligned tensor(s). "
              f"Kept as ParameterGGUF (Linear/Embedding): {n_kept_gguf}. "
              f"Dequantized 4D conv weights to plain Parameter: {n_dequantized_conv}. "
              f"Unwrapped non-quantized to plain Parameter: {n_unwrapped_plain}. "
              f"Total: {n_tensors_seen}.")

        # Match Forge's load_torch_file signature: return (sd, metadata) when
        # asked, else just sd. GGUF files have no safetensors metadata.
        if kwargs.get("return_metadata", False):
            return sd, None
        return sd

    return _wrapped


def apply():
    """
    Idempotent — safe to call multiple times. No-op if already patched.

    Patches every reference to `load_torch_file` we can find:
      1. `backend.utils.load_torch_file` (the canonical location)
      2. Every other module in `sys.modules` that has its own
         `load_torch_file` attribute identical to the original — these are
         the modules that did `from backend.utils import load_torch_file`
         at their own module load time, which happens BEFORE this extension's
         scripts run. They captured a local reference to the original
         function; we need to overwrite each of those too.

    Critical importers we know about (`backend/loader.py`,
    `extensions-builtin/sd_forge_lora/networks.py`, `modules/extras.py`,
    `backend/patcher/clipvision.py`, etc.) all do
    `from backend.utils import load_torch_file` at module level.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    # Rotate the diagnostic log on each apply() so we get a fresh view per
    # Forge restart. Keep the previous one as .log.prev for one-step rollback.
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
    _diag(f"apply() starting · python={sys.version_info.major}.{sys.version_info.minor} "
          f"· log file = {_LOG_PATH}")

    try:
        import backend.utils as _bu
    except Exception as e:
        _diag(f"Forge backend.utils not importable — skipping patch ({e})")
        return False
    if not hasattr(_bu, "load_torch_file"):
        _diag(f"backend.utils.load_torch_file not found — Forge layout has changed, patch skipped.")
        return False
    try:
        _original = _bu.load_torch_file
        _wrapped = _patched_load_torch_file(_original)

        # 1. Patch the canonical location
        _bu.load_torch_file = _wrapped

        # 2. Patch EVERY module that already imported the original by name.
        #    This is the critical step — `backend.loader` already has its
        #    own `load_torch_file` reference, and that's what actually
        #    gets called for .gguf loads.
        n_modules_patched = 0
        patched_module_names = []
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
                    patched_module_names.append(mod_name)
                except Exception:
                    pass
        if n_modules_patched:
            preview = ", ".join(patched_module_names[:10])
            more = f" +{n_modules_patched - 10} more" if n_modules_patched > 10 else ""
            _diag(f"Patched local references in {n_modules_patched} module(s): {preview}{more}")
        else:
            _diag(f"WARNING: 0 modules had load_torch_file matching the original. "
                  f"Patch may not catch downstream callers — investigate sys.modules.")
    except Exception as e:
        _diag(f"Patch install failed: {e}")
        return False
    _PATCH_APPLIED = True
    _diag(f"Active — SDXL .gguf shape-restoration enabled.")
    return True


# Apply at import time. Forge auto-imports every .py file in scripts/, so this
# runs once during Forge's startup `load scripts:` phase.
apply()
