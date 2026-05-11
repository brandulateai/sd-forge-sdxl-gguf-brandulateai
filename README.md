# sd-forge-sdxl-gguf-brandulateai

**SDXL GGUF support for Forge / Forge Neo — load `.gguf` SDXL checkpoints (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, …) directly from the standard checkpoint dropdown.**

Maintained by **Brandulate AI** · [Patreon](https://patreon.com/brandulate) · [Discord](https://discord.gg/DZEenb5wGc)

---

## Why this extension exists

Forge and Forge Neo ship with a GGUF reader, but the SDXL path was never finished. Out of the box, loading an SDXL `.gguf` checkpoint fails with one of:

- `ValueError: Failed to recognize model type!`
- `TypeError: slice indices must be integers or None or have an __index__ method`
- `RuntimeError: Creating a Parameter from an instance of type ParameterGGUF requires that detach() returns an instance of the same type`
- `RuntimeError: The size of tensor a (3) must match the size of tensor b (144) at non-singleton dimension 3`

These are four separate bugs across Forge's loader, `ParameterGGUF` class, model detection, and `ForgeOperationsGGUF`. This extension installs a single startup-time monkey-patch that fixes all four — no UI, no per-user config, no manual baking.

Flux GGUFs were already working in Forge and continue to work unchanged.

## What it supports

- **SDXL `.gguf` checkpoints** — any quant the gguf library handles: Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, plus F16/F32/BF16.
- **Bundled GGUFs** (UNet + CLIP-L + CLIP-G + VAE in one file) → load like a normal `.safetensors` from the dropdown.
- **UNet-only GGUFs** (ComfyUI-GGUF style) → load via Forge's "Additional Modules" picker for external CLIP/VAE.

## Installation

### Option A — Forge's Extensions tab (recommended)

1. In Forge, go to the **Extensions** tab → **Install from URL**.
2. Paste:
   ```
   https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
   ```
3. Click **Install**.
4. Go to **Installed** → **Apply and restart UI**.

### Option B — manual clone

```bash
cd <your-forge-install>/extensions
git clone https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
```
Then restart Forge.

### Verification

After restart, look in Forge's terminal for:
```
[sdxl-gguf-loader] Active — SDXL .gguf shape-restoration enabled.
```
Or check `extensions/sd-forge-sdxl-gguf-brandulateai/gguf_loader.log`.

Drop any SDXL `.gguf` into your `models/Stable-diffusion/` folder and pick it from the checkpoint dropdown.

## Requirements

- Forge or Forge Neo (recent enough to have `backend/operations_gguf.py`)
- Python 3.10+
- `gguf` Python package (already installed if your Forge can load Flux GGUFs; otherwise this extension's `install.py` adds it)

## How it works (one-paragraph summary)

The extension monkey-patches `backend.utils.load_torch_file` to detect `.gguf` paths and route each tensor to one of three handlers: non-quantized tensors (F16/F32/BF16, typically CLIP + VAE) are unwrapped to plain `torch.nn.Parameter` so the regular CLIP/VAE loaders see normal tensors; quantized 2D tensors (Linear/Embedding weights) stay as `ParameterGGUF` with their original logical shape restored from `comfy.gguf.orig_shape.*` metadata; quantized 4D tensors (Conv2d kernels) are dequantized to plain F16 at load time because Forge's `ForgeOperationsGGUF` only ships Linear+Embedding overrides — Conv2d has no GGUF-aware variant in Forge. Linear weights make up ~94 % of SDXL UNet parameters, so keeping them quantized preserves most of the size benefit. Non-`.gguf` paths fall through unchanged.

Full technical breakdown in [`scripts/sdxl_gguf_loader.py`](scripts/sdxl_gguf_loader.py) header.

## Diagnostic log

Every load event appends to `gguf_loader.log` in the extension folder. Rotates to `.log.prev` on each Forge restart. Useful when reporting issues.

## Known limitations

- Conv2d weights are dequantized to F16 at load time, so a Q4_0 SDXL `.gguf` ends up using ~30 % more VRAM than a hypothetical "fully quantized" load would. Net result is still significantly smaller than F16 (~3 GB vs ~5 GB for SDXL UNet).
- Tested on Forge Neo with bundled SDXL Q4_0 GGUFs. Other quants and SD 1.5 GGUFs should work via the same code path but haven't been individually validated yet — open an issue if you hit one that fails.

## Contributing

Issues and PRs welcome. If your SDXL GGUF still fails to load:

1. Grab the relevant section of `extensions/sd-forge-sdxl-gguf-brandulateai/gguf_loader.log`
2. Open an issue including the file's quant type and approximate size
3. We'll iterate

## Support our work

If this extension saves you time, consider supporting Brandulate AI:

- **Patreon:** https://patreon.com/brandulate
- **Discord (Brandulate Server):** https://discord.gg/DZEenb5wGc

## Credits

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — established the `comfy.gguf.orig_shape.*` metadata convention this extension reads.
- [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) / [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic) — the host this extension patches.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — original GGUF format.

## License

MIT — see [LICENSE](LICENSE).
