# sd-forge-sdxl-gguf-brandulateai

**SDXL GGUF support for Forge / Forge Neo.**

Maintained by **Brandulate AI** · [Patreon](https://patreon.com/brandulate) · [Discord](https://discord.gg/DZEenb5wGc)

---

## What it does

Adds support for loading **SDXL `.gguf` checkpoints** in Forge and Forge Neo. Previously only Flux GGUFs worked out of the box — with this extension installed, SDXL GGUFs (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.) load directly from the standard checkpoint dropdown. No UI changes, no extra clicks.

Bundled GGUFs (UNet + CLIP + VAE in one file) and UNet-only GGUFs both work.

## Install

### Step 1 — Get Forge (skip if you already have it)

**Recommended:** [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic) (default `neo` branch).

This is the actively maintained continuation of Forge. Includes built-in support for newer models (Kolors, Ernie, Wan, etc.) and is the only Forge build this extension is regularly tested against. The original lllyasviel/stable-diffusion-webui-forge has been inactive since mid-2025 and is not recommended for new installs.

Requires **Python 3.11+**.

### Step 2 — Install this extension

**Via Forge UI (recommended):**

1. Open Forge → **Extensions** tab → **Install from URL**
2. Paste:
   ```
   https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
   ```
3. Click **Install** → go to **Installed** → **Apply and restart UI**

**Or manually:**

```bash
cd <your-forge>/extensions
git clone https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
```
Then restart Forge.

## Use

Drop any SDXL `.gguf` file into `<your-forge>/models/Stable-diffusion/`. After restart it appears in the checkpoint dropdown — pick it and generate.

## Support our work

If this saves you time, support Brandulate AI:

- **Patreon:** https://patreon.com/brandulate
- **Discord (Brandulate Server):** https://discord.gg/DZEenb5wGc

## License

MIT — see [LICENSE](LICENSE).

## Credits

[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) · [Forge (Haoming02)](https://github.com/Haoming02/sd-webui-forge-classic) · [Forge (lllyasviel, original)](https://github.com/lllyasviel/stable-diffusion-webui-forge) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
