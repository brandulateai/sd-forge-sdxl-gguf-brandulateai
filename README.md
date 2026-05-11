# sd-forge-sdxl-gguf-brandulateai

**SDXL GGUF support for Forge / Forge Neo.**

Maintained by **Brandulate AI** · [Patreon](https://patreon.com/brandulate) · [Discord](https://discord.gg/DZEenb5wGc)

---

## What it does

Adds support for loading **SDXL `.gguf` checkpoints** in Forge and Forge Neo. Previously only Flux GGUFs worked out of the box — with this extension installed, SDXL GGUFs (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.) load directly from the standard checkpoint dropdown. No UI changes, no extra clicks.

Bundled GGUFs (UNet + CLIP + VAE in one file) and UNet-only GGUFs both work.

## Install

### Forge Extensions tab (recommended)

1. Open the **Extensions** tab → **Install from URL**
2. Paste:
   ```
   https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
   ```
3. Click **Install**
4. Go to **Installed** → **Apply and restart UI**

### Manual

```bash
cd <your-forge>/extensions
git clone https://github.com/brandulateai/sd-forge-sdxl-gguf-brandulateai
```
Then restart Forge.

## Use

After restart, your SDXL `.gguf` files appear in the checkpoint dropdown. Pick one and generate.

## Requirements

- Forge or Forge Neo
- Python 3.10+

## Support our work

If this saves you time, support Brandulate AI:

- **Patreon:** https://patreon.com/brandulate
- **Discord (Brandulate Server):** https://discord.gg/DZEenb5wGc

## License

MIT — see [LICENSE](LICENSE).

## Credits

[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) · [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) · [llama.cpp](https://github.com/ggml-org/llama.cpp)
