"""
Install-time hook for Forge / A1111-style extension managers.

Most Forge installations already have `gguf` available (it's a dependency
of Forge's built-in Flux GGUF support). This script is a defensive guard for
older Forge builds that pre-date Flux support.
"""
import sys

try:
    import launch  # provided by Forge / A1111

    if not launch.is_installed("gguf"):
        launch.run_pip(
            "install gguf",
            "gguf (required by sd-forge-sdxl-gguf-brandulateai)",
        )
except ImportError:
    # `launch` is only available when this script is invoked by Forge's
    # extension installer. When run standalone (e.g. via pip from a wheel),
    # users are expected to install `gguf` themselves.
    pass
