import asyncio
import os
from pathlib import Path
from bionic_mind.ui.web import run_web

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    os.chdir(Path(__file__).parent)
    asyncio.run(run_web(config_path=str(config_path)))
