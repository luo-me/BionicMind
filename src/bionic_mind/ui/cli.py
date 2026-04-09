from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="BionicMind - 仿生意识系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="运行模式")
    parser.add_argument("--host", default="0.0.0.0", help="Web模式监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Web模式监听端口")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.mode == "web":
        from bionic_mind.ui.web import run_web
        asyncio.run(run_web(config_path=args.config, host=args.host, port=args.port))
    else:
        from bionic_mind.core.mind import BionicMind
        mind = BionicMind(config_path=args.config)
        asyncio.run(mind.run_interactive())


if __name__ == "__main__":
    main()
