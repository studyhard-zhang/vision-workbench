"""Local preview: python run.py [--port 8765] [--data-dir PATH]."""
import argparse
import logging
from pathlib import Path

try:
    from app.server import create_server
except ModuleNotFoundError as exc:
    if exc.name == 'PIL':
        raise SystemExit('Missing Pillow. Run: python -m pip install --target .vendor -r requirements.txt') from exc
    raise


def main():
    parser = argparse.ArgumentParser(description="视觉训练工作台 · 本地试用")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "var")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = create_server(args.port, args.data_dir / "platform.db")
    print(f"Local: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"Data directory: {args.data_dir.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local workspace.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
