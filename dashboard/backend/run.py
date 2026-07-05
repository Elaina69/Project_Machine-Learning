"""Development runner for the dashboard backend.

Uvicorn defaults to port 8000. On Windows that port is often already used by
another Python process or by a reserved service, which can raise WinError 10013.
This runner starts from 8010 and falls forward to the next available port.
"""

from __future__ import annotations

import argparse
import os
import socket

import uvicorn


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010


def _can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred_port: int, attempts: int = 30) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        if _can_bind(host, port):
            return port
    raise RuntimeError(
        f"Không tìm được cổng trống trong khoảng {preferred_port}-{preferred_port + attempts - 1}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SV16 dashboard backend.")
    parser.add_argument("--host", default=os.getenv("DASHBOARD_BACKEND_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DASHBOARD_BACKEND_PORT", DEFAULT_PORT)),
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of falling forward when the requested port is unavailable.",
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn reload.")
    args = parser.parse_args()

    if args.strict_port and not _can_bind(args.host, args.port):
        raise RuntimeError(f"Port {args.port} đang bận hoặc không bind được.")

    port = args.port if args.strict_port else _pick_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} đang bận/không bind được, dùng port {port} thay thế.")

    print(f"Backend API: http://{args.host}:{port}")
    if port != DEFAULT_PORT:
        print(
            "Nếu chạy Vite dev server, đặt biến môi trường "
            f"VITE_BACKEND_URL=http://{args.host}:{port}"
        )

    uvicorn.run(
        "dashboard.backend.app:app",
        host=args.host,
        port=port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
