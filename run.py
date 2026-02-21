from __future__ import annotations

import argparse
import sys

import uvicorn


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run AgentNetwork services locally (registry search or FlightTravel agent).",
    )

    sub = parser.add_subparsers(dest="service", required=True)

    def add_common_flags(p: argparse.ArgumentParser, default_port: int) -> None:
        p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
        p.add_argument("--port", type=int, default=default_port, help=f"Bind port (default: {default_port})")
        p.add_argument(
            "--reload",
            action="store_true",
            help="Enable auto-reload (dev only).",
        )

    p_registry = sub.add_parser("registry", help="Registry/router API (lists agents + POST /search).")
    add_common_flags(p_registry, default_port=8000)

    p_flight = sub.add_parser("flighttravel", help="FlightTravel agent API (POST /FlightTravel).")
    add_common_flags(p_flight, default_port=9006)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    if ns.service == "registry":
        app_path = "app.main:app"
    elif ns.service == "flighttravel":
        app_path = "app.FlightTravel:app"
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown service: {ns.service}")

    uvicorn.run(app_path, host=ns.host, port=ns.port, reload=ns.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

