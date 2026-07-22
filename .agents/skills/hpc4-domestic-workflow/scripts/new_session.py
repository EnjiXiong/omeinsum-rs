#!/usr/bin/env python3
"""Create a session directory for an HPC4 domestic job workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:48] or "job"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".hpc4-domestic")
    parser.add_argument("--title", default="job")
    parser.add_argument("--request", default="")
    args = parser.parse_args()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = pathlib.Path(args.root) / "sessions" / f"{stamp}-{slugify(args.title)}"
    (session / "logs").mkdir(parents=True, exist_ok=False)

    request = session / "request.md"
    request.write_text(
        f"# HPC4 Domestic Job Request\n\n"
        f"- title: {args.title}\n"
        f"- created_utc: {stamp}\n\n"
        f"## User Request\n\n{args.request.strip() or 'TBD'}\n",
        encoding="utf-8",
    )
    print(json.dumps({"session": str(session), "request": str(request)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
