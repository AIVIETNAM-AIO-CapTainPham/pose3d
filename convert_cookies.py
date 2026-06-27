#!/usr/bin/env python3
"""
Convert cookies JSON (J2TEAM / Cookie-Editor) → Netscape cookies.txt
Dùng được với gallery-dl, yt-dlp, wget, curl...

Cách dùng:
    python3 convert_cookies.py cookies.json
    python3 convert_cookies.py cookies.json cookies.txt   # custom output path
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def convert(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ 2 format: array thẳng hoặc dict {"domain": [...]}
    if isinstance(data, dict):
        cookies = []
        for items in data.values():
            if isinstance(items, list):
                cookies.extend(items)
    elif isinstance(data, list):
        cookies = data
    else:
        print("Không nhận ra format JSON này.")
        sys.exit(1)

    lines = [
        "# Netscape HTTP Cookie File",
        f"# Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    count = 0
    for c in cookies:
        domain  = c.get("domain", "")
        path    = c.get("path", "/")
        secure  = "TRUE" if c.get("secure", False) else "FALSE"
        expires = int(c.get("expirationDate", c.get("expires", 0)) or 0)
        name    = c.get("name", "")
        value   = c.get("value", "")

        if not domain or not name:
            continue

        domain_flag = "TRUE" if domain.startswith(".") else "FALSE"
        if not domain.startswith("."):
            domain = "." + domain

        lines.append("\t".join([domain, domain_flag, path, secure, str(expires), name, value]))
        count += 1

    Path(output_file).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Convert xong: {count} cookies")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python3 convert_cookies.py <input.json> [output.txt]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "cookies.txt"

    if not Path(inp).exists():
        print(f"Không tìm thấy file: {inp}")
        sys.exit(1)

    convert(inp, out)
