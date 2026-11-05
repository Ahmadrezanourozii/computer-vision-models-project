"""Run a command and render its real terminal output as a terminal-window screenshot (for the slides).

    python slides/terminal_screenshot.py out.png -- python -m mooney convert data/Original -o data/Mooney ...
Long outputs are shortened to the first/last lines with an ellipsis line in between.
"""
import html
import os
import subprocess
import sys
from pathlib import Path


def render(cmd_text, lines, out, head=14, tail=4, title="Terminal"):
    if len(lines) > head + tail + 1:
        lines = lines[:head] + [f"... ({len(lines) - head - tail} more lines) ..."] + lines[-tail:]
    body = "\n".join(html.escape(l) for l in lines)
    page = f"""<html><head><meta charset="utf-8"><style>
    body {{ margin:0; background:#0b0d10; }}
    .win {{ display:inline-block; border:1px solid #30363d; border-radius:10px; overflow:hidden; background:#0d1117; }}
    .bar {{ background:#161b22; color:#c9d1d9; font:14px 'DejaVu Sans',Arial; padding:8px 14px; border-bottom:1px solid #30363d; }}
    .dot {{ display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:7px; vertical-align:-1px; }}
    pre {{ margin:0; padding:14px 18px 16px; color:#c9d1d9; font:15px/1.45 'DejaVu Sans Mono',monospace; }}
    .p {{ color:#7ee787; }} .c {{ color:#f0f6fc; font-weight:bold; }}
    </style></head><body><div class="win" id="win"><div class="bar"><span class="dot" style="background:#ff5f56"></span>
    <span class="dot" style="background:#ffbd2e"></span><span class="dot" style="background:#27c93f"></span>
    &nbsp;{html.escape(title)}</div><pre><span class="p">$</span> <span class="c">{html.escape(cmd_text)}</span>
{body}</pre></div></body></html>"""
    hp = Path(out).with_suffix(".html")
    hp.write_text(page)
    js = f"""const {{ chromium }} = require('playwright');
    (async () => {{ const b = await chromium.launch(); const p = await b.newPage({{ deviceScaleFactor: 2 }});
    await p.goto('file://{hp.resolve()}'); const el = await p.$('#win');
    await el.screenshot({{ path: '{Path(out).resolve()}' }}); await b.close(); }})();"""
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["node", "-e", js], check=True, env=dict(os.environ, NODE_PATH=npm_root))
    print("->", out)


def main():
    out = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).rstrip("\n").split("\n")
    shown = " ".join(c if " " not in c else f'"{c}"' for c in cmd).replace(sys.executable, "python")
    render(shown, lines, out, title="mooney - batch conversion (CLI)")


if __name__ == "__main__":
    main()
