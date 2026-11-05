"""Editor-style screenshot of the code that generates Mooney images (for the slides).

Renders the real source of ``mooney.core.make_mooney`` (with its true line numbers in
mooney/core.py) as syntax-highlighted HTML and captures it with a headless browser.
"""
import inspect
import os
import subprocess
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mooney import core  # noqa: E402


def main(out="code_screenshot.png", func="make_mooney", style="github-dark"):
    src, start = inspect.getsourcelines(getattr(core, func))
    code = "".join(src)
    fmt = HtmlFormatter(style=style, linenos="inline", linenostart=start, nobackground=False)
    body = highlight(code, PythonLexer(), fmt)
    css = fmt.get_style_defs(".highlight")
    html = f"""<html><head><meta charset="utf-8"><style>
    body {{ margin:0; background:#0d1117; font-family: 'DejaVu Sans Mono', Consolas, monospace; }}
    .win {{ margin:0; border:1px solid #30363d; border-radius:10px; overflow:hidden; display:inline-block; }}
    .bar {{ background:#161b22; color:#c9d1d9; font: 15px 'DejaVu Sans', Arial; padding:9px 14px; border-bottom:1px solid #30363d; }}
    .dot {{ display:inline-block; width:12px; height:12px; border-radius:6px; margin-right:7px; vertical-align:-1px; }}
    .tab {{ margin-left:18px; padding:4px 10px; background:#0d1117; border-radius:6px 6px 0 0; }}
    {css}
    .highlight pre {{ font-size:17px; line-height:1.45; margin:0; }}
    .highlight {{ padding:14px 22px 16px 10px; }}
    .highlight .linenos {{ color:#6e7681; padding-right:18px; margin-right:6px; }}
    </style></head><body><div class="win" id="win">
    <div class="bar"><span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e"></span>
    <span class="dot" style="background:#27c93f"></span><span class="tab">mooney/core.py</span></div>
    {body}</div></body></html>"""
    html_path = Path(out).with_suffix(".html")
    html_path.write_text(html)
    js = f"""const {{ chromium }} = require('playwright');
    (async () => {{ const b = await chromium.launch(); const p = await b.newPage({{ deviceScaleFactor: 2 }});
    await p.goto('file://{html_path.resolve()}'); const el = await p.$('#win');
    await el.screenshot({{ path: '{Path(out).resolve()}' }}); await b.close(); }})();"""
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["node", "-e", js], check=True, env=dict(os.environ, NODE_PATH=npm_root))
    print("->", out, f"(lines {start}-{start + len(src) - 1})")


if __name__ == "__main__":
    main(*sys.argv[1:])
