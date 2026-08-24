"""Render a published artifact: the model's markdown into a standalone page.

The model writes REVIEW CONTENT (markdown); everything presentational is
deterministic and lives here — one inline stylesheet, diff coloring, heading
anchors, and the per-section comment affordance. That split is the point:
a model asked to also produce HTML produces different HTML every time, and
the operator's phone is not the place to debug it.

Safety: raw HTML in the markdown stays ESCAPED (markdown-it html=False), and
the custom diff renderer escapes every line before wrapping it in spans —
the artifact URL is a capability, so the page must stay inert even if the
markdown carries hostile text quoted from an MR.

Headings gain slug ids; a heading whose text starts with a finding tag
("F3 — …", "F12: …") gets the id "f3"/"f12" so comments and chat replies can
name the same anchor.
"""
from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML

if TYPE_CHECKING:
    from collections.abc import Sequence

    from markdown_it.token import Token
    from markdown_it.utils import EnvType, OptionsDict

_FINDING_RE = re.compile(r"^(f\d{1,3})\b", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Kept deliberately small: page chrome, readable measure, diff colors that
# survive both themes, and the comment box. No framework, no external fetch.
_CSS = """\
:root { --bg:#f6f7fb; --card:#ffffff; --ink:#1a1d23; --muted:#5f6672;
  --line:#e4e7ee; --accent:#2563eb; --add-bg:#e7f5ec; --add-ink:#116932;
  --del-bg:#fdeaea; --del-ink:#b42318; --hunk:#6941c6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#101318; --card:#181c23; --ink:#e8eaf0; --muted:#9aa1ad;
    --line:#2a2f3a; --accent:#7ba3f7; --add-bg:#12291b; --add-ink:#5fd38a;
    --del-bg:#331717; --del-ink:#f08a80; --hunk:#b69df8; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width:860px; margin:0 auto; padding:24px 16px 96px; }
header.art { padding:20px 0 4px; }
header.art h1 { margin:0; font-size:26px; font-weight:900; }
header.art .meta { color:var(--muted); font-size:13px; margin-top:4px; }
article { background:var(--card); border:1px solid var(--line);
  border-radius:16px; padding:8px 28px 24px; }
h2, h3, h4 { scroll-margin-top:12px; }
h2 { border-bottom:1px solid var(--line); padding-bottom:6px; }
a { color:var(--accent); }
code { background:var(--bg); border:1px solid var(--line);
  border-radius:6px; padding:1px 5px; font-size:87%; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; overflow-x:auto; }
pre code { background:none; border:none; padding:0; }
pre.diff { line-height:1.45; }
pre.diff .add { display:block; background:var(--add-bg); color:var(--add-ink); }
pre.diff .del { display:block; background:var(--del-bg); color:var(--del-ink); }
pre.diff .hunk { display:block; color:var(--hunk); font-weight:600; }
table { border-collapse:collapse; width:100%; }
th, td { border:1px solid var(--line); padding:6px 10px; text-align:left; }
blockquote { margin:0; padding:2px 16px; border-left:3px solid var(--accent);
  color:var(--muted); }
.anchor-tools { float:right; }
.anchor-tools button { border:1px solid var(--line); background:var(--card);
  color:var(--muted); border-radius:8px; font-size:12px; padding:2px 10px;
  cursor:pointer; }
#comment { position:fixed; inset:auto 0 0 0; background:var(--card);
  border-top:1px solid var(--line); padding:10px 16px;
  display:flex; gap:8px; align-items:center; }
#comment input[type=text] { flex:1; padding:9px 12px; border-radius:10px;
  border:1px solid var(--line); background:var(--bg); color:var(--ink); }
#comment button { padding:9px 16px; border-radius:10px; border:none;
  background:var(--accent); color:#fff; font-weight:700; cursor:pointer; }
#comment .tag { color:var(--muted); font-size:13px; white-space:nowrap; }
#toast { position:fixed; bottom:70px; left:50%; transform:translateX(-50%);
  background:var(--ink); color:var(--bg); border-radius:10px;
  padding:8px 16px; font-size:14px; opacity:0; transition:opacity .3s; }
"""

# The comment box POSTs to <artifact>/comment as JSON. "Re <anchor>" is set
# by the per-heading buttons; the reply is fire-and-forget with a toast.
_JS = """\
(function () {
  var tag = document.getElementById('c-tag');
  var box = document.getElementById('c-text');
  var anchor = '';
  document.querySelectorAll('h2[id], h3[id], h4[id]').forEach(function (h) {
    var b = document.createElement('button');
    b.textContent = '\\u{1F4AC} comment';
    b.onclick = function () {
      anchor = h.id;
      tag.textContent = 'Re ' + h.id.toUpperCase() + ':';
      box.focus();
    };
    var wrap = document.createElement('span');
    wrap.className = 'anchor-tools';
    wrap.appendChild(b);
    h.appendChild(wrap);
  });
  function toast(msg) {
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.style.opacity = '1';
    setTimeout(function () { t.style.opacity = '0'; }, 2500);
  }
  document.getElementById('c-send').onclick = function () {
    var text = box.value.trim();
    if (!text) { return; }
    fetch(window.location.pathname.replace(/\\/$/, '') + '/comment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anchor: anchor, text: text }),
    }).then(function (r) {
      if (r.ok) {
        box.value = '';
        anchor = '';
        tag.textContent = '';
        toast('Sent to the bot \\u2713');
      } else if (r.status === 429) {
        toast('Slow down \\u2014 try again in a few seconds');
      } else {
        toast('Failed to send (' + r.status + ')');
      }
    }).catch(function () { toast('Failed to send'); });
  };
})();
"""


def _slug(text: str) -> str:
    """Anchor id for a heading; finding tags win so 'F3 — x' anchors as f3."""
    plain = text.strip().lower()
    m = _FINDING_RE.match(plain)
    if m:
        return m.group(1)
    return _SLUG_RE.sub("-", plain).strip("-")[:64]


def _render_diff(code: str) -> str:
    out: list[str] = []
    for line in code.splitlines():
        esc = html.escape(line) or " "
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f'<span class="add">{esc}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f'<span class="del">{esc}</span>')
        elif line.startswith("@@"):
            out.append(f'<span class="hunk">{esc}</span>')
        else:
            out.append(f"<span>{esc}</span>")
    return f'<pre class="diff"><code>{"".join(out)}</code></pre>\n'


class _Renderer(RendererHTML):
    """Stock HTML renderer + diff fences + heading anchor ids."""

    def fence(
        self, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
    ) -> str:
        token = tokens[idx]
        lang = (token.info or "").strip().split(" ")[0].lower()
        if lang == "diff":
            return _render_diff(token.content)
        return super().fence(tokens, idx, options, env)

    def heading_open(
        self, tokens: Sequence[Token], idx: int, options: OptionsDict, env: EnvType
    ) -> str:
        token = tokens[idx]
        inline = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if inline is not None and inline.type == "inline":
            anchor = _slug(inline.content)
            if anchor:
                token.attrSet("id", anchor)
        return super().renderToken(tokens, idx, options, env)


def _markdown() -> MarkdownIt:
    # html=False (commonmark default here) keeps raw HTML escaped — the one
    # non-negotiable: quoted MR content must never become live markup.
    md = MarkdownIt("commonmark", renderer_cls=_Renderer)
    md.enable("table")
    md.enable("strikethrough")
    md.options["html"] = False
    md.options["linkify"] = False
    return md


def render_artifact(title: str, markdown: str, *, updated: str = "") -> str:
    """One self-contained HTML page: no external fetches, both themes."""
    body = _markdown().render(markdown or "")
    safe_title = html.escape(title or "artifact")
    meta = f'<div class="meta">updated {html.escape(updated)}</div>' if updated else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{safe_title}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header class="art"><h1>{safe_title}</h1>{meta}</header>
<article>
{body}
</article>
</main>
<div id="comment">
  <span class="tag" id="c-tag"></span>
  <input type="text" id="c-text" placeholder="Comment — reaches the bot in chat" maxlength="2000">
  <button id="c-send">Send</button>
</div>
<div id="toast"></div>
<script>{_JS}</script>
</body>
</html>
"""


def anchor_for(heading: str) -> str:
    """Public helper: the id a given heading text will anchor as."""
    return _slug(heading)


__all__ = ["anchor_for", "render_artifact"]
