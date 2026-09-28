"""Generate syntax-highlighted LaTeX for the response-format JSON blocks.

Reads the raw snippets from wind-chapter/src/*.js and emits self-contained
LaTeX (fancyvrb Verbatim + \\PYG macros) into wind-chapter/generated/, so the
chapter builds with plain pdflatex (no minted, no --shell-escape).

    python highlight_response.py

Palette (custom navy scheme): keys in navy, string/scalar values in a lighter
navy, comments and URLs in a distinct muted green.
"""
from pathlib import Path
from pygments import highlight
from pygments.lexer import RegexLexer, bygroups
from pygments.token import (Token, Name, String, Comment, Keyword, Number,
                            Punctuation, Text, Error)
from pygments.style import Style
from pygments.formatters import LatexFormatter

HERE = Path(__file__).resolve().parent
WIND = HERE.parent.parent / "wind-chapter"
SRC = WIND / "src"
OUT = WIND / "generated"
OUT.mkdir(parents=True, exist_ok=True)

_STR = r'"(?:[^"\\]|\\.)*"'

class JsoncLexer(RegexLexer):
    """JSON plus // line comments; keys (a string before ':') are tagged."""
    name = "JSONC"
    tokens = {
        "root": [
            (r"\s+", Text),
            (r"//[^\n]*", Comment.Single),
            (rf"({_STR})(\s*)(:)", bygroups(Name.Tag, Text, Punctuation)),
            (_STR, String.Double),
            (r"\b(true|false|null)\b", Keyword.Constant),
            (r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", Number),
            (r"[{}\[\],]", Punctuation),
            (r"[^\s]", Text),
        ]
    }

NAVY = "#2B4C78"      # keys
LNAVY = "#6986AE"     # values (lighter navy)
GREEN = "#4E7A4E"     # comments / urls
GREY = "#555555"      # punctuation

class NavyStyle(Style):
    background_color = "#ffffff"
    styles = {
        Token:            "#333333",
        Name.Tag:         f"bold {NAVY}",
        String:           LNAVY,
        String.Double:    LNAVY,
        Number:           LNAVY,
        Keyword:          LNAVY,
        Keyword.Constant: LNAVY,
        Comment:          f"italic {GREEN}",
        Comment.Single:   f"italic {GREEN}",
        Punctuation:      GREY,
        Error:            "#333333",
    }

def main():
    fmt = LatexFormatter(style=NavyStyle)
    (OUT / "pyg_style.tex").write_text(fmt.get_style_defs())
    for name in ("response_schema", "response_example"):
        code = (SRC / f"{name}.js").read_text()
        (OUT / f"{name}.tex").write_text(highlight(code, JsoncLexer(), fmt))
    print("wrote:", *(p.name for p in sorted(OUT.glob("*.tex"))))

if __name__ == "__main__":
    main()
