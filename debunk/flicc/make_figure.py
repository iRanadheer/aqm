"""Build an HTML/SVG dumbbell chart and export with the thesis's browser renderer."""
from html import escape
from pathlib import Path
from xml.etree import ElementTree

OUT = Path(__file__).resolve().parents[3] / "dissertation/chapters/debunk/figures/adjudication_review"
# Same model-family palette as cards/figures/domain_slope.html.
# Point estimates from score_adjudicated.py, ordered by original accuracy.
ACCURACIES = [
    ("DeBERTa", "v2-xlarge", .742, .707, "#9a9a93"),
    ("FLICC-Qwen3.5", "9B", .695, .738, "#3da77a"),
    ("GPT-5.5", "", .656, .758, "#c0561f"),
    ("Claude Opus", "4.7", .656, .750, "#e0922e"),
    ("FLICC-Qwen3.5", "27B", .637, .824, "#1f7a55"),
]


def render():
    assert all(0 <= a <= 1 and 0 <= b <= 1 for _, _, a, b, _ in ACCURACIES)
    def x(score):
        return 300 + (score - .60) / .25 * 510
    parts = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 480"
role="img" aria-labelledby="title description">
<title id="title">Original and expert-review accuracy by model</title>
<desc id="description">Five paired scores. Hollow dots show original gold and filled dots show expert review. Review targeted the 27B model's errors, so reviewed scores are not a fair ranking.</desc>
<style>
text { font-family: 'Helvetica Neue', Arial, sans-serif; fill: #3b4046; font-size: 14px; }
.value { font-variant-numeric: tabular-nums; font-size: 13px; }
</style>
<rect width="1000" height="480" fill="white"/>
<circle cx="300" cy="35" r="5" fill="white" stroke="#555" stroke-width="1.8"/>
<text x="315" y="40">Original gold</text>
<circle cx="465" cy="35" r="5" fill="#555"/>
<text x="480" y="40">Expert review</text>
<text x="920" y="31" text-anchor="middle">Change</text>
<text x="920" y="49" text-anchor="middle" class="value">percentage points</text>''']
    for tick in (.60, .65, .70, .75, .80, .85):
        px = x(tick)
        parts.append(f'<path d="M {px} 70 V 390" stroke="#ecece7"/>')
        parts.append(f'<text x="{px}" y="416" text-anchor="middle" class="value">{tick:.0%}</text>')
    for i, (name, size, original, reviewed, color) in enumerate(ACCURACIES):
        py = 105 + i * 62
        full_name = f"{name}-{size}" if name == "FLICC-Qwen3.5" else f"{name} {size}".strip()
        parts.append(f'''<text x="260" y="{py + 5}" text-anchor="end">{escape(full_name)}</text>
<path d="M {x(original)} {py} H {x(reviewed)}" stroke="{color}" stroke-opacity="0.4" stroke-width="2.6"/>
<circle cx="{x(original)}" cy="{py}" r="5.5" fill="white" stroke="{color}" stroke-width="1.8"/>
<circle cx="{x(reviewed)}" cy="{py}" r="5.5" fill="{color}" stroke="white" stroke-width="1"/>
<text x="{x(original)}" y="{py - 14}" text-anchor="middle" class="value">{original:.1%}</text>
<text x="{x(reviewed)}" y="{py - 14}" text-anchor="middle" class="value">{reviewed:.1%}</text>
<text x="920" y="{py + 5}" text-anchor="middle">{100 * (reviewed - original):+.1f}</text>''')
    parts.append('''<path d="M 300 390 H 810" stroke="#cfcfc9"/>
<text x="555" y="446" text-anchor="middle">Test accuracy</text>
<text x="300" y="473" class="value">n = 256. Review targeted the 27B model’s errors; not a fair model ranking.</text>
</svg>''')
    return "\n".join(parts)


def main():
    svg = render()
    ElementTree.fromstring(svg)
    assert svg.count('<circle') == 2 + 2 * len(ACCURACIES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix('.svg').write_text(svg, encoding='utf-8')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expert-review accuracy</title>
<style>
@page { size: 250mm 120mm; margin: 0; }
body { margin: 0; background: white; }
main { max-width: 1200px; margin: 24px auto; }
svg { display: block; width: 100%; height: auto; }
@media print { main { margin: 0; max-width: none; } }
</style></head><body><main>''' + svg + '</main></body></html>\n'
    html = OUT.with_suffix('.html')
    html.write_text(page, encoding='utf-8')
    # Reuse the installed browser renderer used by the other thesis figures.
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page(viewport={"width": 1200, "height": 520}, device_scale_factor=2)
        tab.goto(html.as_uri(), wait_until='networkidle')
        tab.locator('svg').screenshot(path=str(OUT.with_suffix('.png')))
        tab.pdf(path=str(OUT.with_suffix('.pdf')), prefer_css_page_size=True, print_background=True)
        browser.close()
    for suffix in ('.html', '.svg', '.png', '.pdf'):
        print(f"wrote {OUT.with_suffix(suffix)}")


if __name__ == '__main__':
    main()
