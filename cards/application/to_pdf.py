import pathlib
from playwright.sync_api import sync_playwright

src = pathlib.Path("descriptive_stats.html").resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 1400})
    pg.goto(src, wait_until="networkidle")
    pg.wait_for_timeout(3000)          # let Plotly finish drawing
    pg.pdf(path="descriptive_stats.pdf", format="A4", print_background=True,
           margin={"top": "12mm", "bottom": "12mm", "left": "8mm", "right": "8mm"})
    b.close()
print("wrote descriptive_stats.pdf")
