#!/usr/bin/env python3
"""Regenerate the Phase-1 terminal banner for the GitHub profile README.

- Stipple portrait generated from assets/github-pic.png
- SYSTEM.INFO content sourced from Profile.pdf (page 1)
- Outputs: banner-dark.png / banner-light.png (2x) + matching vector SVGs
"""
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

SCALE = 2
W, H = 1180, 620

# ---------------------------------------------------------------- portrait
def load_dots():
    img = Image.open("assets/github-pic.png").convert("L")
    w, h = img.size
    # crop to head + shoulders (fractions tuned for this photo)
    box = (int(0.215 * w), int(0.015 * h), int(0.825 * w), int(0.785 * h))
    img = img.crop(box)
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.GaussianBlur(1.1))
    COLS, ROWS = 88, 112
    img = img.resize((COLS, ROWS), Image.LANCZOS)
    px = img.load()
    dots = []
    for gy in range(ROWS):
        for gx in range(COLS):
            b = px[gx, gy]
            if b < 14:
                continue
            t = (b / 255.0) ** 1.30
            r = 0.22 + t * 0.74          # fraction of half-cell
            dots.append((gx, gy, t, r))
    return dots, COLS, ROWS

def lerp(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))

DOT_LO = (31, 111, 235)   # #1F6FEB
DOT_HI = (165, 214, 255)  # #A5D6FF

# ---------------------------------------------------------------- fonts
def font(size, bold=False):
    path = "/System/Library/Fonts/Menlo.ttc"
    try:
        return ImageFont.truetype(path, size * SCALE, index=1 if bold else 0)
    except Exception:
        return ImageFont.truetype(path, size * SCALE)

# ---------------------------------------------------------------- themes
DARK = dict(bg="#070B12", card="#0D1117", border="#30363D",
            titlebar="#161B22", titletext="#8B949E",
            panel="#05080E", panelborder="#1E2A3A",
            key="#79C0FF", value="#E6EDF3", leader="#2D333B",
            chip="#1F6FEB", chiptext="#FFFFFF", green="#3FB950", dim="#8B949E")
LIGHT = dict(bg="#DDE3EA", card="#FFFFFF", border="#C9D1D9",
             titlebar="#EFF1F4", titletext="#57606A",
             panel="#0B1220", panelborder="#0B1220",
             key="#0969DA", value="#1F2328", leader="#C6CDD3",
             chip="#0969DA", chiptext="#FFFFFF", green="#1A7F37", dim="#57606A")

ROWS = [
    ("SUBJECT",   "Ankit Ranjan"),
    ("ROLE",      "Data Science & AI Engineer"),
    ("BASE",      "Liverpool, England, UK"),
    ("EDUCATION", "MSc Data Science & AI - University of Liverpool"),
    ("TERM",      "Jan 2026 - Jan 2027"),
    ("FOCUS",     "Machine Learning - NLP - PyTorch - Statistics"),
    ("CERTS",     "4x MIT edX - Probability - Stats - Data Analysis - ML"),
    ("MAIL",      "ankit0ranjan@gmail.com"),
    ("LINKEDIN",  "/in/ankit-ranjan-datascience"),
    ("GITHUB",    "/ankitranjan-dsai"),
]

# portrait panel geometry (1x coords)
PX, PY, PW, PH = 36, 92, 400, 492
PAD = 16

# ---------------------------------------------------------------- PNG
def build_png(theme, path):
    img = Image.new("RGB", (W * SCALE, H * SCALE), theme["bg"])
    d = ImageDraw.Draw(img)

    def rect(x, y, w, h, r, fill, outline=None, ow=1):
        d.rounded_rectangle([x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE],
                            radius=r * SCALE, fill=fill,
                            outline=outline, width=max(1, int(ow * SCALE)) if outline else 1)

    def text(x, y, s, f, fill, anchor="la"):
        d.text((x * SCALE, y * SCALE), s, font=f, fill=fill, anchor=anchor)

    def tw(s, f):
        return d.textlength(s, font=f) / SCALE

    # card
    rect(8, 8, W - 16, H - 16, 14, theme["card"], theme["border"], 1.5)
    # title bar
    rect(8, 8, W - 16, 52, 14, theme["titlebar"])
    d.rectangle([8 * SCALE, 40 * SCALE, (W - 8) * SCALE, 60 * SCALE], fill=theme["titlebar"])
    d.line([8 * SCALE, 60 * SCALE, (W - 8) * SCALE, 60 * SCALE], fill=theme["border"], width=SCALE)
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        cx, cy, r = (34 + i * 24) * SCALE, 34 * SCALE, 7 * SCALE
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    text(W / 2, 35, "ankit@dsai: ~/profile.sh --live", font(14), theme["titletext"], anchor="mm")

    # portrait panel (dark "screen" on both themes)
    rect(PX, PY, PW, PH, 10, theme["panel"], theme["panelborder"], 1.5)
    text(PX + 14, PY + 16, "VISUAL.MAP", font(12, bold=True), "#7EE787")
    text(PX + PW - 14, PY + 16, "stipple.render", font(11), "#5B6470", anchor="ra")

    dots, COLS, ROWS_G = load_dots()
    area_x, area_y = PX + PAD, PY + 46
    area_w, area_h = PW - 2 * PAD, PH - 46 - 30
    cw, ch = area_w / COLS, area_h / ROWS_G
    half = min(cw, ch) / 2
    for gx, gy, t, r in dots:
        cx = (area_x + gx * cw + cw / 2) * SCALE
        cy = (area_y + gy * ch + ch / 2) * SCALE
        rr = max(r * half * SCALE, 0.4)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=lerp(DOT_LO, DOT_HI, t))
    text(PX + 14, PY + PH - 16, "source: github-pic.png — ok", font(10), "#5B6470")

    # right column
    RX = PX + PW + 34
    RWIDTH = W - RX - 40
    chip_w = tw("SYSTEM.INFO", font(13, bold=True)) + 26
    rect(RX, PY + 2, chip_w, 30, 6, theme["chip"])
    text(RX + 13, PY + 17, "SYSTEM.INFO", font(13, bold=True), theme["chiptext"], anchor="lm")
    text(RX + RWIDTH, PY + 17, "pid 2026 · zsh", font(11), theme["dim"], anchor="rm")

    key_f, val_f, lead_f = font(13, bold=True), font(14), font(13)
    y = PY + 58
    lh = 41
    for k, v in ROWS:
        text(RX, y, k, key_f, theme["key"])
        kw = tw(k, key_f)
        vw = tw(v, val_f)
        leader_start = RX + kw + 14
        leader_end = RX + RWIDTH - vw - 14
        n_dots = max(int((leader_end - leader_start) / tw(".", lead_f)), 0)
        if n_dots:
            text(leader_start, y, "." * n_dots, lead_f, theme["leader"])
        text(RX + RWIDTH, y, v, val_f, theme["value"], anchor="ra")
        y += lh

    # status line
    sy = y + 6
    d.line([RX * SCALE, (sy - 12) * SCALE, (RX + RWIDTH) * SCALE, (sy - 12) * SCALE],
           fill=theme["border"], width=SCALE)
    cx, cy, r = (RX + 7) * SCALE, (sy + 8) * SCALE, 5 * SCALE
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=theme["green"])
    text(RX + 20, sy, "open_to_work = True  # AI/ML · Data Science · UK",
         font(12, bold=True), theme["green"])

    img.save(path)
    print("wrote", path, img.size)

# ---------------------------------------------------------------- SVG
def svg_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_svg(theme, path):
    P = []
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    P.append(f'<rect x="8" y="8" width="{W-16}" height="{H-16}" rx="14" fill="{theme["card"]}" stroke="{theme["border"]}" stroke-width="1.5"/>')
    P.append(f'<path d="M8,22 a14,14 0 0 1 14,-14 h{W-44} a14,14 0 0 1 14,14 v38 h-{W-16} z" fill="{theme["titlebar"]}"/>')
    P.append(f'<line x1="8" y1="60" x2="{W-8}" y2="60" stroke="{theme["border"]}"/>')
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        P.append(f'<circle cx="{34 + i * 24}" cy="34" r="7" fill="{c}"/>')
    P.append(f'<text x="{W/2}" y="39" text-anchor="middle" font-family="Menlo,monospace" font-size="14" fill="{theme["titletext"]}">ankit@dsai: ~/profile.sh --live</text>')

    P.append(f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="10" fill="{theme["panel"]}" stroke="{theme["panelborder"]}" stroke-width="1.5"/>')
    P.append(f'<text x="{PX+14}" y="{PY+28}" font-family="Menlo,monospace" font-size="12" font-weight="bold" fill="#7EE787">VISUAL.MAP</text>')
    P.append(f'<text x="{PX+PW-14}" y="{PY+28}" text-anchor="end" font-family="Menlo,monospace" font-size="11" fill="#5B6470">stipple.render</text>')

    dots, COLS, ROWS_G = load_dots()
    area_x, area_y = PX + PAD, PY + 46
    area_w, area_h = PW - 2 * PAD, PH - 46 - 30
    cw, ch = area_w / COLS, area_h / ROWS_G
    half = min(cw, ch) / 2
    P.append('<g>')
    for gx, gy, t, r in dots:
        cx = area_x + gx * cw + cw / 2
        cy = area_y + gy * ch + ch / 2
        rr = max(r * half, 0.2)
        col = lerp(DOT_LO, DOT_HI, t)
        P.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr:.2f}" fill="#{col[0]:02X}{col[1]:02X}{col[2]:02X}"/>')
    P.append('</g>')
    P.append(f'<text x="{PX+14}" y="{PY+PH-8}" font-family="Menlo,monospace" font-size="10" fill="#5B6470">source: github-pic.png — ok</text>')

    RX = PX + PW + 34
    RWIDTH = W - RX - 40
    P.append(f'<rect x="{RX}" y="{PY+2}" width="118" height="30" rx="6" fill="{theme["chip"]}"/>')
    P.append(f'<text x="{RX+13}" y="{PY+22}" font-family="Menlo,monospace" font-size="13" font-weight="bold" fill="{theme["chiptext"]}">SYSTEM.INFO</text>')
    P.append(f'<text x="{RX+RWIDTH}" y="{PY+22}" text-anchor="end" font-family="Menlo,monospace" font-size="11" fill="{theme["dim"]}">pid 2026 · zsh</text>')

    y = PY + 58
    lh = 41
    char_w = 8.43   # Menlo 14px ≈ 0.602em; leader uses 13px ≈ 7.8
    for k, v in ROWS:
        P.append(f'<text x="{RX}" y="{y+11}" font-family="Menlo,monospace" font-size="13" font-weight="bold" fill="{theme["key"]}">{svg_escape(k)}</text>')
        kw = len(k) * 7.8
        vw = len(v) * 8.43
        leader_start = RX + kw + 14
        leader_end = RX + RWIDTH - vw - 14
        n = max(int((leader_end - leader_start) / 7.8), 0)
        if n:
            P.append(f'<text x="{leader_start}" y="{y+11}" font-family="Menlo,monospace" font-size="13" fill="{theme["leader"]}">{"." * n}</text>')
        P.append(f'<text x="{RX+RWIDTH}" y="{y+11}" text-anchor="end" font-family="Menlo,monospace" font-size="14" fill="{theme["value"]}">{svg_escape(v)}</text>')
        y += lh

    sy = y + 6
    P.append(f'<line x1="{RX}" y1="{sy-12}" x2="{RX+RWIDTH}" y2="{sy-12}" stroke="{theme["border"]}"/>')
    P.append(f'<circle cx="{RX+7}" cy="{sy+8}" r="5" fill="{theme["green"]}"/>')
    P.append(f'<text x="{RX+20}" y="{sy+12}" font-family="Menlo,monospace" font-size="12" font-weight="bold" fill="{theme["green"]}">open_to_work = True  # AI/ML · Data Science · UK</text>')

    P.append('</svg>')
    open(path, "w").write("\n".join(P))
    print("wrote", path)

if __name__ == "__main__":
    build_png(DARK, "out/banner-dark.png")
    build_png(LIGHT, "out/banner-light.png")
    build_svg(DARK, "out/banner-dark.svg")
    build_svg(LIGHT, "out/banner-light.svg")
