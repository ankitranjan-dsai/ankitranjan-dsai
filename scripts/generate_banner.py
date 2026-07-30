#!/usr/bin/env python3
"""Regenerate the Phase-1 terminal banner for the GitHub profile README.

Portrait pipeline (v2):
  * head-and-shoulders crop, not a tight face crop
  * subject isolated by **warmth** (R-B), not luminance — the photo has a dark,
    non-uniform background (a code editor on the left, blue rim light on the
    right) that a luminance threshold reads as subject; skin is warm, both of
    those are not
  * autocontrast(cutoff=1) + UnsharpMask(radius=3, percent=140), then a mild
    contrast lift with the lit skin rolled down off the ceiling so the face
    keeps its modelling instead of quantising to a solid slab
  * 1-bit **Floyd-Steinberg dither in serpentine order**; all tone comes from
    dot DENSITY at a single hue, never from per-dot radius or a colour ramp
  * dots emitted as <path> runs with shape-rendering="crispEdges"

Animation (SVG only, CSS @keyframes — same technique as the snake SVG that
already animates on this profile's README):
  * intro — dots assemble in ~60 interleaved RANDOM groups so the whole portrait
    thickens at once; grouping by spatial region would read as a patch wipe
  * idle — each group drifts on explicit uneven keyTimes, so no group's motion
    lines up with its neighbour's

The info rows are locked with textLength + lengthAdjust="spacingAndGlyphs" so
right-aligned values stay on their dotted leaders in any browser font.

Usage:  python3 scripts/generate_banner.py        (writes into out/)
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from scipy import ndimage

SCALE = 2
W, H = 1180, 620

# Source photo. NOTE: github-pic.png is not committed to this repo — it lives in
# the workspace one level up. Committing it (or setting BANNER_PHOTO) is what
# makes this script reproducible from a fresh clone.
PHOTO_CANDIDATES = [
    os.environ.get("BANNER_PHOTO", ""),
    "assets/github-pic.png",
    "../assets/github-pic.png",
]
COLS, ROWS_G = 176, 212
HAIR_LO, HAIR_HI = 34.0, 122.0   # density band the hair is rescaled into
CONTRAST, SKIN_GAIN = 1.10, 0.86 # tone curve; see the note in dither_mask()
DOT_HUE = "#79C0FF"
DOT_RGB = (0x79, 0xC0, 0xFF)

# portrait panel geometry (1x coords)
PX, PY, PW, PH = 36, 92, 400, 492
PAD = 16

# Menlo advance width is 0.60245em — used for textLength, so the SVG never
# depends on the viewer having Menlo installed.
EM = 0.60245


# ---------------------------------------------------------------- portrait
def find_photo():
    for p in PHOTO_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise SystemExit("github-pic.png not found; set BANNER_PHOTO=/path/to/photo.png")


def dither_mask():
    """Return a boolean (ROWS_G, COLS) array: True = draw a dot."""
    src = Image.open(find_photo()).convert("RGB")
    w, h = src.size
    # head + shoulders; 0.825 aspect matches the 400x492 panel
    src = src.crop((int(0.10 * w), int(0.02 * h), int(0.90 * w), int(0.99 * h)))

    # --- skin mask from warmth (R-B): skin is warm, background glow is blue
    a = np.asarray(src.resize((COLS, ROWS_G), Image.LANCZOS)).astype(np.float32)
    warm = ndimage.gaussian_filter(a[:, :, 0] - a[:, :, 2], 2.0)
    skin = ndimage.binary_opening(warm > 8, np.ones((5, 5)))
    lab, n = ndimage.label(skin)
    if n:
        skin = lab == int(np.argmax(ndimage.sum(skin, lab, range(1, n + 1)))) + 1
    skin = ndimage.binary_fill_holes(ndimage.binary_closing(skin, np.ones((9, 9))))

    # --- head mask = skin + hair + ears. The hair is nearly black in the source
    # (mean luminance ~7/255), so no tone threshold can find it — but it is still
    # marginally warmer than the blue-lit backdrop, so a much lower warmth cut,
    # taken as the component touching the skin, recovers the silhouette.
    head = ndimage.binary_closing(warm > -2, np.ones((5, 5)))
    lab, n = ndimage.label(head)
    if n:
        overlap = [int(((lab == i) & skin).sum()) for i in range(1, n + 1)]
        head = lab == int(np.argmax(overlap)) + 1
    head = ndimage.binary_fill_holes(ndimage.binary_closing(head, np.ones((7, 7))))
    # hard-clear the mask edge so error diffusion cannot bleed a halo outward
    head = ndimage.binary_erosion(head, np.ones((3, 3)))
    # erosion can shear thin slivers off the silhouette; drop the orphans
    lab, n = ndimage.label(head)
    if n > 1:
        head = lab == int(np.argmax(ndimage.sum(head, lab, range(1, n + 1)))) + 1

    # --- tone prep
    g = ImageOps.autocontrast(src.convert("L"), cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    t = np.asarray(g.resize((COLS, ROWS_G), Image.LANCZOS)).astype(np.float32)
    t = np.clip((t - 128.0) * CONTRAST + 128.0, 0, 255)

    # Roll the lit skin down off the ceiling. At full contrast the whole lit side
    # of the face quantises to solid ink and the modelling disappears; scaling it
    # keeps cheekbone, jaw and eye-socket texture inside the dither.
    t[skin] *= SKIN_GAIN

    # Lift the hair into a visible density band. Left at its true tone it
    # quantises to zero dots and the portrait reads as bald; rescaling its own
    # percentiles into [HAIR_LO, HAIR_HI] keeps the internal texture (lit right
    # side stays denser) while guaranteeing the silhouette survives 1 bit.
    hair = head & ~skin
    if hair.any():
        v = t[hair]
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        span = max(hi - lo, 1.0)
        t[hair] = HAIR_LO + (np.clip(v, lo, hi) - lo) / span * (HAIR_HI - HAIR_LO)

    t[~head] = 0.0                   # background -> black -> no ink
    m = head

    # --- 1-bit Floyd-Steinberg, serpentine
    buf = t.copy()
    out = np.zeros((ROWS_G, COLS), dtype=bool)
    for y in range(ROWS_G):
        rng = range(COLS) if y % 2 == 0 else range(COLS - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in rng:
            old = buf[y, x]
            new = 255.0 if old >= 128.0 else 0.0
            out[y, x] = new == 255.0          # ink on the LIT parts
            err = old - new
            if 0 <= x + d < COLS:
                buf[y, x + d] += err * 7 / 16
            if y + 1 < ROWS_G:
                if 0 <= x - d < COLS:
                    buf[y + 1, x - d] += err * 3 / 16
                buf[y + 1, x] += err * 5 / 16
                if 0 <= x + d < COLS:
                    buf[y + 1, x + d] += err * 1 / 16
    return out & m


def cell_geometry():
    area_x, area_y = PX + PAD, PY + 46
    area_w, area_h = PW - 2 * PAD, PH - 46 - 30
    return area_x, area_y, area_w / COLS, area_h / ROWS_G


def run_buckets(mask, groups=60, seed=7):
    """Horizontal ink runs, split into `groups` interleaved random buckets."""
    ax, ay, cw, ch = cell_geometry()
    rng = random.Random(seed)
    buckets = [[] for _ in range(groups)]
    for y in range(ROWS_G):
        x = 0
        while x < COLS:
            if not mask[y, x]:
                x += 1
                continue
            x0 = x
            while x < COLS and mask[y, x]:
                x += 1
            px, py = ax + x0 * cw, ay + y * ch
            wpx = (x - x0) * cw
            buckets[rng.randrange(groups)].append(
                f"M{px:.2f},{py:.2f}h{wpx:.2f}v{ch:.2f}h{-wpx:.2f}z")
    return buckets


# ---------------------------------------------------------------- animation
# CSS @keyframes, not SMIL. The contribution-snake SVG already on this profile
# animates on GitHub using exactly this technique, which makes it the one
# approach with direct evidence of working in a GitHub README. It is also far
# more compact: 60 one-line rules instead of 120 SMIL elements.
#
# Following the snake's own defensive pattern (.c.c0 sets a static fill AND an
# animation-name), every dot group's resting opacity is 1 and only the keyframe
# carries the 0 — so a renderer that ignores CSS animation shows the finished
# portrait rather than an empty panel.
INTRO_SPAN = 2.0      # last group starts its fade here
INTRO_FADE = 0.55     # per-group fade duration
LOOP_DUR = 14.2       # idle drift cycle
DRIFT_R = 1.6         # drift radius, px
DRIFT_DIRS = 12       # shared direction keyframes

# A staggered fade-in is, by definition, invisible at t=0. That is fine in a
# browser, but a renderer that rasterises the SVG at time zero would show an
# empty portrait panel — and README banners get rasterised by link unfurlers,
# feed readers and social-card scrapers, none of which run CSS animation.
#
# So drift-only is the DEFAULT: its t=0 state is already the finished portrait,
# which makes the worst-case failure a correct still image instead of a blank
# box. Build with BANNER_INTRO=1 to opt into the assemble-on-load intro.
INTRO = os.environ.get("BANNER_INTRO", "0") != "0"


def drift_css():
    out = []
    for k in range(DRIFT_DIRS):
        ang = 6.283185307 * k / DRIFT_DIRS
        ax, ay = DRIFT_R * np.cos(ang), DRIFT_R * np.sin(ang)
        # uneven stops: a long hold, a quick drift out, a slow return
        out.append(
            f"@keyframes k{k}{{0%,100%{{transform:translate(0,0)}}"
            f"21%{{transform:translate({ax:.2f}px,{ay:.2f}px)}}"
            f"35%{{transform:translate({ax * .6:.2f}px,{ay * .6:.2f}px)}}"
            f"72%{{transform:translate({-ax * .5:.2f}px,{-ay * .5:.2f}px)}}}}")
    return "".join(out)


def portrait_group(buckets, animate=True, seed=11):
    rng = random.Random(seed)
    n = len(buckets)
    css, body = [], []
    for i, d in enumerate(buckets):
        if not d:
            continue
        cls = ""
        if animate:
            begin = INTRO_SPAN * (i / max(n - 1, 1))
            k = rng.randrange(DRIFT_DIRS)
            phase = rng.uniform(0, LOOP_DUR)
            cls = f' class="p{i}"'
            intro = f"fi {INTRO_FADE}s {begin:.2f}s both," if INTRO else ""
            css.append(
                f".p{i}{{animation:{intro}"
                f"k{k} {LOOP_DUR}s {INTRO_SPAN + INTRO_FADE + phase:.2f}s "
                f"cubic-bezier(.4,0,.2,1) infinite}}")
        body.append(f'<path{cls} d="{"".join(d)}"/>')
    style = ""
    if animate:
        fi = "@keyframes fi{from{opacity:0}to{opacity:1}}" if INTRO else ""
        style = "<style>" + fi + drift_css() + "".join(css) + "</style>"
    return (style + f'<g shape-rendering="crispEdges" fill="{DOT_HUE}">'
            + "".join(body) + "</g>")


# ---------------------------------------------------------------- content
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


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- SVG
def build_svg(theme, path, mask, animate=True):
    P = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Menlo,Consolas,monospace">']
    P.append(f'<rect x="8" y="8" width="{W-16}" height="{H-16}" rx="14" '
             f'fill="{theme["card"]}" stroke="{theme["border"]}" stroke-width="1.5"/>')
    P.append(f'<path d="M8,22 a14,14 0 0 1 14,-14 h{W-44} a14,14 0 0 1 14,14 v38 h-{W-16} z" '
             f'fill="{theme["titlebar"]}"/>')
    P.append(f'<line x1="8" y1="60" x2="{W-8}" y2="60" stroke="{theme["border"]}"/>')
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        P.append(f'<circle cx="{34 + i * 24}" cy="34" r="7" fill="{c}"/>')
    P.append(f'<text x="{W/2}" y="39" text-anchor="middle" font-size="14" '
             f'fill="{theme["titletext"]}">ankit@dsai: ~/profile.sh --live</text>')

    # portrait panel
    P.append(f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="10" '
             f'fill="{theme["panel"]}" stroke="{theme["panelborder"]}" stroke-width="1.5"/>')
    P.append(f'<text x="{PX+14}" y="{PY+28}" font-size="12" font-weight="bold" '
             f'fill="#7EE787">VISUAL.MAP</text>')
    P.append(f'<text x="{PX+PW-14}" y="{PY+28}" text-anchor="end" font-size="11" '
             f'fill="#5B6470">dither.render</text>')
    P.append(portrait_group(run_buckets(mask), animate=animate))
    P.append(f'<text x="{PX+14}" y="{PY+PH-8}" font-size="10" fill="#5B6470">'
             f'source: github-pic.png &#8212; ok</text>')

    # right column
    RX = PX + PW + 34
    RWIDTH = W - RX - 40
    chip_w = len("SYSTEM.INFO") * 13 * EM + 26
    P.append(f'<rect x="{RX}" y="{PY+2}" width="{chip_w:.1f}" height="30" rx="6" '
             f'fill="{theme["chip"]}"/>')
    P.append(f'<text x="{RX+13}" y="{PY+22}" font-size="13" font-weight="bold" '
             f'fill="{theme["chiptext"]}" textLength="{len("SYSTEM.INFO")*13*EM:.1f}" '
             f'lengthAdjust="spacingAndGlyphs">SYSTEM.INFO</text>')
    P.append(f'<text x="{RX+RWIDTH}" y="{PY+22}" text-anchor="end" font-size="11" '
             f'fill="{theme["dim"]}">pid 2026 &#183; zsh</text>')

    y = PY + 58
    kw_em, vw_em, lw_em = 13 * EM, 14 * EM, 13 * EM
    for k, v in ROWS:
        kw, vw = len(k) * kw_em, len(v) * vw_em
        P.append(f'<text x="{RX}" y="{y+11}" font-size="13" font-weight="bold" '
                 f'fill="{theme["key"]}" textLength="{kw:.1f}" '
                 f'lengthAdjust="spacingAndGlyphs">{esc(k)}</text>')
        lead_start = RX + kw + 14
        lead_end = RX + RWIDTH - vw - 14
        n = max(int((lead_end - lead_start) / lw_em), 0)
        if n:
            P.append(f'<text x="{lead_start:.1f}" y="{y+11}" font-size="13" '
                     f'fill="{theme["leader"]}" textLength="{n*lw_em:.1f}" '
                     f'lengthAdjust="spacingAndGlyphs">{"." * n}</text>')
        P.append(f'<text x="{RX+RWIDTH}" y="{y+11}" text-anchor="end" font-size="14" '
                 f'fill="{theme["value"]}" textLength="{vw:.1f}" '
                 f'lengthAdjust="spacingAndGlyphs">{esc(v)}</text>')
        y += 41

    sy = y + 6
    P.append(f'<line x1="{RX}" y1="{sy-12}" x2="{RX+RWIDTH}" y2="{sy-12}" '
             f'stroke="{theme["border"]}"/>')
    if animate:
        P.append('<style>@keyframes pl{0%,100%{opacity:1}50%{opacity:.25}}'
                 '.pl{animation:pl 2s ease-in-out infinite}</style>')
    P.append(f'<circle class="pl" cx="{RX+7}" cy="{sy+8}" r="5" '
             f'fill="{theme["green"]}"/>')
    P.append(f'<text x="{RX+20}" y="{sy+12}" font-size="12" font-weight="bold" '
             f'fill="{theme["green"]}">open_to_work = True  # AI/ML &#183; Data Science &#183; UK</text>')
    P.append('</svg>')
    open(path, "w").write("\n".join(P))
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")


# ---------------------------------------------------------------- PNG
def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc",
                                  size * SCALE, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default(size * SCALE)


def build_png(theme, path, mask):
    img = Image.new("RGB", (W * SCALE, H * SCALE), theme["bg"])
    d = ImageDraw.Draw(img)
    S = SCALE

    def rect(x, y, w, h, r, fill, outline=None, ow=1):
        d.rounded_rectangle([x*S, y*S, (x+w)*S, (y+h)*S], radius=r*S, fill=fill,
                            outline=outline, width=max(1, int(ow*S)) if outline else 1)

    def text(x, y, s, f, fill, anchor="la"):
        d.text((x*S, y*S), s, font=f, fill=fill, anchor=anchor)

    rect(8, 8, W-16, H-16, 14, theme["card"], theme["border"], 1.5)
    rect(8, 8, W-16, 52, 14, theme["titlebar"])
    d.rectangle([8*S, 40*S, (W-8)*S, 60*S], fill=theme["titlebar"])
    d.line([8*S, 60*S, (W-8)*S, 60*S], fill=theme["border"], width=S)
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        cx, cy, r = (34+i*24)*S, 34*S, 7*S
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=c)
    text(W/2, 35, "ankit@dsai: ~/profile.sh --live", font(14), theme["titletext"], anchor="mm")

    rect(PX, PY, PW, PH, 10, theme["panel"], theme["panelborder"], 1.5)
    text(PX+14, PY+16, "VISUAL.MAP", font(12, True), "#7EE787")
    text(PX+PW-14, PY+16, "dither.render", font(11), "#5B6470", anchor="ra")
    ax, ay, cw, ch = cell_geometry()
    for gy in range(ROWS_G):
        gx = 0
        while gx < COLS:
            if not mask[gy, gx]:
                gx += 1
                continue
            x0 = gx
            while gx < COLS and mask[gy, gx]:
                gx += 1
            d.rectangle([(ax+x0*cw)*S, (ay+gy*ch)*S,
                         (ax+gx*cw)*S, (ay+(gy+1)*ch)*S], fill=DOT_RGB)
    text(PX+14, PY+PH-16, "source: github-pic.png — ok", font(10), "#5B6470")

    RX = PX + PW + 34
    RWIDTH = W - RX - 40
    kf, vf, lf = font(13, True), font(14), font(13)
    cw_chip = d.textlength("SYSTEM.INFO", font=font(13, True))/S + 26
    rect(RX, PY+2, cw_chip, 30, 6, theme["chip"])
    text(RX+13, PY+17, "SYSTEM.INFO", font(13, True), theme["chiptext"], anchor="lm")
    text(RX+RWIDTH, PY+17, "pid 2026 · zsh", font(11), theme["dim"], anchor="rm")
    y = PY + 58
    for k, v in ROWS:
        text(RX, y, k, kf, theme["key"])
        kw = d.textlength(k, font=kf)/S
        vw = d.textlength(v, font=vf)/S
        ls, le = RX+kw+14, RX+RWIDTH-vw-14
        n = max(int((le-ls)/(d.textlength(".", font=lf)/S)), 0)
        if n:
            text(ls, y, "."*n, lf, theme["leader"])
        text(RX+RWIDTH, y, v, vf, theme["value"], anchor="ra")
        y += 41
    sy = y + 6
    d.line([RX*S, (sy-12)*S, (RX+RWIDTH)*S, (sy-12)*S], fill=theme["border"], width=S)
    cx, cy, r = (RX+7)*S, (sy+8)*S, 5*S
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=theme["green"])
    text(RX+20, sy, "open_to_work = True  # AI/ML · Data Science · UK",
         font(12, True), theme["green"])
    img.save(path)
    print(f"wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)  {img.size}")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    m = dither_mask()
    print(f"portrait: {COLS}x{ROWS_G} grid, {m.sum()} dots ({m.mean():.1%} ink)")
    build_svg(DARK,  "out/banner-dark.svg",  m, animate=True)
    build_svg(LIGHT, "out/banner-light.svg", m, animate=True)
    build_png(DARK,  "out/banner-dark.png",  m)
    build_png(LIGHT, "out/banner-light.png", m)
