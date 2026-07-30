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
  * morph — 1200 of the dots detach and run PORTRAIT -> "ANKIT RANJAN" -> GLOBE
    -> portrait on a 24s loop, while the face itself dims to 7% for the
    excursion. The portrait is the resting state and holds 62% of the cycle

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
DOT_LIT = "#D6ECFF"   # travelling dots lift to this while away from the face

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


# ---------------------------------------------------------------- morph
# A subset of the portrait's dots detaches and runs a three-beat sequence —
# PORTRAIT, the name "ANKIT RANJAN", a GLOBE — then flies home. Three choices
# here are load-bearing, not stylistic:
#
#  * The portrait is the RESTING state, not something assembled from nothing.
#    A build-from-empty would make the t=0 frame a blank panel, and README
#    banners get rasterised at t=0 by link unfurlers, feed readers and
#    social-card scrapers, none of which run CSS. The running belongs in the
#    transitions; the still frame has to be the finished picture.
#  * The portrait holds ~62% of the cycle, and the first 9.6s after load are
#    pure portrait. Visitors glance for seconds, so the face must be what they
#    almost certainly see; the sequence is a reward for lingering.
#  * Both targets are sampled to the SAME count, because one pool of dots forms
#    both. That constraint is why the name is a 2-cell stroke (1850 cells) and
#    the globe a 1-cell stroke (1630) — they land within 10% of each other, so
#    neither has to be violently thinned to match.
# The beats are a LIST, so the sequence is data rather than a hardcoded pair.
# Available targets are in TARGETS below; BANNER_BEATS overrides the default,
# e.g. BANNER_BEATS=name,globe,mono adds the AR monogram as a third beat, and
# BANNER_BEATS=mono runs the monogram alone. The loop length is derived from the
# beat count (see morph_timeline) so the portrait keeps its share whatever you
# pick. BANNER_MORPH=0 falls back to drift-only.
MORPH = os.environ.get("BANNER_MORPH", "1") != "0"
BEATS = tuple(b for b in os.environ.get("BANNER_BEATS", "name,globe").split(",")
              if b)
MORPH_N = 1200          # travellers, and the count every target samples to
GW, GH = 22, 34         # glyph box, in grid cells
GLYPH_CH = 4            # chamfer size — the motif that unifies the letterforms
NAME_STROKE = 2         # name monoline width, in cells
GLOBE_STROKE = 1        # globe monoline width, in cells
NAME_LINES = ("ANKIT", "RANJAN")
LIVERPOOL = (-3.0, 53.4)   # lon, lat

MONO_W, MONO_H = 152, 108   # AR monogram box, in grid cells
MONO_T, MONO_B = 5, 102     # cap height within that box
MONO_CH = 7                 # the monogram's own, larger chamfer


def _glyphs():
    """Monoline letterforms for the glyphs in "ANKIT RANJAN": A N K I T R J.

    Constructed vertex by vertex rather than set in a font, so the banner never
    depends on the viewer having a particular face installed, and so a repeated
    45-degree CHAMFER motif can run through every glyph — the N's shoulders,
    the A's apex, the R's bowl. That one repeated cut is what makes seven
    hand-built letters look like a single family.

    Monoline rather than filled or outlined: at the panel's real width (~370px
    in a README) an outlined letter puts two edges within a couple of pixels and
    turns to mush, and a filled one costs three times the dots.
    """
    H, C = GH, GLYPH_CH
    return {
        "A": [[(1, H), (9, 3), (13, 3), (21, H)],          # chamfered apex
              [(5, 23), (17, 23)]],                        # crossbar
        "N": [[(1, H), (1, C), (1 + C, 0), (21 - C, H), (21, H - C), (21, 0)]],
        "K": [[(1, 0), (1, H)],
              [(20, 0), (16, 0), (3, 17)],                 # upper arm
              [(3, 17), (16, H), (20, H)]],                # lower leg
        "I": [[(11, 0), (11, H)],
              [(6, 0), (16, 0)], [(6, H), (16, H)]],       # serifs
        "T": [[(1, 0), (21, 0)], [(11, 0), (11, H)]],
        "R": [[(1, H), (1, 0), (15, 0), (20, 5), (20, 12), (15, 17), (1, 17)],
              [(9, 17), (20, H)]],                         # kicked leg
        "J": [[(9, 0), (21, 0)],                           # top bar
              [(15, 0), (15, H - 7), (11, H), (5, H), (1, H - 5)]],
    }


def _plot(d, pts, thick):
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) > 1:
        d.line(pts, fill=255, width=thick, joint="curve")


def name_mask(lines=NAME_LINES, gap=6, leading=14, thick=NAME_STROKE):
    """Boolean (ROWS_G, COLS) mask of the name, set as a justified block.

    Two details do the work. First, each glyph advances by its own ink width
    plus a constant gap: with a fixed 22-cell slot the narrow letters carried
    11 cells of sidebearing where A carried 7, and "ANKIT" read as "ANK I T".
    Second, the lines are JUSTIFIED to a common width rather than centred —
    "ANKIT" is tracked out until it measures the same as "RANJAN", so the two
    words form a rectangle. Centred words read as typed; a flush block reads as
    a logotype, which is the difference between a caption and a mark.
    """
    G = _glyphs()
    ink = {ch: (min(p[0] for poly in polys for p in poly),
                max(p[0] for poly in polys for p in poly))
           for ch, polys in G.items()}
    widths = {ln: [ink[c][1] - ink[c][0] for c in ln] for ln in lines}
    target = max(sum(w) + gap * (len(w) - 1) for w in widths.values())
    x0 = (COLS - target) // 2
    y0 = (ROWS_G - (len(lines) * GH + (len(lines) - 1) * leading)) // 2
    im = Image.new("L", (COLS, ROWS_G), 0)
    d = ImageDraw.Draw(im)
    for li, line in enumerate(lines):
        w = widths[line]
        track = (target - sum(w)) / max(len(w) - 1, 1)
        ty, pen = y0 + li * (GH + leading), float(x0)
        for ch, cw in zip(line, w):
            tx = pen - ink[ch][0]
            for poly in G[ch]:
                _plot(d, [(tx + px, ty + py) for px, py in poly], thick)
            pen += cw + track
    return np.asarray(im) > 128


def globe_mask(r=62, tilt=0.40, thick=GLOBE_STROKE):
    """Boolean (ROWS_G, COLS) wireframe globe with a Liverpool marker.

    A bare dot-globe is one of the most overused motifs on dev profiles, so
    this one is specific to Ankit: it pins Liverpool, where he actually is, and
    carries a tilted orbit ring that breaks the silhouette out of a plain
    circle.

    Deliberately sparse — 3 latitudes, 4 longitudes. A denser mesh looked
    better at full resolution, but the globe and the name share one pool of
    dots, and a dense globe thinned to the name's count fell apart into
    confetti.
    """
    cx, cy = COLS / 2, ROWS_G / 2
    im = Image.new("L", (COLS, ROWS_G), 0)
    d = ImageDraw.Draw(im)
    t = np.linspace(0, 2 * np.pi, 400)

    def project(lon, lat):
        """Sphere -> screen, rotated by `tilt` about the screen x-axis."""
        x = np.cos(lat) * np.sin(lon)
        y, z = np.sin(lat), np.cos(lat) * np.cos(lon)
        ys = y * np.cos(tilt) - z * np.sin(tilt)
        zs = y * np.sin(tilt) + z * np.cos(tilt)
        return cx + r * x, cy - r * ys, zs        # zs>0 = facing the viewer

    def culled(lon, lat):
        """Draw a parametric curve, broken wherever it passes behind the globe."""
        X, Y, Z = project(lon, lat)
        seg = []
        for x, y, z in zip(X, Y, Z):
            if z >= 0:
                seg.append((x, y))
            else:
                _plot(d, seg, thick)
                seg = []
        _plot(d, seg, thick)

    _plot(d, zip(cx + r * np.cos(t), cy + r * np.sin(t)), thick)      # limb
    for lat in np.radians([-38, 0, 38]):
        culled(t, np.full_like(t, lat))
    lat_arc = np.linspace(-np.pi / 2, np.pi / 2, 220)
    for lon in np.radians([-60, -20, 20, 60]):
        culled(np.full_like(lat_arc, lon), lat_arc)

    a, b = r * 1.34, r * 0.30                                        # orbit
    ang = np.radians(-18)
    ex, ey = a * np.cos(t), b * np.sin(t)
    _plot(d, zip(cx + ex * np.cos(ang) - ey * np.sin(ang),
                 cy + ex * np.sin(ang) + ey * np.cos(ang)), thick)

    mesh = np.asarray(im) > 128

    # Liverpool. Two failed attempts are worth recording. A crosshair drawn
    # straight onto the mesh was invisible, because the lat/long lines run right
    # through it — hence the cleared disc. A leader line and label tick then ran
    # back OVER the mesh outside that disc and were camouflaged again. So the
    # whole marker now lives inside the clearance: ring, centre dot, and four
    # diagonal registration ticks. Self-contained means it always reads.
    lx, ly, _ = project(np.radians(LIVERPOOL[0]), np.radians(LIVERPOOL[1]))
    lx, ly = float(lx), float(ly)
    yy, xx = np.mgrid[0:ROWS_G, 0:COLS]
    mesh &= ~(np.hypot(xx - lx, yy - ly) < 13)

    pin = Image.new("L", (COLS, ROWS_G), 0)
    p = ImageDraw.Draw(pin)
    p.ellipse([lx - 2, ly - 2, lx + 2, ly + 2], fill=255)
    p.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], outline=255, width=thick)
    for a in np.radians([45, 135, 225, 315]):
        ca, sa = np.cos(a), np.sin(a)
        _plot(p, [(lx + 9 * ca, ly + 9 * sa), (lx + 12 * ca, ly + 12 * sa)],
              thick)
    return mesh | (np.asarray(pin) > 128)


def monogram_mask():
    """Boolean (ROWS_G, COLS) mask of an interlocked "AR" monogram, outline only.

    RETAINED BUT NOT IN THE DEFAULT SEQUENCE. This was the first design tried,
    and it was rejected on looks; the name-and-globe sequence replaced it. It is
    kept because the letterform construction is reusable and the study cost real
    iteration — switch it on with BANNER_BEATS=name,globe,mono.

    Two findings from that iteration are worth keeping with the code. An earlier
    version fused the A's right stroke into the R's stem as a ligature, and at
    the panel's real width it read as one slab, so the letters are separated and
    unified by the repeated 45-degree chamfer instead. And a FILLED monogram read
    as a mushy slab at this dot count, so only the outline is used.

    Note it yields ~1080 cells, fewer than MORPH_N, so including this beat pulls
    the traveller count down for every beat — they all share one pool of dots.
    """
    T, B, C = MONO_T, MONO_B, MONO_CH
    im = Image.new("L", (MONO_W, MONO_H), 0)
    d = ImageDraw.Draw(im)
    for p in (
        # A — two diagonal bands sharing one flat-chamfered apex, plus a slab
        # crossbar. The counter is simply the space the two bands leave; cutting
        # a separate counter polygon made it read as damaged.
        [(2, B), (16, B), (50, T), (36, T)],
        [(70, B), (84, B), (50, T), (36, T)],
        [(22, 68), (64, 68), (64, 80), (22, 80)],
        # R — stem, chamfered bowl, kicked leg
        [(92, T), (106, T), (106, B), (92, B)],
        [(106, T), (130, T), (146, 20), (146, 38), (130, 52), (106, 52)],
        [(110, 50), (124, 50), (146, B), (132, B)],
    ):
        d.polygon(p, fill=255)
    for p in (
        [(106, 17), (126, 17), (132, 24), (132, 34), (126, 41), (106, 41)],
        # the chamfer motif, applied to every remaining square terminal
        [(2, B - C), (2, B), (2 + C, B)],
        [(84, B - C), (84, B), (84 - C, B)],
        [(92, T + C), (92, T), (92 + C, T)],
        [(92, B - C), (92, B), (92 + C, B)],
        [(146, B - C), (146, B), (146 - C, B)],
    ):
        d.polygon(p, fill=0)
    solid = np.asarray(im) > 128
    outline = solid & ~ndimage.binary_erosion(solid, np.ones((3, 3)))
    out = np.zeros((ROWS_G, COLS), dtype=bool)
    ox, oy = (COLS - MONO_W) // 2, (ROWS_G - MONO_H) // 2
    out[oy:oy + MONO_H, ox:ox + MONO_W] = outline
    return out


TARGETS = {"name": name_mask, "globe": globe_mask, "mono": monogram_mask}


def _sample(mask, n, seed):
    """Thin a mask to exactly n cells."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), size=n, replace=False)]


def _angular_order(pts):
    """Sort points by angle about their centroid, then radius.

    Pairing all three sets in this order keeps every morph radially coherent —
    dots sweep outward together instead of crossing into a scribble. Because
    each set is ranked the same way, src[i], name[i] and globe[i] all hold the
    same angular rank, so BOTH transitions inherit the coherence, not just the
    first one.
    """
    d = pts - pts.mean(axis=0)
    return np.lexsort((np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])))


def travellers(mask, seed=17):
    """Pair a sample of portrait dots with each beat's target cells.

    Returns (home, *beat_targets) aligned row-for-row, or None when morphing is
    switched off. Every set is the same length, because one pool of dots forms
    all of them: the count is the smallest beat, capped at MORPH_N.
    """
    if not (MORPH and BEATS):
        return None
    unknown = [b for b in BEATS if b not in TARGETS]
    if unknown:
        raise SystemExit(f"unknown beat(s) {unknown}; choose from "
                         f"{sorted(TARGETS)}")
    full = [TARGETS[b]() for b in BEATS]
    ys, xs = np.nonzero(mask)
    src_all = np.stack([xs, ys], axis=1)
    n = min([MORPH_N, len(src_all)] + [int(m.sum()) for m in full])
    rng = np.random.default_rng(seed)
    src = src_all[rng.choice(len(src_all), size=n, replace=False)]
    sets = [src] + [_sample(m, n, 101 + i) for i, m in enumerate(full)]
    return tuple(s[_angular_order(s)] for s in sets)


# Timings in SECONDS, from which the percentage stops are derived. Holding these
# absolute and letting the LOOP LENGTH grow with the beat count is what keeps the
# portrait's share constant: adding a third beat lengthens the loop to ~34s
# rather than squeezing the face down to 46% of it.
FLY_S, HOLD_S = 1.44, 2.40
REST_FRAC = 0.62       # share of the loop the portrait is at rest
HEAD_SHARE = 0.65      # of that rest, how much falls BEFORE the first flight,
                       # so the first impression after load is a long portrait
MORPH_STAGGER = 0.6    # s; well under one hold, so each beat still settles
MORPH_LANES = 8        # shared delay classes


def morph_timeline(k):
    """Stops for k beats: (duration_s, out%, [(in%, out%)...], home%)."""
    excursion = k * (FLY_S + HOLD_S) + FLY_S
    dur = excursion / (1.0 - REST_FRAC)
    head = (dur - excursion) * HEAD_SHARE
    pc, t, beats = lambda s: 100.0 * s / dur, head, []
    for _ in range(k):
        t += FLY_S
        a = t
        t += HOLD_S
        beats.append((pc(a), pc(t)))
    return dur, pc(head), beats, pc(t + FLY_S)


def traveller_layer(src, *beats):
    """One shared keyframe; per-dot deltas ride in CSS custom properties.

    A keyframe per dot would be 1200 rules and megabytes. Instead each beat's
    delta is a static per-element value substituted into one shared transform,
    so exact landing positions survive — a quantised delta would blur the
    letterform, which is the entire point of the layer.

    Three economies keep this affordable at 1200 dots: the deltas are unitless
    and get their px in the keyframe's calc(), so each value drops two
    characters; the stagger lives in eight shared lane classes instead of a
    per-element animation-delay; and the dot itself is a <defs> rect reused by
    <use>, which is the shortest per-element markup available.

    Degradation is deliberate. A renderer that ignores CSS animation, or that
    does not support var(), leaves every dot at its home cell — the finished
    portrait. Nothing here is hidden at t=0: no opacity 0, no inline transform.
    """
    ax, ay, cw, ch = cell_geometry()
    rng = random.Random(23)
    dur, t_out, holds, t_home = morph_timeline(len(beats))
    prop = lambda i: chr(ord("a") + i)      # beat i uses --{2i} and --{2i+1}

    def stops(fmt, home, values):
        s = f"0%,{t_out:.1f}%{{{fmt.format(home)}}}"
        for (a, b), v in zip(holds, values):
            s += f"{a:.1f}%,{b:.1f}%{{{fmt.format(v)}}}"
        return s + f"{t_home:.1f}%,100%{{{fmt.format(home)}}}"

    css = ["@keyframes fly{" + stops("transform:{}", "translate(0,0)", [
        f"translate(calc(var(--{prop(2 * i)})*1px),"
        f"calc(var(--{prop(2 * i + 1)})*1px))" for i in range(len(beats))]) + "}"]
    # The travellers lift to near-white for the whole excursion, including the
    # crossings between beats. Measured, not guessed: a wireframe in the portrait
    # hue over the portrait is simply invisible.
    css.append(f"@keyframes glow{{0%,{t_out:.1f}%{{fill:{DOT_HUE}}}"
               f"{holds[0][0]:.1f}%,{holds[-1][1]:.1f}%{{fill:{DOT_LIT}}}"
               f"{t_home:.1f}%,100%{{fill:{DOT_HUE}}}}}")
    for i in range(MORPH_LANES):
        delay = MORPH_STAGGER * i / max(MORPH_LANES - 1, 1)
        css.append(f".s{i}{{animation:"
                   f"fly {dur:.1f}s cubic-bezier(.5,0,.2,1) {delay:.2f}s infinite,"
                   f"glow {dur:.1f}s ease-in-out {delay:.2f}s infinite}}")

    out = [f'<defs><rect id="d" width="{cw:.2f}" height="{ch:.2f}"/></defs>',
           "<style>" + "".join(css) + "</style>",
           f'<g fill="{DOT_HUE}" shape-rendering="crispEdges">']
    for i, (sx, sy) in enumerate(src):
        deltas = "".join(
            f"--{prop(2 * b)}:{(int(t[i][0]) - int(sx)) * cw:.0f};"
            f"--{prop(2 * b + 1)}:{(int(t[i][1]) - int(sy)) * ch:.0f};"
            for b, t in enumerate(beats))
        out.append(
            f'<use href="#d" class="s{rng.randrange(MORPH_LANES)}" '
            f'x="{ax + sx * cw:.1f}" y="{ay + sy * ch:.1f}" '
            f'style="{deltas[:-1]}"/>')
    out.append("</g>")
    return "".join(out)


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


# A wireframe laid over an 18%-ink portrait in the same hue is simply invisible
# — measured, not guessed: the first build of this had the target shape present
# and unreadable. So the portrait dips almost to black for the whole excursion
# and comes back after. It stays down across the name-to-globe crossing too;
# flashing the face back between beats broke the sequence into two unrelated
# events. One rule on the parent <g> dims all 60 groups at once, and the
# travelling dots live in a separate <g>, so they keep full opacity.
MORPH_DIM = 0.07


def dim_css(k):
    dur, t_out, holds, t_home = morph_timeline(k)
    return (f"@keyframes dim{{0%,{t_out:.1f}%{{opacity:1}}"
            f"{holds[0][0]:.1f}%,{holds[-1][1]:.1f}%{{opacity:{MORPH_DIM}}}"
            f"{t_home:.1f}%,100%{{opacity:1}}}}"
            f".pf{{animation:dim {dur:.1f}s ease-in-out infinite}}")


def portrait_group(buckets, animate=True, dim=0, seed=11):
    """`dim` is the BEAT COUNT (0 = no morph), since the dim keyframe's stops
    are derived from it."""
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
        style = ("<style>" + fi + drift_css() + (dim_css(dim) if dim else "")
                 + "".join(css) + "</style>")
    cls = ' class="pf"' if animate and dim else ""
    return (style + f'<g{cls} shape-rendering="crispEdges" fill="{DOT_HUE}">'
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
    # Travelling dots are drawn individually so they can each carry their own
    # deltas, so they must come OUT of the merged-run layer or they'd be drawn
    # twice. At rest they sit at their home cells, so the portrait is whole and
    # the traveller count costs the face nothing — only file size.
    tr = travellers(mask) if animate else None
    static = mask
    if tr:
        src = tr[0]
        static = mask.copy()
        static[src[:, 1], src[:, 0]] = False
    P.append(portrait_group(run_buckets(static), animate=animate,
                            dim=len(tr) - 1 if tr else 0))
    if tr:
        P.append(traveller_layer(*tr))
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
