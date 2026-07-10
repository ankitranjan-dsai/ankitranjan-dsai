#!/usr/bin/env python3
"""Generate assets/contribution-graph.svg — Ankit's Contribution Graph (last 6 months).

Uses GitHub's own GraphQL API (authenticated via GITHUB_TOKEN / GH_TOKEN).
Writes the SVG only on full success, so a failed run can never corrupt the asset.
"""
import json, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

LOGIN = "ankitranjan-dsai"
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "contribution-graph.svg")
MONTHS_BACK_DAYS = 182  # ~6 months

token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
if not token:
    sys.exit("error: GITHUB_TOKEN / GH_TOKEN not set")

now = datetime.now(timezone.utc)
frm = now - timedelta(days=MONTHS_BACK_DAYS)
query = """
query($login:String!, $from:DateTime!, $to:DateTime!){
  user(login:$login){
    contributionsCollection(from:$from, to:$to){
      contributionCalendar{
        totalContributions
        weeks{ contributionDays{ date contributionCount weekday } }
      }
    }
  }
}"""
body = json.dumps({"query": query, "variables": {
    "login": LOGIN, "from": frm.isoformat(), "to": now.isoformat()}}).encode()
req = urllib.request.Request(
    "https://api.github.com/graphql", data=body,
    headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)
cal = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]
weeks, total = cal["weeks"], cal["totalContributions"]

# --- theme ---
BG, BG2, BORDER = "#0d1117", "#161b2e", "#30363d"
FG, MUT = "#f0f6fc", "#8b949e"
LEVELS = ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353"]
def level(c): return 0 if c == 0 else 1 if c <= 2 else 2 if c <= 5 else 3 if c <= 9 else 4

CELL, GAP = 24, 6
STEP = CELL + GAP
LEFT, TOP = 74, 108
W = LEFT + len(weeks) * STEP - GAP + 34
H = TOP + 7 * STEP - GAP + 64

cells, month_labels, prev_month = [], [], None
for wi, week in enumerate(weeks):
    days = week["contributionDays"]
    if days:
        m = datetime.fromisoformat(days[0]["date"]).strftime("%b")
        if m != prev_month:
            if wi > 0 or True:
                month_labels.append((LEFT + wi * STEP, m))
            prev_month = m
    for d in days:
        x, y = LEFT + wi * STEP, TOP + d["weekday"] * STEP
        cells.append(
            f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="5" '
            f'fill="{LEVELS[level(d["contributionCount"])]}"><title>{d["date"]}: '
            f'{d["contributionCount"]} contributions</title></rect>')

# drop a month label if it collides with the previous one
pruned, last_x = [], -1e9
for x, m in month_labels:
    if x - last_x >= 44:
        pruned.append((x, m)); last_x = x

svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
       f"font-family=\"'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif\">",
       '<defs><linearGradient id="cbg" x1="0" y1="0" x2="1" y2="1">'
       f'<stop offset="0%" stop-color="{BG}"/><stop offset="100%" stop-color="{BG2}"/>'
       '</linearGradient></defs>',
       f'<rect width="{W}" height="{H}" rx="18" fill="url(#cbg)" stroke="{BORDER}"/>',
       f'<circle cx="40" cy="42" r="6" fill="#7ee787"/>',
       f'<text x="58" y="48" fill="{FG}" font-size="22" font-weight="800">Ankit&#8217;s Contribution Graph</text>',
       f'<text x="58" y="72" fill="{MUT}" font-size="14">{total} contributions in the last 6 months</text>']
for x, m in pruned:
    svg.append(f'<text x="{x}" y="{TOP-14}" fill="{MUT}" font-size="13" font-weight="600">{m}</text>')
for row, lbl in ((1, "Mon"), (3, "Wed"), (5, "Fri")):
    svg.append(f'<text x="{LEFT-10}" y="{TOP + row*STEP + 16}" fill="{MUT}" '
               f'font-size="12" text-anchor="end">{lbl}</text>')
svg += cells
ly = TOP + 7 * STEP - GAP + 34
lx = W - 34 - 5 * STEP - 88
svg.append(f'<text x="{lx-10}" y="{ly+16}" fill="{MUT}" font-size="12" text-anchor="end">Less</text>')
for i, c in enumerate(LEVELS):
    svg.append(f'<rect x="{lx + i*STEP}" y="{ly}" width="{CELL}" height="{CELL}" rx="5" fill="{c}"/>')
svg.append(f'<text x="{lx + 5*STEP + 6}" y="{ly+16}" fill="{MUT}" font-size="12">More</text>')
svg.append("</svg>")

out = "\n".join(svg) + "\n"
if not (out.startswith("<svg") and len(out) > 2000):
    sys.exit("error: generated SVG failed validation; not writing")
with open(OUT, "w") as f:
    f.write(out)
print(f"ok: wrote {os.path.normpath(OUT)} ({len(out)} bytes, {total} contributions, {len(weeks)} weeks)")
