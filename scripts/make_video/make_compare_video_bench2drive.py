#!/usr/bin/env python
"""Render a side-by-side baseline-vs-ours comparison clip from dumped eval frames.

Both panels share one simulation clock: for a given sim time ``t`` each panel shows
the frame it recorded at ``t`` (step ``round(t * 20) + 1``, the agent dumps one PNG
per CARLA tick at 20 Hz). A panel whose route already finished holds its last frame
under a ROUTE COMPLETE card while the other keeps driving.

The dumped PNGs are ``camera (1024x512)`` stacked on a 400 px black text box whose
text is clipped on the right, so only the camera region is reused; speed, commentary
and infraction badges are re-rendered here from ``log.json`` / ``*_res.json``.

Usage:
    python tools/make_compare_video.py tools/shotlists/route111.json -o out/route111
    python tools/make_compare_video.py tools/shotlists/route111.json --preview 3,88.9
"""

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CARLA_FPS = 20.0  # agent dumps one frame per tick; see agent config carla_frame_rate

# ---------------------------------------------------------------------------
# layout (1920x1080)
# ---------------------------------------------------------------------------
W, H = 1920, 1080
MARGIN, GUTTER = 8, 16
PANEL_W, PANEL_H = 944, 472
CAM_SRC_H = 512  # rows of the dumped PNG that hold the camera image

TITLE_Y0, TITLE_Y1 = 0, 58
LABEL_Y0, LABEL_Y1 = 58, 98
CAM_Y0 = 102
HUD_Y0, HUD_Y1 = CAM_Y0 + PANEL_H + 8, CAM_Y0 + PANEL_H + 72
REASON_Y0, REASON_Y1 = HUD_Y1 + 10, HUD_Y1 + 158
BADGE_Y0, BADGE_Y1 = REASON_Y1 + 10, REASON_Y1 + 76
CAPTION_Y0, CAPTION_Y1 = BADGE_Y1 + 12, BADGE_Y1 + 126
FOOTER_Y0 = CAPTION_Y1 + 8

COL_X = (MARGIN, MARGIN + PANEL_W + GUTTER)

BG = (13, 15, 18)
CAPTION_BG = (8, 10, 13)
CAPTION_FG = (255, 255, 255)
CARD = (23, 27, 33)
CARD_EDGE = (44, 51, 61)
FG = (236, 240, 245)
DIM = (140, 150, 163)
ACCENT = {"ours": (56, 189, 248), "baseline": (148, 163, 184)}
BAD = (239, 68, 68)
GOOD = (52, 211, 153)
WARN = (251, 191, 36)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = font(30, bold=True)
F_LABEL = font(26, bold=True)
F_HUD_BIG = font(34, bold=True)
F_HUD = font(20)
F_REASON = font(25)
F_BADGE = font(21, bold=True)
F_CAPTION = font(45, bold=True)
F_SMALL = font(18)
F_LEGEND = font(25)  # footer key; wider than F_SMALL but still clears the rate badge
F_CARD_BIG = font(46, bold=True)

# The one thing worth colouring in the commentary is the speed decision, so the
# viewer can see at a glance what each model chose to do. Everything else stays
# plain: highlighting the objects and distances too left the panel too busy to read.
HL_ACTION = re.compile(
    r"\b(Accelerate|Decelerate|Remain stopped|Maintain (?:your current speed|the reduced speed)|Brake)\b"
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
LOC_RE = re.compile(r"\(x=(-?[\d.]+),\s*y=(-?[\d.]+),\s*z=(-?[\d.]+)\)")

# Infractions that flash the camera border. Only impacts qualify: the leaderboard
# logs a red light at its trigger box, by which point the signal is behind the car
# and out of frame, so a flash there reads as unmotivated. Those stay badge-only.
FLASH_EVENTS = {"collisions_vehicle", "collisions_pedestrian", "collisions_layout"}

INFRACTION_LABELS = {
    "collisions_vehicle": ("COLLISION", BAD),
    "collisions_pedestrian": ("PEDESTRIAN HIT", BAD),
    "collisions_layout": ("COLLISION", BAD),
    "red_light": ("RAN RED LIGHT", BAD),
    "stop_infraction": ("RAN STOP SIGN", BAD),
    "outside_route_lanes": ("OFF ROUTE LANE", WARN),
    "route_dev": ("ROUTE DEVIATION", BAD),
    "vehicle_blocked": ("BLOCKED", WARN),
    "scenario_timeouts": ("SCENARIO TIMEOUT", WARN),
    "yield_emergency_vehicle_infractions": ("NO YIELD", WARN),
}


def _one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern!r} under {root}")
    return hits[-1]


class Run:
    """One evaluated route: frames, per-step reasoning, score, infraction steps."""

    def __init__(self, run_dir, route_id, label, side, scenario_name=None,
                 event_steps=None, scores_override=None, events_override=None):
        self.dir = Path(run_dir)
        self.route_id = str(route_id)
        self.label = label
        self.side = side
        self.accent = ACCENT[side]

        # A retried route leaves several RouteScenario_* dirs side by side and only
        # the newest one is described by res/*.json, so pin the attempt explicitly
        # whenever there is more than one.
        found = sorted((self.dir / "viz" / self.route_id).glob("RouteScenario_*"))
        if scenario_name:
            scenario = self.dir / "viz" / self.route_id / scenario_name
            if not scenario.is_dir():
                raise FileNotFoundError(
                    f"{scenario_name!r} not found under {self.dir}/viz/{self.route_id}; "
                    f"have: {[p.name for p in found]}")
        else:
            if len(found) > 1:
                print(f"  ! {self.dir.name}: {len(found)} attempts present, using the "
                      f"newest ({found[-1].name}). Set \"scenario\" to pin another.")
            scenario = _one(self.dir, f"viz/{self.route_id}/RouteScenario_*")
        self.scenario_name = scenario.name
        self.is_newest_attempt = scenario == found[-1]
        self.images = _one(scenario, "debug_viz/*/*/*/images").parent / "images"
        metric = _one(scenario, "debug_viz/*/*/*/metric/metric_info.json")

        self.log = {int(e["step"]): e for e in json.loads((scenario / "log.json").read_text())}
        self.metric = {int(k): v for k, v in json.loads(metric.read_text()).items()}
        self.last_step = max(self.log)
        self.duration = self.last_step / CARLA_FPS
        self.progress = self._cumulative_distance()
        self.total_distance = max(self.progress.values())
        self.stopped_for = self._stationary_time()

        res = json.loads(_one(self.dir, f"res/{self.route_id}_res.json").read_text())
        record = res["_checkpoint"]["records"][0]
        self.scores = record["scores"]
        self.infractions = record.get("infractions", {})
        # A rerun overwrites res/*.json while the earlier attempt's frames stay on
        # disk, so the JSON can describe a run these frames are not from. When that
        # happens the shot list states the real outcome instead; main() prints a
        # notice, because these numbers no longer come from the leaderboard.
        self.overridden = bool(scores_override) or events_override is not None
        if scores_override:
            self.scores = {**self.scores, **scores_override}
        if events_override is None:
            self.events = self._locate_infractions(event_steps or {})
        else:
            self.events = self._declared_events(events_override)
        self.flash_frames = 21  # replaced in main() once the output fps is known
        self._cache = (None, None)

    def _cumulative_distance(self):
        """Metres driven per step, so 'the baseline has not moved' is on screen."""
        steps = sorted(self.metric)
        out, total = {steps[0]: 0.0}, 0.0
        for prev, step in zip(steps, steps[1:]):
            total += math.dist(self.metric[prev]["location"][:2],
                               self.metric[step]["location"][:2])
            out[step] = total
        return out

    def _stationary_time(self):
        """Seconds the ego has been continuously at rest, per step."""
        out, since = {}, None
        for step in sorted(self.log):
            moving = float(self.log[step].get("speed", 0.0) or 0.0) > 0.2
            if moving:
                since = None
            elif since is None:
                since = step
            out[step] = 0.0 if since is None else (step - since) / CARLA_FPS
        return out

    def _declared_events(self, declared):
        """Build the badge list straight from the shot list.

        Each entry is ``{"step": <int>, "type": <INFRACTION_LABELS key>}`` plus an
        optional ``count``. Steps follow the step_at() convention, so a hold on that
        step shows the badge and the border flash on the same frame.
        """
        events = []
        for spec in declared:
            key = spec["type"]
            if key not in INFRACTION_LABELS:
                raise ValueError(f"unknown infraction type {key!r}; "
                                 f"expected one of {sorted(INFRACTION_LABELS)}")
            text, colour = INFRACTION_LABELS[key]
            count = int(spec.get("count", 1))
            step = int(spec["step"])
            events.append({
                "step": step, "t": (step - 1) / CARLA_FPS,
                "text": f"{text} ×{count}" if count > 1 else text,
                "colour": colour, "dist": 0.0,
                "flash": key in FLASH_EVENTS, "count": count})
        return sorted(events, key=lambda e: e["t"])

    def _locate_infractions(self, overrides):
        """Map located infractions onto the step whose ego pose matches.

        Sustained contact is logged once per tick it persists, so several entries
        can share a location. They collapse onto one badge carrying a count; the
        score is still computed from every entry, which is why the card says x2.
        """
        found = {}
        for key, entries in self.infractions.items():
            if key not in INFRACTION_LABELS:
                continue
            text, colour = INFRACTION_LABELS[key]
            for entry in entries:
                m = LOC_RE.search(entry)
                if not m:  # min_speed and friends carry no location
                    continue
                x, y = float(m.group(1)), float(m.group(2))
                step, dist = min(
                    ((s, math.dist(v["location"][:2], (x, y))) for s, v in self.metric.items()),
                    key=lambda sd: sd[1],
                )
                slot = found.setdefault((text, step), {
                    "step": step, "t": step / CARLA_FPS, "text": text,
                    "colour": colour, "dist": dist,
                    "flash": key in FLASH_EVENTS, "count": 0})
                slot["count"] += 1
        # The leaderboard records where an infraction happened, not which frame
        # reads as the moment of it, and the nearest-pose match can land a tick or
        # two off. ``event_steps`` in the shot list re-pins one by name, using the
        # same step<->time convention as step_at() so a hold on that step shows the
        # badge and the border flash together.
        for event in found.values():
            if event["text"] in overrides:
                event["step"] = int(overrides[event["text"]])
                event["t"] = (event["step"] - 1) / CARLA_FPS
        events = sorted(found.values(), key=lambda e: e["t"])
        for event in events:
            if event["count"] > 1:
                event["text"] = f"{event['text']} ×{event['count']}"
        return events

    def step_at(self, t):
        return int(min(max(round(t * CARLA_FPS) + 1, 1), self.last_step))

    def camera(self, step):
        """Top CAM_SRC_H rows of the dumped frame, scaled to the panel."""
        if self._cache[0] == step:
            return self._cache[1]
        img = Image.open(self.images / f"{step}.png").convert("RGB")
        cam = img.crop((0, 0, img.width, min(CAM_SRC_H, img.height)))
        cam = cam.resize((PANEL_W, PANEL_H), Image.LANCZOS)
        self._cache = (step, cam)
        return cam

    def entry(self, step):
        return self.log.get(step, {})



# ---------------------------------------------------------------------------
# timeline: screen frame -> sim time
# ---------------------------------------------------------------------------
CAPTION_WRAP = 61  # characters per line at F_CAPTION; 2 lines fit the caption bar
CAPTION_MAX_LINES = 2
# Flash length is wall-clock on screen, not sim seconds: an infraction that lands
# inside a fast-forward segment would otherwise blink past in a couple of frames.
FLASH_SECONDS = 0.7


class Copy:
    """All on-screen wording, kept in its own JSON so it can be edited alone.

    ``text`` in the shot list points at that file (relative to the shot list).
    Segments then carry a caption *key*; a segment whose caption is not a known
    key is treated as literal text, so an inline shot list still works.
    """

    def __init__(self, spec, shotlist_path):
        ref = spec.get("text")
        self.data = {}
        self.path = None
        if isinstance(ref, str):
            self.path = (Path(shotlist_path).parent / ref).resolve()
            self.data = json.loads(self.path.read_text())
        elif isinstance(ref, dict):
            self.data = ref

        self.captions = self.data.get("captions", {})
        labels = self.data.get("labels", {})
        self.title = self.data.get("title") or spec.get("title", "")
        self.left_label = labels.get("left") or spec["left"].get("label", "baseline")
        self.right_label = labels.get("right") or spec["right"].get("label", "ours")
        self.legend = self.data.get("legend") or spec.get("legend", "")
        self.score_heading = self.data.get("score_card", {}).get("heading") or self.title

    def caption(self, value):
        if not value:
            return ""
        return self.captions.get(value, value)

    def check(self, segments):
        """Warn about caption keys that are missing or too long to fit the bar."""
        known = set(self.captions)
        for i, seg in enumerate(segments, 1):
            key = seg.get("caption")
            if not key:
                continue
            if known and key not in known and " " not in key:
                print(f"  ! segment {i}: caption key {key!r} is not in "
                      f"{self.path.name if self.path else 'the text block'}; "
                      f"drawing it as literal text")
                continue
            lines = textwrap.wrap(self.caption(key), width=CAPTION_WRAP)
            if len(lines) > CAPTION_MAX_LINES:
                dropped = " ".join(lines[CAPTION_MAX_LINES:])
                print(f"  ! segment {i} caption {key!r} needs {len(lines)} lines; "
                      f"only {CAPTION_MAX_LINES} are drawn. Dropped: {dropped!r}")
        for key in known - {s.get("caption") for s in segments}:
            print(f"  ! caption {key!r} is defined but never used")


class Timeline:
    """Piecewise shot list. ``play`` walks sim time at ``rate``; ``hold`` freezes it."""

    def __init__(self, segments, fps):
        self.fps = fps
        self.shots = []
        cursor = 0.0
        for seg in segments:
            kind = seg.get("kind", "play")
            if kind == "play":
                span = (seg["t1"] - seg["t0"]) / seg["rate"]
            elif kind == "hold":
                span = seg["dur"]
            else:
                raise ValueError(f"unknown segment kind {kind!r}")
            self.shots.append({**seg, "kind": kind, "screen_t0": cursor, "span": span})
            cursor += span
        self.screen_duration = cursor
        self.n_frames = int(round(cursor * fps))
        # Running count of frames on which the clock actually moved. The impact
        # flash is timed against this, so freezing on the crash keeps the border
        # lit for the whole freeze instead of going dark mid-hold.
        self.advance, moved = [], 0
        for i in range(self.n_frames):
            if self.at(i)[1] != 0.0:
                moved += 1
            self.advance.append(moved)

    def first_frame_at(self, t_sim_target):
        """Output frame on which the clock first reaches ``t_sim_target``."""
        for i in range(self.n_frames):
            if self.at(i)[0] >= t_sim_target:
                return i
        return self.n_frames  # never reached: the event stays off screen

    def at(self, frame_idx):
        """-> (sim_time, rate, caption, card) for one output frame."""
        screen_t = frame_idx / self.fps
        shot = self.shots[-1]
        for cand in self.shots:
            if screen_t < cand["screen_t0"] + cand["span"]:
                shot = cand
                break
        local = screen_t - shot["screen_t0"]
        if shot["kind"] == "play":
            t_sim = shot["t0"] + local * shot["rate"]
            rate = shot["rate"]
        else:
            t_sim = shot["t"]
            rate = 0.0
        return t_sim, rate, shot.get("caption", ""), shot.get("card")


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def card(draw, box, fill=CARD, edge=CARD_EDGE, radius=10):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=edge, width=1)


def spans(text):
    """Split commentary into (substring, colour) runs so the speed decision pops."""
    out, cursor = [], 0
    for m in HL_ACTION.finditer(text):
        start, end = m.start(), m.end()
        if start < cursor:
            continue
        if start > cursor:
            out.append((text[cursor:start], FG))
        out.append((text[start:end], WARN))
        cursor = end
    if cursor < len(text):
        out.append((text[cursor:], FG))
    return out


def wrap_spans(runs, width):
    """Re-wrap coloured runs to ``width`` characters, keeping colours per line."""
    lines, line, used = [], [], 0
    for text, colour in runs:
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            if word.isspace():
                if line:
                    line.append((word, colour))
                    used += len(word)
                continue
            # never start a line with dangling punctuation left over by a highlight span
            punctuation_only = not any(c.isalnum() for c in word)
            if used + len(word) > width and line and not punctuation_only:
                lines.append(line)
                line, used = [], 0
            line.append((word, colour))
            used += len(word)
    if line:
        lines.append(line)
    return lines


def draw_spans(draw, xy, lines, fnt, line_h):
    x0, y = xy
    for line in lines:
        x = x0
        for text, colour in line:
            draw.text((x, y), text, font=fnt, fill=colour)
            x += draw.textlength(text, font=fnt)
        y += line_h


def centred(draw, text, fnt, cx, y, fill=FG):
    draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)


def speed_bar(draw, box, value, vmax, colour):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill=(38, 43, 52))
    frac = max(0.0, min(1.0, value / vmax))
    if frac > 0.01:
        draw.rounded_rectangle((x0, y0, x0 + (x1 - x0) * frac, y1), radius=5, fill=colour)


# ---------------------------------------------------------------------------
# panel + frame composition
# ---------------------------------------------------------------------------
def draw_panel(img, draw, run: Run, t_sim, x0, adv_now, rate):
    finished = t_sim >= run.duration
    step = run.step_at(t_sim)
    entry = run.entry(step)
    cx = x0 + PANEL_W / 2

    # column label
    draw.text((x0 + 4, LABEL_Y0 + 6), run.label, font=F_LABEL, fill=run.accent)

    # camera
    img.paste(run.camera(step), (x0, CAM_Y0))
    draw.rectangle((x0, CAM_Y0, x0 + PANEL_W - 1, CAM_Y0 + PANEL_H - 1),
                   outline=CARD_EDGE, width=1)

    fired = [e for e in run.events if t_sim >= e["t"]]
    fresh = [e for e in fired
             if e["flash"] and 0 <= adv_now - e["adv"] < run.flash_frames]
    if fresh:  # flash the camera border on the infraction
        draw.rectangle((x0, CAM_Y0, x0 + PANEL_W - 1, CAM_Y0 + PANEL_H - 1),
                       outline=fresh[0]["colour"], width=8)

    # fast-forward marker on the image itself: during a compressed stretch the
    # motion looks wrong, and the footer badge is too small to catch the eye
    if rate > 1.05:
        chip = f"▶▶ {rate:g}x".replace(".0x", "x")
        w = draw.textlength(chip, font=F_BADGE) + 28
        bx, by = x0 + PANEL_W - 14 - w, CAM_Y0 + 14
        card(draw, (bx, by, bx + w, by + 42), fill=(46, 32, 8), edge=WARN)
        draw.text((bx + 14, by + 10), chip, font=F_BADGE, fill=WARN)

    # how long this agent has been standing still — SimLingo's main failure here
    idle = run.stopped_for.get(step, 0.0)
    if idle >= 2.0 and not finished:
        chip = f"STOPPED {idle:.1f} s"
        w = draw.textlength(chip, font=F_BADGE) + 28
        bx, by = x0 + 14, CAM_Y0 + PANEL_H - 56
        card(draw, (bx, by, bx + w, by + 42), fill=(46, 32, 8), edge=WARN)
        draw.text((bx + 14, by + 10), chip, font=F_BADGE, fill=WARN)

    if finished:
        overlay = Image.new("RGBA", (PANEL_W, PANEL_H), (13, 15, 18, 170))
        img.paste(Image.alpha_composite(
            img.crop((x0, CAM_Y0, x0 + PANEL_W, CAM_Y0 + PANEL_H)).convert("RGBA"),
            overlay).convert("RGB"), (x0, CAM_Y0))
        ds = run.scores["score_composed"]
        colour = GOOD if ds >= 99.5 else BAD
        centred(draw, "ROUTE COMPLETE", F_LABEL, cx, CAM_Y0 + PANEL_H / 2 - 62, DIM)
        centred(draw, f"DS {ds:.0f}", F_CARD_BIG, cx, CAM_Y0 + PANEL_H / 2 - 20, colour)
        centred(draw, f"RC {run.scores['score_route']:.0f}  ·  "
                      f"penalty {run.scores['score_penalty']:.2f}  ·  "
                      f"{run.duration:.1f} s", F_HUD, cx, CAM_Y0 + PANEL_H / 2 + 46, DIM)

    # HUD: speed readout + bar, step number on the right
    card(draw, (x0, HUD_Y0, x0 + PANEL_W, HUD_Y1))
    speed = max(float(entry.get("speed", 0.0) or 0.0), 0.0)
    draw.text((x0 + 16, HUD_Y0 + 12), f"{speed:.1f}", font=F_HUD_BIG, fill=FG)
    draw.text((x0 + 106, HUD_Y0 + 26), "m/s", font=F_HUD, fill=DIM)
    speed_bar(draw, (x0 + 152, HUD_Y0 + 26, x0 + 640, HUD_Y0 + 42), speed, 14.0, run.accent)
    tag = f"step {step}"
    draw.text((x0 + PANEL_W - 16 - draw.textlength(tag, font=F_HUD), HUD_Y0 + 26),
              tag, font=F_HUD, fill=DIM)

    # reasoning
    card(draw, (x0, REASON_Y0, x0 + PANEL_W, REASON_Y1))
    draw.text((x0 + 16, REASON_Y0 + 10), "REASONING", font=F_SMALL, fill=run.accent)
    commentary = entry.get("commentary", "")
    if commentary:
        draw_spans(draw, (x0 + 16, REASON_Y0 + 40),
                   wrap_spans(spans(commentary), 58), F_REASON, 34)

    # badges
    if fired:
        bx = x0
        for event in fired:
            label = event["text"]
            w = draw.textlength(label, font=F_BADGE) + 34
            card(draw, (bx, BADGE_Y0, bx + w, BADGE_Y1), fill=(48, 18, 22), edge=event["colour"])
            draw.text((bx + 17, BADGE_Y0 + 20), label, font=F_BADGE, fill=event["colour"])
            bx += w + 10
    elif finished:
        # only worth saying once the route is actually over - a clean badge while
        # the run is still going says nothing, and reads oddly right before a crash
        label = "NO INFRACTION"
        w = draw.textlength(label, font=F_BADGE) + 34  # same padding as the red badges
        card(draw, (x0, BADGE_Y0, x0 + w, BADGE_Y1), fill=(14, 40, 33), edge=GOOD)
        draw.text((x0 + 17, BADGE_Y0 + 20), label, font=F_BADGE, fill=GOOD)


def draw_score_card(img, draw, runs, heading):
    """Full-frame summary card used by ``hold`` segments with ``card: "scores"``."""
    draw.rectangle((0, 0, W, H), fill=BG)
    centred(draw, heading, F_TITLE, W / 2, 120, FG)
    for run, x0 in zip(runs, COL_X):
        cx = x0 + PANEL_W / 2
        card(draw, (x0, 220, x0 + PANEL_W, 820))
        centred(draw, run.label, F_LABEL, cx, 258, run.accent)
        ds = run.scores["score_composed"]
        centred(draw, f"{ds:.0f}", font(120, bold=True), cx, 330,
                GOOD if ds >= 99.5 else BAD)
        centred(draw, "DRIVING SCORE", F_SMALL, cx, 480, DIM)
        centred(draw, f"RC {run.scores['score_route']:.0f}   ·   "
                      f"penalty {run.scores['score_penalty']:.2f}", F_HUD, cx, 520, FG)
        centred(draw, f"route time {run.duration:.1f} s", F_HUD, cx, 552, DIM)
        y = 606
        # min-speed infractions are left off: they carry no penalty multiplier
        # (0.42 here is exactly 0.60 collision x 0.70 red light) and both agents
        # rack up about fifteen, so listing them says nothing about the gap.
        rows = [(e["text"], e["colour"]) for e in run.events]
        if not rows:
            rows = [("no infractions", GOOD)]
        for text, colour in rows[:5]:
            centred(draw, text, F_BADGE, cx, y, colour)
            y += 38


def draw_frame(runs, timeline, frame_idx, copy: "Copy"):
    t_sim, rate, caption_key, card_kind = timeline.at(frame_idx)
    caption = copy.caption(caption_key)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    if card_kind == "scores":
        draw_score_card(img, draw, runs, copy.score_heading)
        return img

    draw.rectangle((0, TITLE_Y0, W, TITLE_Y1), fill=(18, 21, 26))
    draw.text((MARGIN + 6, TITLE_Y0 + 14), copy.title, font=F_TITLE, fill=FG)

    for run, x0 in zip(runs, COL_X):
        draw_panel(img, draw, run, t_sim, x0, timeline.advance[frame_idx], rate)

    draw.line((COL_X[1] - GUTTER // 2, LABEL_Y0, COL_X[1] - GUTTER // 2, FOOTER_Y0),
              fill=CARD_EDGE, width=1)

    if caption:
        card(draw, (MARGIN, CAPTION_Y0, W - MARGIN, CAPTION_Y1), fill=CAPTION_BG)
        lines = textwrap.wrap(caption, width=CAPTION_WRAP)[:CAPTION_MAX_LINES]
        line_h = 54
        y = CAPTION_Y0 + (CAPTION_Y1 - CAPTION_Y0 - line_h * len(lines)) / 2
        for line in lines:
            centred(draw, line, F_CAPTION, W / 2, y, CAPTION_FG)
            y += line_h

    draw.text((MARGIN + 6, FOOTER_Y0 + 14), copy.legend, font=F_LEGEND, fill=DIM)
    if rate and abs(rate - 1.0) > 0.05:
        # DejaVu has no U+23E9 fast-forward glyph, so double the play triangle
        tag = f"{rate:g}x".replace(".0x", "x")
        tag = ("▶▶ " if rate > 1 else "▶ ") + tag
        w = draw.textlength(tag, font=F_BADGE) + 30
        card(draw, (W - MARGIN - w, FOOTER_Y0 + 10, W - MARGIN, FOOTER_Y0 + 52),
             fill=(18, 21, 26), edge=WARN)
        draw.text((W - MARGIN - w + 15, FOOTER_Y0 + 20), tag, font=F_BADGE, fill=WARN)
    return img


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shotlist", help="JSON shot list (see tools/shotlists/)")
    ap.add_argument("-o", "--out", help="output dir (default: shotlist's out_dir)")
    ap.add_argument("--preview", help="comma-separated sim times; render those frames only")
    ap.add_argument("--no-video", action="store_true", help="write the PNG sequence only")
    ap.add_argument("--check", action="store_true",
                    help="validate the shot list and its copy, render nothing")
    ap.add_argument("--fps", type=int,
                    help="override the shot list's output fps (source is 20 Hz)")
    args = ap.parse_args()

    spec = json.loads(Path(args.shotlist).read_text())
    copy = Copy(spec, args.shotlist)
    copy.check(spec["segments"])

    runs = [Run(side["run"], spec["route_id"], label, key, side.get("scenario"),
                side.get("event_steps"), side.get("scores"), side.get("events"))
            for key, side, label in (("baseline", spec["left"], copy.left_label),
                                     ("ours", spec["right"], copy.right_label))]
    timeline = Timeline(spec["segments"], args.fps or spec.get("fps", 30))
    out = Path(args.out or spec["out_dir"])

    # pin each infraction to the output frame it lands on, so the border flash
    # lasts the same on-screen time whatever rate that segment runs at
    for run in runs:
        run.flash_frames = round(FLASH_SECONDS * timeline.fps)
        for event in run.events:
            event["frame"] = timeline.first_frame_at(event["t"])
            event["adv"] = timeline.advance[min(event["frame"], timeline.n_frames - 1)]

    for run in runs:
        events = ", ".join(
            f"{e['text']} @ {e['t']:.1f}s (frame {e['frame']}"
            + (", flash)" if e["flash"] else ", badge only)")
            for e in run.events) or "none"
        print(f"[{run.side:8s}] {run.label}: {run.last_step} steps / "
              f"{run.duration:.1f} s, DS {run.scores['score_composed']:.0f}")
        print(f"           events: {events}")
        if run.overridden:
            print(f"  ! {run.label}: score/infractions come from the shot list, not "
                  f"res/{run.route_id}_res.json (that file describes a different run)")
        if not run.is_newest_attempt:
            print(f"  ! {run.label}: showing {run.scenario_name}, but res/"
                  f"{run.route_id}_res.json describes the newest attempt. The score "
                  f"card and the infraction badges belong to that other run, not this one.")

    print(f"{timeline.screen_duration:.1f} s @ {timeline.fps} fps "
          f"= {timeline.n_frames} frames")
    if args.check:
        return

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        for t in [float(x) for x in args.preview.split(",")]:
            idx = next((i for i in range(timeline.n_frames)
                        if timeline.at(i)[0] >= t), timeline.n_frames - 1)
            path = out / f"preview_t{t:g}.png"
            draw_frame(runs, timeline, idx, copy).save(path)
            print(f"preview t={t:g}s (screen {idx / timeline.fps:.2f}s) -> {path}")
        return

    for i in range(timeline.n_frames):
        draw_frame(runs, timeline, i, copy).save(frames_dir / f"{i:05d}.png")
        if i % 30 == 0:
            print(f"  {i:5d}/{timeline.n_frames}  sim t={timeline.at(i)[0]:6.2f}s", flush=True)
    print(f"frames -> {frames_dir}")

    if not args.no_video:
        encode(frames_dir, out / f"{spec['route_id']}_compare.mp4", timeline.fps)


def find_ffmpeg():
    """OpenCV here only ships mp4v, so H.264 needs a real ffmpeg.

    ``pip install imageio-ffmpeg`` drops a static build into the env and is the
    least invasive way to get one; $FFMPEG_BIN overrides everything.
    """
    import os
    import shutil

    if os.environ.get("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode(frames_dir, out_path, fps):
    """Prefer ffmpeg/H.264; fall back to OpenCV's mp4v so a clip always lands."""
    import subprocess

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
               "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd, check=True)
        print(f"video -> {out_path}")
        return

    import cv2
    print("ffmpeg not found; falling back to OpenCV mp4v (re-encode to H.264 later)")
    paths = sorted(frames_dir.glob("*.png"))
    writer = cv2.VideoWriter(str(out_path.with_suffix(".mp4v.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for p in paths:
        writer.write(np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1])
    writer.release()
    print(f"video -> {out_path.with_suffix('.mp4v.mp4')}")


if __name__ == "__main__":
    main()
