#!/usr/bin/env python
"""把一段 nuPlan clip 拼成 Baseline vs Ours 的对比视频。

版式(1920x1080):
    ┌ 标题 ────────────────────────────────────────────────┐
    │ 三路相机 + GT/Ours/Baseline 三条轨迹 (直接用预渲染的条带) │
    ├──────────────────────┬───────────────────────────────┤
    │ Baseline 的 reasoning │ Ours 的 reasoning              │
    ├──────────────────────┴───────────────────────────────┤
    │ description                                           │
    └───────────────────────────────────────────────────────┘

与 make_compare_video_bench2drive.py 的关系:那边是闭环,两套仿真两条时间线,
一半代码在做双面板时钟同步。nuPlan navtest 是开环 —— 每帧两个模型看的是同一张
GT 回放图,所以只有一排相机、一条时间线。沿用的是它的 Copy(文案外置 JSON)、
违规闪边框、分数卡和 ffmpeg 编码这几块。

节奏:源数据是 2 Hz(0.5s 一帧),按真实速度播就是 2 fps 幻灯片、reasoning 根本
来不及读。所以每帧占屏时间是【屏幕时间】,默认 1.2s;baseline 出问题的帧延长并
闪边框 —— 红 = 撞车/出界(硬违规,PDMS 直接归零),琥珀 = TTC 归零或该走不走
(加权项掉分)。一律用红会把"progress 低"说成撞车。

  python scripts/make_video/make_compare_video_nuplan.py                  # 三段全做
  python scripts/make_video/make_compare_video_nuplan.py --clips lead_stop
  python scripts/make_video/make_compare_video_nuplan.py --clips lead_stop --preview 654,658
"""
import os, re, json, math, difflib, textwrap, argparse, subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("/home/nvidia/workspace/other_repo/AutoVLA")
SRC = REPO / "logs/0920_nvidia/video_sources"
OUT = REPO / "logs/0920_nvidia/video"
USABLE = ["lead_stop", "lead_truck", "stopsign_pullaway"]

# 片头标题。meta 里的 note 是中文备注,不适合直接当标题,这里给英文的。
# 生成 copy JSON 时写进去,之后改 JSON 即可,不用动代码。
TITLES = {
    "lead_stop": "Following a lead vehicle that is slowing to a stop",
    "lead_truck": "Closing on a stopped truck",
    "stopsign_pullaway": "Pulling away from a stop line",
    "junction_queue": "Joining a queue at a junction",
    "approach_stop": "Decelerating to a stop",
    "lot_leftturn": "Low-speed left turn",
    "leftturn_pullaway": "Unprotected left turn from standstill",
}

# ---------------------------------------------------------------------------
# 版式
# ---------------------------------------------------------------------------
W, H = 1920, 1080
MARGIN, GUTTER = 8, 16
STRIP_W, STRIP_H = 1920, 540           # render_clip_frames.py 的输出尺寸

TITLE_Y0, TITLE_Y1 = 0, 56
STRIP_Y0 = 60
# reasoning 框高按 4 行正文算(三段实测最多 3 行,留一行余量),剩下的全给字幕。
# 4 行 = 90(标题行+chip 行) + 4*43 + 14 内边距
REASON_Y0, REASON_Y1 = STRIP_Y0 + STRIP_H + 12, STRIP_Y0 + STRIP_H + 288
CAPTION_Y0, CAPTION_Y1 = REASON_Y1 + 12, H - 8

PANEL_W = (W - 2 * MARGIN - GUTTER) // 2
COL_X = (MARGIN, MARGIN + PANEL_W + GUTTER)

BG, CARD, CARD_EDGE = (13, 15, 18), (23, 27, 33), (44, 51, 61)
CAPTION_BG = (8, 10, 13)
FG, DIM, DIMMER = (236, 240, 245), (150, 159, 171), (108, 116, 128)
C_OURS, C_BASE, C_GT = (56, 189, 248), (224, 138, 92), (39, 176, 66)
BAD, WARN, GOOD = (239, 68, 68), (251, 191, 36), (52, 211, 153)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
CJK_RE = re.compile(r"[⺀-鿿＀-￯　-〿]")
_CJK_CACHE = {}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def fit(text, fnt):
    """文案里有中日韩字符就换字体 —— DejaVu 没有 CJK 字形,会整排渲成豆腐块。"""
    if not text or not CJK_RE.search(text) or not CJK_FONT.exists():
        return fnt
    size = getattr(fnt, "size", 20)
    if size not in _CJK_CACHE:
        _CJK_CACHE[size] = ImageFont.truetype(str(CJK_FONT), size)
    return _CJK_CACHE[size]


def wrap_text(text, width):
    """英文按词换行;中文没有空格,按字数换。"""
    if CJK_RE.search(text or ""):
        n = max(8, int(width * 0.52))       # 汉字大约占两个西文字符宽
        return [text[i:i + n] for i in range(0, len(text), n)]
    return textwrap.wrap(text, width=width)


F_TITLE = font(28, bold=True)
F_SUB = font(18)
F_LABEL = font(24, bold=True)
F_PANEL = font(29, bold=True)   # 面板上的模型名
F_PLAN = font(21, bold=True)
F_THINK = font(32)
F_THINK_B = font(32, bold=True)
F_CHIP = font(19, bold=True)
F_CAPTION = font(38, bold=True)
F_SUMMARY = font(26)           # 片尾卡里那句 reasoning 总结
F_SMALL = font(17)
F_CARD_BIG = font(110, bold=True)

# ---------------------------------------------------------------------------
# 违规判定。只认本次推理的 PDMS 子分(meta 里的 o/b),不自己按几何推断。
#   bad  = 撞车 / 出界,乘法项归零 -> 该帧 PDMS 直接是 0 -> 画红框
#   warn = 该走不走,只是 progress 低 -> 挂 chip,不画框
#
# TTC 归零【不报】:它只说明车头离前车太近,并不是真撞了。lead_stop 的 f656 就是
# 这种 —— baseline 规划 11.3 m、前车尾部在 10.8 m,只超出 0.5 m,PDMS 闭环仿真里
# 前车自己挪开了,没有接触(NC=1, PDMS 58.3)。给它挂 COLLISION 或红框,会和同一
# 块面板上的 PDMS 58 自相矛盾。
# ---------------------------------------------------------------------------
SLOW_GAP = 0.40        # ego_progress 落后多少才算"该走不走"
SRC_DT = 0.5           # navtest 是 2 Hz,一帧 0.5s 真实时间


def violations(m, other):
    """-> [(文字, 颜色, 等级)]，m 是这个模型的子分，other 是对手的。"""
    out = []
    hard = m["no_at_fault_collisions"] * m["drivable_area_compliance"]
    if m["no_at_fault_collisions"] == 0:
        out.append(("COLLISION", BAD, "bad"))
    if m["drivable_area_compliance"] == 0:
        out.append(("OFF-ROAD", BAD, "bad"))
    # ego_progress 只有在乘法项没归零时才说明问题:PDMS 聚合时会把 progress 乘上
    # 乘法项(pdm_scorer._aggregate_scores),撞了车 progress 必然是 0 —— 那是撞车的
    # 结果,不是"走得太慢"。两边都干净时比才有意义。
    hard_other = other["no_at_fault_collisions"] * other["drivable_area_compliance"]
    if hard == 1 and hard_other == 1 and other["ego_progress"] - m["ego_progress"] >= SLOW_GAP:
        out.append(("TOO SLOW", WARN, "warn"))
    return out


def severity(fr):
    """整帧的等级,由【baseline】决定 —— 边框讲的是 baseline 的故事。"""
    lv = [v[2] for v in violations(fr["b"], fr["o"])]
    return "bad" if "bad" in lv else ("warn" if "warn" in lv else "ok")


# ---------------------------------------------------------------------------
# 文案:与 bench2drive 的 Copy 一样外置成 JSON,改字不用重渲
# ---------------------------------------------------------------------------
class Copy:
    def __init__(self, clip, path):
        self.path = Path(path)
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.title = d.get("title") or TITLES.get(clip["tag"], clip["tag"])
        self.subtitle = d.get("subtitle") or (
            f"nuPlan navtest · {clip['log']} · f{clip['i0']}-{clip['i1']}")
        self.left = d.get("labels", {}).get("left", "SFT model")
        self.right = d.get("labels", {}).get("right", "Ours")
        # 空串 = 不画。不能用 `or`,那样空串会退回默认值
        self.legend = d["legend"] if "legend" in d else ""
        self.captions = d.get("captions", {})
        self.card = d.get("card", {})
        # 片尾卡上每个模型一句 reasoning 总结。空着就不画
        self.summary = self.card.get("summary", {})

    def caption_at(self, fi):
        """字幕是粘的:没给新句子就沿用上一句,和字幕轨一个道理。"""
        keys = sorted(int(k.lstrip("f")) for k in self.captions)
        cur = ""
        for k in keys:
            if k <= fi:
                cur = self.captions[f"f{k:05d}"] if f"f{k:05d}" in self.captions \
                    else self.captions[f"f{k}"]
            else:
                break
        return cur

    @staticmethod
    def template(clip, path):
        """按每帧的违规变化生成一份 description 草稿,之后人工改。"""
        caps, prev = {}, None
        for fr in clip["frames"]:
            vb = violations(fr["b"], fr["o"])
            key = tuple(v[0].split("  ")[0] for v in vb)
            if key == prev:
                continue
            prev = key
            if not vb:
                txt = "Both models agree here."
            elif "COLLISION" in key:
                txt = (f"The SFT model keeps driving — it plans {fr['disp5s_b']:.0f} m "
                       f"where the expert travels {fr['disp5s_gt']:.0f} m — and hits the "
                       f"vehicle ahead. PDMS drops to 0.")
            elif any(k.startswith("TOO SLOW") for k in key):
                txt = (f"The way is clear and the expert pulls away "
                       f"({fr['disp5s_gt']:.0f} m in 5 s), but the baseline still plans "
                       f"{fr['disp5s_b']:.1f} m — it stays put.")
            else:
                txt = (f"The SFT model closes on the vehicle ahead too fast "
                       f"({fr['disp5s_b']:.0f} m vs the expert's {fr['disp5s_gt']:.0f} m).")
            caps[f"f{fr['fi']:05d}"] = txt
        def top_action(field):
            """模型自己最常说的那个动作('I should ...' 之后那截)。"""
            from collections import Counter
            c = Counter()
            for fr in clip["frames"]:
                m = re.search(r"I should ([a-z][^.,:;]*)", think_of(fr.get(field, "")))
                if m:
                    c[m.group(1).strip()] += 1
            return c.most_common(1)[0][0] if c else ""

        n_coll = sum(1 for fr in clip["frames"] if fr["b"]["no_at_fault_collisions"] == 0)
        summary = {
            "left": f'Wrongly reasons "{top_action("raw_base")}"'
                    + (", then hits the car ahead." if n_coll else "."),
            "right": f'Reasons "{top_action("raw_ours")}".',
        }
        d = dict(title=TITLES.get(clip["tag"], clip["tag"]),
                 subtitle=f"nuPlan navtest · {clip['log']} · f{clip['i0']}-{clip['i1']}",
                 labels=dict(left="SFT model", right="Ours"),
                 legend="",     # 片尾卡底部那行;留空就不画
                 captions=caps,
                 card=dict(heading=TITLES.get(clip["tag"], clip["tag"]), summary=summary))
        path.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        return d


class Reasoning:
    """两个模型每帧的 <think> / <PLAN>,单独一份 JSON,与文案一样可手改。

    原始来源是 run_clip_dump.sh 落盘的 dump/*.jsonl(raw_output),
    render_clip_frames.py 把它原样抄进 video_sources/<tag>/meta.json 的
    raw_ours / raw_base。这里第一次运行时把 <think> 和 <PLAN> 抽出来落成
    <tag>.reasoning.json,之后就以这份为准,已存在不覆盖。

    ⚠️ 改 think / plan 只改画面上显示的字,【不会】改模型真正的输出,也不影响
    PDMS 分数和轨迹 —— 那两样来自同一次推理、已经定死了。加载时会拿每条和
    meta.json 里的模型原文比一遍,不一致就打印出来。

    基准取 meta.json 而不是 JSON 里自带的 *_original:后者跟 think 放在同一个
    文件里,一起改掉就等于没有基准了(实际发生过)。meta.json 是推理落盘的产物,
    不该手动动它。
    """

    def __init__(self, clip, path):
        self.path = Path(path)
        if not self.path.exists():
            self.template(clip, self.path)
            print(f"  reasoning -> {self.path}  (可手改,已存在不覆盖)")
        self.data = json.loads(self.path.read_text())["frames"]
        edited = []
        for fr in clip["frames"]:
            key = f"f{fr['fi']:05d}"
            for src, who in (("raw_base", "baseline"), ("raw_ours", "ours")):
                truth, plan = split_plan(think_of(fr.get(src, "")))
                m = self.data.get(key, {}).get(who, {})
                if (m.get("think", "").strip() != truth.strip()
                        or m.get("plan", "").strip() != plan.strip()):
                    edited.append(f"{key}/{who}")
        if edited:
            print(f"  ! reasoning 与 meta.json 里的模型原文不一致 {len(edited)} 处,"
                  f"画面上的话已不是模型原话: "
                  f"{', '.join(edited[:6])}{' ...' if len(edited) > 6 else ''}")

    WHO = {"b": "baseline", "o": "ours"}

    def get(self, fi, side):
        m = self.data.get(f"f{fi:05d}", {}).get(self.WHO[side], {})
        return m.get("think", ""), m.get("plan", "")

    @staticmethod
    def template(clip, path):
        frames = {}
        for fr in clip["frames"]:
            entry = {}
            for src, who in (("raw_base", "baseline"), ("raw_ours", "ours")):
                think, plan = split_plan(think_of(fr.get(src, "")))
                entry[who] = dict(think=think, plan=plan)
            frames[f"f{fr['fi']:05d}"] = entry
        path.write_text(json.dumps(dict(
            _source=(f"video_sources/{clip['tag']}/meta.json 的 raw_ours / raw_base,"
                     f"即 run_clip_dump.sh 落盘的 dump/*.jsonl"),
            _note=("改 think / plan 只改画面显示,不改模型真实输出,也不影响 PDMS 和轨迹。"
                   "每次渲染都会和 meta.json 的原文比对,不一致会在终端提示。"),
            frames=frames), indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 时间线:每帧占多少【屏幕时间】
# ---------------------------------------------------------------------------
class Timeline:
    def __init__(self, clip, a):
        self.fps = a.fps
        self.shots, cursor = [], 0.0
        for i, fr in enumerate(clip["frames"]):
            sev = severity(fr)
            # 源数据 2 Hz,所以 0.5s 的真实时间摊到 hold 秒的屏幕时间上 = hold 倍速
            hold = SRC_DT / {"bad": a.speed_bad, "warn": a.speed_warn,
                             "ok": a.speed}[sev]
            dur = hold
            if i == 0:
                dur += a.lead_in
            if i == len(clip["frames"]) - 1:
                dur += a.lead_out
            # hold 与 dur 分开存:首末帧额外加了停顿,拿 dur 去算倍速会显示错的数
            self.shots.append(dict(kind="frame", i=i, sev=sev, t0=cursor,
                                   dur=dur, hold=hold))
            cursor += dur
        if a.card_sec > 0:
            self.shots.append(dict(kind="card", i=-1, sev="ok", t0=cursor,
                                   dur=a.card_sec, hold=a.card_sec))
            cursor += a.card_sec
        self.duration = cursor
        self.n_frames = int(round(cursor * a.fps))

    def at(self, k):
        t = k / self.fps
        shot = self.shots[-1]
        for cand in self.shots:
            if t < cand["t0"] + cand["dur"]:
                shot = cand
                break
        return shot, t - shot["t0"]


# ---------------------------------------------------------------------------
# reasoning 文本
# ---------------------------------------------------------------------------
def think_of(raw):
    m = re.search(r"<think>(.*?)</think>", raw or "", re.S)
    return (m.group(1) if m else (raw or "")[:300]).strip()


def split_plan(txt):
    m = re.search(r"<PLAN>(.*?)</PLAN>", txt)
    plan = m.group(1) if m else ""
    body = re.sub(r"\s*:?\s*<PLAN>.*?</PLAN>", "", txt).strip().rstrip(":").strip()
    return body, plan


# 自车的速度动作说法。扫过这三段 reasoning 统计出来的实际用词:
#   stay behind it 43 / come to a stop 16 / remain stopped 16 /
#   hold my current speed 14 / slow down 13 / speed up 5
# 长的必须排在前面 —— re 的 | 是最左最先匹配,"remain stopped" 排在 "stop" 后面
# 就永远轮不到。"stopped"(描述前车)不在表里:那是感知读数,不是自车的决定。
SPEED_RE = re.compile(
    r"\b(?:come to a (?:complete |full )?stop|remain stopped|stay stopped|"
    r"hold (?:my|the) current speed|maintain (?:my|the) (?:current )?speed|"
    r"keep (?:my|the) (?:current )?speed|speed up|slow down|"
    r"accelerate|decelerate|brake|stop)\b", re.I)


def speed_spans(text, colour):
    """把自车的速度动作加粗染色,其余保持正常。

    上一版是拿两边 reasoning 做 diff 染色,结果连 "14" vs "13" 这种米数差异都被
    高亮了 —— 那是感知读数不是决策,抢了速度动词的注意力。
    """
    hot = set()
    for m in SPEED_RE.finditer(text):
        hot.update(range(m.start(), m.end()))
    out, pos = [], 0
    for word in text.split():
        i = text.index(word, pos)
        pos = i + len(word)
        on = any(j in hot for j in range(i, pos))
        out.append((word, colour if on else DIM, F_THINK_B if on else F_THINK))
    return out


def wrap_words(words, draw, max_w):
    """逐词换行。每个词自带字体(速度动词用粗体),宽度得按各自的字体量。"""
    lines, line, w_used = [], [], 0.0
    for word, colour, fnt in words:
        ww = draw.textlength(word, font=fnt)
        space = draw.textlength(" ", font=fnt)
        if line and w_used + space + ww > max_w:
            lines.append(line)
            line, w_used = [], 0.0
        if line:
            w_used += space
        line.append((word, colour, fnt))
        w_used += ww
    if line:
        lines.append(line)
    return lines


def draw_words(draw, xy, lines, line_h):
    x0, y = xy
    for line in lines:
        x = x0
        for word, colour, fnt in line:
            draw.text((x, y), word, font=fnt, fill=colour)
            x += draw.textlength(word, font=fnt) + draw.textlength(" ", font=fnt)
        y += line_h


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------
def card(draw, box, fill=CARD, edge=CARD_EDGE, radius=10, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=edge, width=width)


def centred(draw, text, fnt, cx, y, fill=FG):
    fnt = fit(text, fnt)
    draw.text((cx - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)


def chip(draw, x, y, text, colour, fnt=F_CHIP, pad=13):
    w = draw.textlength(text, font=fnt) + pad * 2
    card(draw, (x, y, x + w, y + 32), fill=(18, 21, 26), edge=colour)
    draw.text((x + pad, y + 5), text, font=fnt, fill=colour)
    return w + 8


def draw_reason(img, draw, fr, side, x0, label, colour, reasoning, a_plan_chip=False):
    """一个模型的 reasoning 面板:标题行 + PLAN chip + <think> 正文。"""
    other = "o" if side == "b" else "b"
    mine_txt, plan = reasoning.get(fr["fi"], side)

    card(draw, (x0, REASON_Y0, x0 + PANEL_W, REASON_Y1))
    f_lab = fit(label, F_PANEL)
    draw.text((x0 + 18, REASON_Y0 + 7), label, font=f_lab, fill=colour)
    draw.text((x0 + 18 + draw.textlength(label, font=f_lab) + 14, REASON_Y0 + 19),
              "REASONING", font=F_SMALL, fill=DIMMER)

    pdms = fr[f"pdms_{'b' if side == 'b' else 'o'}"]
    tag = f"PDMS {pdms:.0f}"
    tc = GOOD if pdms >= 99.5 else (BAD if pdms <= 0.1 else WARN)
    draw.text((x0 + PANEL_W - 18 - draw.textlength(tag, font=F_LABEL), REASON_Y0 + 10),
              tag, font=F_LABEL, fill=tc)

    # 违规 chip(两个模型都标,Ours 自己犯的也照标)。
    # meta-action 的 chip(DECELERATE,STRAIGHT 之类)默认不画 —— 速度动词已经在
    # 正文里加粗染色了,再挂一个框是同一件事说两遍。--plan-chip 可开回来。
    y = REASON_Y0 + 46
    x = x0 + 18
    if plan and a_plan_chip:
        x += chip(draw, x, y, plan, colour)
    for text, col, _lv in violations(fr[side], fr[other]):
        x += chip(draw, x, y, text, col)

    lines = wrap_words(speed_spans(mine_txt, colour), draw, PANEL_W - 36)
    draw_words(draw, (x0 + 18, y + 44), lines[:4], 43)


PROG_GAP = 0.20        # progress 差多少才值得摆上卡片


def card_metric(clip):
    """卡片第二行:显示【真正造成这个分差的那一项】子分 -> {side: (文字, 颜色)}。

    写死成碰撞数在 lead_stop 那种片子上是对的(4 帧归零把均分压到 64),
    但 stopsign_pullaway 两边都没撞,26 分的差【全部】来自 ego_progress:
        progress 差 0.62 x 权重 5/12(41.7 分) = 25.8  ≈  实际分差 94-68=26
    那种片子上摆一行 "0 of 5 frames collide" 既没信息,还把注意力引到不相干的
    指标上,所以按哪一项拉开了差距来选。
    """
    n = clip["n"]
    coll = {s: sum(1 for f in clip["frames"] if f[s]["no_at_fault_collisions"] == 0)
            for s in ("b", "o")}
    if coll["b"] or coll["o"]:
        return {s: (f"{coll[s]} of {n} frames collide", BAD if coll[s] else GOOD)
                for s in ("b", "o")}
    prog = {s: sum(f[s]["ego_progress"] for f in clip["frames"]) / n for s in ("b", "o")}
    if abs(prog["o"] - prog["b"]) >= PROG_GAP:
        best = max(prog.values())
        return {s: (f"mean progress {prog[s]:.2f}",
                    GOOD if prog[s] >= best - 0.05 else WARN) for s in ("b", "o")}
    return {}


def draw_card(img, draw, clip, copy, a):
    """片尾分数卡。"""
    draw.rectangle((0, 0, W, H), fill=BG)
    centred(draw, copy.card.get("heading", copy.title), F_TITLE, W / 2, 130)
    centred(draw, copy.subtitle, F_SUB, W / 2, 176, DIM)
    metric = card_metric(clip)
    for (x0, label, colour, pdms, key, side) in (
            (COL_X[0], copy.left, C_BASE, clip["pdms_b"], "b", "left"),
            (COL_X[1], copy.right, C_OURS, clip["pdms_o"], "o", "right")):
        cx = x0 + PANEL_W / 2
        card(draw, (x0, 250, x0 + PANEL_W, 800))
        centred(draw, label, F_PANEL, cx, 287, colour)
        centred(draw, f"{pdms:.0f}", F_CARD_BIG, cx, 360,
                GOOD if pdms >= 90 else (WARN if pdms >= 60 else BAD))
        centred(draw, "mean PDMS over the clip", F_SMALL, cx, 510, DIM)
        if key in metric:
            text, col = metric[key]
            centred(draw, text, F_CHIP, cx, 560, col)
        # reasoning 总结:卡片内居中。分数说结果,这句说【为什么】
        txt = copy.summary.get(side, "")
        if txt:
            y = 636
            for line in wrap_text(txt, 54)[:3]:
                centred(draw, line, F_SUMMARY, cx, y, FG)
                y += 36
    y = 850
    if copy.legend:
        centred(draw, copy.legend, F_SMALL, W / 2, y, DIM)
        y += 30
    # 每帧页脚已经去掉,倍速和源帧率这个交代放在这里,不然片子里就没人说了
    centred(draw, f"source 2 Hz · normal frames {a.speed:g}x · collisions held at {a.speed_bad:g}x",
            F_SMALL, W / 2, y, DIMMER)


def draw_frame(clip, strips, copy, reasoning, timeline, k, a):
    shot, local = timeline.at(k)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    if shot["kind"] == "card":
        draw_card(img, draw, clip, copy, a)
        return img

    fr = clip["frames"][shot["i"]]

    draw.rectangle((0, TITLE_Y0, W, TITLE_Y1), fill=(18, 21, 26))
    ft = fit(copy.title, F_TITLE)
    draw.text((MARGIN + 8, TITLE_Y0 + 6), copy.title, font=ft, fill=FG)
    draw.text((MARGIN + 8 + draw.textlength(copy.title, font=ft) + 20,
               TITLE_Y0 + 16), copy.subtitle, font=fit(copy.subtitle, F_SUB), fill=DIM)

    img.paste(strips[shot["i"]], (0, STRIP_Y0))

    # baseline 出问题 -> 闪边框。前 flash 秒加粗,之后留一圈细的,停顿期间一直挂着
    vb = violations(fr["b"], fr["o"])
    # 默认只有真撞车 / 出界才画框。warn(该走不走)默认不画 —— 那类片子往往整段
    # 每帧都命中,框一直亮着就成了背景色,不再是信号。--warn-border 可开。
    if shot["sev"] == "bad" or (shot["sev"] == "warn" and a.warn_border):
        col = BAD if shot["sev"] == "bad" else WARN
        lw = 10 if local < a.flash else 5
        draw.rectangle((0, STRIP_Y0, W - 1, STRIP_Y0 + STRIP_H - 1), outline=col, width=lw)
        text = f"{copy.left}: " + vb[0][0].split("  ")[0]
        tw = draw.textlength(text, font=F_LABEL) + 30
        bx, by = MARGIN + 12, STRIP_Y0 + STRIP_H - 60
        card(draw, (bx, by, bx + tw, by + 44), fill=(28, 10, 12), edge=col, width=2)
        draw.text((bx + 15, by + 8), text, font=fit(text, F_LABEL), fill=col)

    draw_reason(img, draw, fr, "b", COL_X[0], copy.left, C_BASE, reasoning, a.plan_chip)
    draw_reason(img, draw, fr, "o", COL_X[1], copy.right, C_OURS, reasoning, a.plan_chip)

    caption = copy.caption_at(fr["fi"])
    if caption:
        card(draw, (MARGIN, CAPTION_Y0, W - MARGIN, CAPTION_Y1), fill=CAPTION_BG)
        lines = wrap_text(caption, a.caption_wrap)[:3]
        y = CAPTION_Y0 + (CAPTION_Y1 - CAPTION_Y0 - 46 * len(lines)) / 2
        for line in lines:
            centred(draw, line, F_CAPTION, W / 2, y)
            y += 46

    return img


# ---------------------------------------------------------------------------
def find_ffmpeg():
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


X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def encode(frames_dir, out_path, fps):
    """把已有的 PNG 序列编成 mp4(--encode-only 走这条)。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("  ! 找不到 ffmpeg。pip install imageio-ffmpeg 可解决")
        return
    subprocess.run([ffmpeg, "-y", "-framerate", str(fps), "-i",
                    str(frames_dir / "%05d.png"), *X264, str(out_path)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  video -> {out_path}")


def encode_stream(out_path, fps, frames):
    """边画边往 ffmpeg 的 stdin 喂裸 RGB。

    落 PNG 再编码一遍要多花十几分钟、多占几个 G(这三段一共 2.8 GB 中间文件),
    而改一句字幕就得整段重渲。管道省掉这一趟。要 PNG 序列加 --png。
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    p = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-framerate", str(fps), "-i", "-", *X264, str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for img in frames:
        p.stdin.write(img.tobytes())
    p.stdin.close()
    p.wait()
    print(f"  video -> {out_path}")
    return True


def run_clip(tag, a):
    src = SRC / tag
    clip = json.loads((src / "meta.json").read_text())
    copy_path = OUT / f"{tag}.copy.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if not copy_path.exists():
        Copy.template(clip, copy_path)
        print(f"  文案草稿 -> {copy_path}  (改完再跑一次即可,不会覆盖)")
    copy = Copy(clip, copy_path)
    reasoning = Reasoning(clip, OUT / f"{tag}.reasoning.json")
    strips = [Image.open(src / f"f{fr['fi']:05d}.png").convert("RGB")
              for fr in clip["frames"]]
    timeline = Timeline(clip, a)

    n_bad = sum(1 for s in timeline.shots if s["sev"] == "bad")
    n_warn = sum(1 for s in timeline.shots if s["sev"] == "warn")
    print(f"[{tag}] {clip['n']} 帧 -> {timeline.duration:.1f}s @ {a.fps}fps "
          f"= {timeline.n_frames} 帧画面   红框 {n_bad} 帧 · 琥珀 {n_warn} 帧")

    if a.preview:
        for fi in [int(x) for x in a.preview.split(",")]:
            i = next(k for k, fr in enumerate(clip["frames"]) if fr["fi"] == fi)
            k = int(timeline.shots[i]["t0"] * a.fps) + int(a.fps * 0.5)
            p = OUT / f"{tag}_preview_f{fi}.png"
            draw_frame(clip, strips, copy, reasoning, timeline, k, a).save(p)
            print(f"  preview f{fi} -> {p}")
        return

    frames_dir = OUT / tag / "frames"
    if a.encode_only:
        encode(frames_dir, OUT / f"{tag}.mp4", a.fps)
        return

    def render():
        for k in range(timeline.n_frames):
            img = draw_frame(clip, strips, copy, reasoning, timeline, k, a)
            if a.png:
                img.save(frames_dir / f"{k:05d}.png")
            yield img

    if a.png:
        frames_dir.mkdir(parents=True, exist_ok=True)
    if not encode_stream(OUT / f"{tag}.mp4", a.fps, render()):
        print("  ! 找不到 ffmpeg,退回写 PNG 序列。pip install imageio-ffmpeg 可解决")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for k in range(timeline.n_frames):
            draw_frame(clip, strips, copy, reasoning, timeline, k, a).save(frames_dir / f"{k:05d}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default=",".join(USABLE), help="逗号分隔的 clip tag")
    p.add_argument("--fps", type=int, default=28, help="编码帧率;内容变化率由 --speed 决定")
    # 节奏按【倍速】给,不按秒 —— 源是 2 Hz,占屏秒数 = 0.5 / 倍速
    p.add_argument("--speed", type=float, default=0.5, help="普通帧倍速(0.5 = 每帧 1.0s)")
    p.add_argument("--speed-warn", type=float, default=0.35, help="该走不走的帧")
    p.add_argument("--speed-bad", type=float, default=0.35, help="撞车 / 出界的帧")
    p.add_argument("--lead-in", type=float, default=0.0, help="首帧额外停顿")
    p.add_argument("--lead-out", type=float, default=0.0, help="末帧额外停顿")
    p.add_argument("--card-sec", type=float, default=2.0, help="片尾分数卡秒数,0=不要")
    p.add_argument("--flash", type=float, default=0.45, help="边框加粗持续秒数")
    p.add_argument("--caption-wrap", type=int, default=72)
    p.add_argument("--preview", default="", help="只渲染这些原始帧号,逗号分隔")
    p.add_argument("--warn-border", action="store_true",
                   help='给该走不走的帧也画黄框(默认不画,见 draw_frame 注释)')
    p.add_argument("--plan-chip", action="store_true",
                   help="在 reasoning 面板上挂 meta-action 的框(DECELERATE,STRAIGHT)")
    p.add_argument("--png", action="store_true", help="同时留一份 PNG 序列(很占盘)")
    p.add_argument("--encode-only", action="store_true",
                   help="不重渲,直接把已有的 PNG 序列编成 mp4")
    a = p.parse_args()
    for tag in a.clips.split(","):
        run_clip(tag.strip(), a)


if __name__ == "__main__":
    main()
