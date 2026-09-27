#!/usr/bin/env python
"""把 GT / Ours / Baseline 三条轨迹渲染到每一帧的三路相机上,输出视频源帧。

版式(1920x540,每帧一张 PNG):

    ┌── CAM_L0 (裁右半) ──┬──────── CAM_F0 (全幅) ────────┬── CAM_R0 (裁左半) ──┐
    │       474 px        │            960 px             │       474 px        │

三路各用自己的 cam_intrinsic + sensor2lidar_* 投影,所以转弯时冲出前视图的那截
会接着出现在侧视图里。先在原始分辨率(3840x1080)上画,再缩到一半 —— PIL 的线没有
抗锯齿,降采样是最省事的补偿。

轨迹来源:
  GT        navtest_nocot/<token>.json 的 gt_trajectory
  Ours/Base run_clip_dump.sh 落盘的 dump/*.jsonl 的 trajectory
都是自车系 10 个点 x 0.5s = 5s。

  python scripts/make_video/render_clip_frames.py
  python scripts/make_video/render_clip_frames.py --clean --clips lead_stop,lead_truck
"""
import os, sys, json, glob, pickle, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = "/home/nvidia/workspace/other_repo/AutoVLA"
sys.path.insert(0, f"{REPO}/scripts/0914")
from viz_case_halfcolumn import projector          # 标定投影,与论文图同一套

JSON_DIR = "/data/autovla_data/nuplan/navtest_nocot"
DUMP_ROOT = "/data/autovla_data/eval/nuplan/case_study/videoclips"

# 配色与论文图一致(viz_case_study.py)
C_GT, C_OURS, C_BASE = (39, 176, 66), (56, 189, 248), (224, 138, 92)
WHITE, INK = (255, 255, 255), (16, 18, 22)

# 原始分辨率下的版式:前视全幅,左右各留 948 px(靠近前视的一侧)
CAM_W, CAM_H = 1920, 1080
SIDE_W, GAP = 948, 12
STRIP_W = SIDE_W + GAP + CAM_W + GAP + SIDE_W        # 3840
SCALE = 0.5                                          # 输出 1920x540

# (相机 key, navtest json 字段, 源图里保留的 x 区间, 贴到条带上的 x)
SLOTS = [("CAM_L0", "front_left_camera_paths", (CAM_W - SIDE_W, CAM_W), 0),
         ("CAM_F0", "front_camera_paths", (0, CAM_W), SIDE_W + GAP),
         ("CAM_R0", "front_right_camera_paths", (0, SIDE_W), SIDE_W + GAP + CAM_W + GAP)]

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(f"{FONT_DIR}/DejaVuSans{'-Bold' if bold else ''}.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def load_dumps(name):
    """token -> (预测轨迹, <think> 原文)。重复出现以最后一条为准。"""
    out = {}
    for f in sorted(glob.glob(f"{DUMP_ROOT}/{name}/dump/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[r["token"]] = (np.array(r["trajectory"], dtype=float)[:, :2],
                               r.get("raw_output", ""))
    return out


SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance"]


def load_run_scores(name):
    """本次 dump 跑出来的 PDMS 总分 + 子分。

    贪心解码也不是逐位复现的(kernel/batch 组成会变),88 帧里有几帧和全量 eval 对不上。
    画面上的分数、红框判据都必须来自【画出来的这条轨迹】那次推理,所以用这份;
    全量 eval 那份另存成 *_full 备查。
    """
    import pandas as pd
    csvs = glob.glob(f"{DUMP_ROOT}/{name}/**/*.csv", recursive=True)
    if not csvs:
        return {}
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df.token.notna()].drop_duplicates("token").set_index("token")
    return {t: dict(score=round(float(r["score"]) * 100, 1),
                    **{k: float(r[k]) for k in SUB if k in r})
            for t, r in df.iterrows()}


def polyline(draw, uv, colour, lw, r):
    """画一条折线 + 节点(r<=0 则不画节点)。NaN(相机后方)处断开,越界交给 PIL 裁。"""
    run = []
    for p in uv:
        if np.isnan(p[0]):
            if len(run) > 1:
                draw.line([tuple(q) for q in run], fill=colour, width=lw, joint="curve")
            run = []
        else:
            run.append(p)
    if len(run) > 1:
        draw.line([tuple(q) for q in run], fill=colour, width=lw, joint="curve")
    if r <= 0:              # r=0 只画线不画点(GT 用),与 BEV 里 GT 的画法一致
        return
    for p in uv:
        if not np.isnan(p[0]):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=colour,
                         outline=WHITE, width=max(1, lw // 4))


def render_frame(clip, fr, trajs, a):
    log = clip["log"]
    j = json.load(open(f"{JSON_DIR}/{fr['token']}.json"))
    strip = Image.new("RGB", (STRIP_W, CAM_H), (10, 12, 15))

    for cam, field, (x0, x1), paste_x in SLOTS:
        tile = Image.open(j[field][-1]).convert("RGB").crop((x0, 0, x1, CAM_H))
        draw = ImageDraw.Draw(tile)
        proj = projector(log, fr["fi"], cam=cam)
        # GT 不画节点:它是参照底带,画上点会和两个预测的节点混成一片
        for key, colour, lw, r in (("gt", C_GT, 14, 0),
                                   ("base", C_BASE, 10, 8),
                                   ("ours", C_OURS, 10, 8)):
            traj = trajs.get(key)
            if traj is None:
                continue
            uv = proj(np.vstack([[0.0, 0.0], traj]))       # 从车身原点起画
            uv[:, 0] -= x0                                  # 裁切后的图像坐标
            polyline(draw, uv, colour, lw, r)
        strip.paste(tile, (paste_x, 0))

    out = strip.resize((int(STRIP_W * SCALE), int(CAM_H * SCALE)), Image.LANCZOS)
    if not a.clean:
        annotate(out, clip, fr, trajs)
    return out


MIN_VISIBLE = 3.0   # 5s 位移小于这个值时,整条轨迹基本落在相机视野下方


def annotate(img, clip, fr, trajs):
    d = ImageDraw.Draw(img, "RGBA")
    f_b, f_s = font(22, True), font(18)
    def badge(x, text, fnt):
        d.rectangle([x, 8, x + 12 + d.textlength(text, font=fnt) + 12, 42],
                    fill=(0, 0, 0, 165))
        d.text((x + 12, 13), text, font=fnt, fill=WHITE)

    # 帧号留在最左(左视图上角);自车速度挪到中间那张前视图的左上角 ——
    # 速度是看轨迹时要对照的数,放在前视图里眼睛不用来回跑。
    badge(8, f"f{fr['fi']}", f_b)
    badge(int(SLOTS[1][3] * SCALE) + 8, f"v={fr['speed']:.1f} m/s", f_b)
    # 右上:图例 + 5s 位移。位移是这里最要紧的读数 —— 模型"该走不走"时预测轨迹只有
    # 一两米,整条都在车头下方、相机根本拍不到,只有这个数字能说明它停住了。
    rows = [("GT", C_GT, float(np.hypot(*trajs["gt"][-1])))]
    # 名字要和视频里两个面板的标题一致(那边来自 <tag>.copy.json 的 labels),
    # 否则同一个模型在一帧里会有两个叫法
    for key, name, colour in (("ours", "Ours", C_OURS), ("base", "SFT model", C_BASE)):
        if key in trajs:
            rows.append((name, colour, float(np.hypot(*trajs[key][-1]))))
    f_leg, f_note = font(24, True), font(18)
    note = any(r[2] < MIN_VISIBLE for r in rows[1:])   # ↓ = 轨迹短到出了相机视野
    w, lh = 330, 36
    # 注脚画进框里。原来它落在框外面,压在相机画面上,浅灰字叠在天空上几乎看不清
    h = 12 + lh * len(rows) + (26 if note else 0)
    d.rectangle([img.width - w - 8, 8, img.width - 8, 8 + h], fill=(0, 0, 0, 175))
    y = 14
    for name, colour, dist in rows:
        d.line([img.width - w + 6, y + 13, img.width - w + 44, y + 13],
               fill=colour, width=7)
        d.text((img.width - w + 56, y), name, font=f_leg, fill=WHITE)
        tail = f"{dist:.1f} m" + ("  ↓" if dist < MIN_VISIBLE else "")
        d.text((img.width - 18, y), tail, font=f_leg, fill=colour, anchor="ra")
        y += lh
    if note:
        d.text((img.width - 18, y - 2), "↓ below camera FOV", font=f_note,
               fill=(190, 196, 204), anchor="ra")


def main(a):
    clips = json.load(open(a.clips))
    if a.clips_filter:
        want = set(a.clips_filter.split(","))
        clips = [c for c in clips if c["tag"] in want]
    dumps = {k: load_dumps(k) for k in ("ours", "base")}
    runsc = {k: load_run_scores(k) for k in ("ours", "base")}
    print(f"dump: ours {len(dumps['ours'])} 帧, base {len(dumps['base'])} 帧")

    for c in clips:
        dst = os.path.join(a.out, c["tag"])
        os.makedirs(dst, exist_ok=True)
        missing = 0
        for fr in c["frames"]:
            tok = fr["token"]
            j = json.load(open(f"{JSON_DIR}/{tok}.json"))
            trajs = {"gt": np.array(j["gt_trajectory"], dtype=float)[:, :2]}
            for k, side in (("ours", "o"), ("base", "b")):
                if tok in dumps[k]:
                    trajs[k], raw = dumps[k][tok]
                    fr[f"raw_{k}"] = raw
                    fr[f"disp5s_{side}"] = round(float(np.hypot(*trajs[k][-1])), 2)
                else:
                    missing += 1
                # 全量 eval 的总分和子分都留作对照,画面上用的是本次这份
                fr[f"pdms_{side}_full"] = fr.pop(f"pdms_{side}")
                fr[f"{side}_full"] = fr.pop(side)
                row = runsc[k].get(tok) or {}
                fr[f"pdms_{side}"] = row.get("score")
                fr[side] = {m: row[m] for m in SUB if m in row}
            fr["disp5s_gt"] = round(float(np.hypot(*trajs["gt"][-1])), 2)
            render_frame(c, fr, trajs, a).save(os.path.join(dst, f"f{fr['fi']:05d}.png"))
        for side in ("o", "b"):
            got = [f[f"pdms_{side}"] for f in c["frames"] if f[f"pdms_{side}"] is not None]
            c[f"pdms_{'o' if side == 'o' else 'b'}"] = round(float(np.mean(got)), 1) if got else None
        json.dump(c, open(os.path.join(dst, "meta.json"), "w"), indent=1, ensure_ascii=False)
        print(f"  {c['tag']:20s} {c['n']:2d} 帧  Ours {c['pdms_o']:5.1f} / Base {c['pdms_b']:5.1f}"
              f"  -> {dst}" + (f"   ⚠️ 缺 {missing} 条预测轨迹" if missing else ""))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default=f"{REPO}/logs/0920_nvidia/video_clips.json")
    p.add_argument("--clips-filter", default="", help="只渲染这些 tag,逗号分隔")
    p.add_argument("--out", default=f"{REPO}/logs/0920_nvidia/video_sources")
    p.add_argument("--clean", action="store_true", help="不画帧号/图例,只留轨迹")
    main(p.parse_args())
