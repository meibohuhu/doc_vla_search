#!/usr/bin/env python
"""每帧一张 BEV:路面场景 + 周围车 + 三条轨迹。

为什么必须有这张图:前视相机看不到车前 ~6.5 m 以内(v=1080 对应约 6.5 m),
5s 位移短于这个数的轨迹一个点都投不出来。"该走不走"的片段里 baseline 只预测
1~2 m,整条都在画面外 —— 只看相机会以为它没输出。BEV 没有这个盲区。

只借 viz_case_study.map_layers 取地图要素,不用它的 draw_bev:那份是给论文半栏图
写的,目标筛选写死成 -4 < x < 32 且 |y| < 12,窗口比它大时窗口里的车会被悄悄丢掉
(stopsign_pullaway 23 辆只画了 7 辆)。这里按【当前窗口】筛,不写死。

画面上只有三类东西:路面底图、周围目标框、三条轨迹。车身扫掠框默认不画 —— 一帧
两个模型各三个框,叠在轨迹上反而看不清轨迹本身。

坐标:自车系 x 前 y 左;画面上 横轴 = -y(右手边在右), 纵轴 = x(前方朝上)。

  python scripts/make_video/render_clip_bev.py
  python scripts/make_video/render_clip_bev.py --clips-filter lot_leftturn --agents all
"""
import os, sys, json, glob, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

REPO = "/home/nvidia/workspace/other_repo/AutoVLA"
DATA = "/data/autovla_data/nuplan"

# 细底图要走 nuplan 的地图 API(navsim.common.dataclasses.Scene._build_map_api),
# 没有这几项就只能退回 roadblock 级的粗底图。在 import 之前设好。
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{DATA}/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("OPENSCENE_DATA_ROOT", DATA)
sys.path.insert(0, f"{REPO}/navsim")
sys.path.insert(0, f"{REPO}/scripts/0914")

JSON_DIR = f"{DATA}/navtest_nocot"
LOGS = f"{DATA}/navsim_logs/test"
DUMP_ROOT = "/data/autovla_data/eval/nuplan/case_study/videoclips"

# 与 render_clip_frames.py / 论文图同一套主色
C_GT, C_OURS, C_BASE = "#27b042", "#38bdf8", "#e08a5c"
# 底图:路面比周围【亮】,这样可行驶区域自己会跳出来(论文图那版反过来,整张糊成一片米色)
OFFROAD = "#e5e3dc"
M_ROAD, M_INTER, M_PARK = "#f8f7f4", "#f2f0ec", "#efeee9"
M_WALK, M_BOUND, M_CENTER, M_STOP = "#dedbd2", "#cfcbbe", "#d8d4c7", "#bdb8a9"
GRID = "#dcdad3"
EGO_FC, EGO_EC = "#ffffff", "#16181c"
VEH_FC, VEH_EC = "#c9c6bf", "#8d8a82"
PED_C, CONE_C = "#6f6c64", "#c9a227"

EGO_L, EGO_W = 5.176, 2.297      # nuPlan ego 几何(m)
EGO_BACK = 1.127                 # 后轴到车尾
SIDE = {"ours": "o", "base": "b"}


def load_dumps(name):
    """token -> 10x3 轨迹(x, y, heading)。扫掠框要用 heading,不能只取前两列。"""
    out = {}
    for f in sorted(glob.glob(f"{DUMP_ROOT}/{name}/dump/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out[r["token"]] = np.array(r["trajectory"], dtype=float)
    return out


def load_metrics(name):
    """本次 dump 的 PDMS 子分。碰撞只认 PDMS 的 NC=0,不自己按静态框几何推断 ——
    PDMS 是闭环仿真、前车也在动,静态几何会在"只超 TTC、没撞"的帧上误报。"""
    csvs = glob.glob(f"{DUMP_ROOT}/{name}/**/*.csv", recursive=True)
    if not csvs:
        return {}
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df.token.notna()].drop_duplicates("token").set_index("token")
    return {t: r.to_dict() for t, r in df.iterrows()}


def box(ax, x, y, l, w, yaw, fc, ec, lw, z, alpha=1.0):
    """在【框心】(x,y) 按 yaw 摆一个 l×w 的车身框。

    anns 里的 gt_boxes 给的就是框心,直接用。自车和预测位姿不是 —— 见 ego_box。
    """
    r = Rectangle((-l / 2, -w / 2), l, w, facecolor=fc, edgecolor=ec, lw=lw,
                  zorder=z, alpha=alpha)
    # 先把框摆成"车头朝上"(+90°),再叠上 yaw,最后平移到画面坐标 (-y, x)
    r.set_transform(Affine2D().rotate(yaw).rotate_deg(90).translate(-y, x) + ax.transData)
    ax.add_patch(r)


# 后轴 -> 车身几何中心的距离。轨迹的每个位姿都是【后轴】位置(与 metric cache 的
# ego_state.rear_axle 同一口径),车身框心在它前方 1.46 m 处,不做这个偏移,
# 扫掠框会整体比自车框偏后半个车身。
REAR2CENTER = EGO_L / 2 - EGO_BACK


def pose_box(ax, x, y, yaw, fc, ec, lw, z, alpha=1.0):
    """按一个【后轴位姿】摆车身框。"""
    box(ax, x + REAR2CENTER * np.cos(yaw), y + REAR2CENTER * np.sin(yaw),
        EGO_L, EGO_W, yaw, fc, ec, lw, z, alpha)


def window(fdatas, a):
    """整段固定一个窗口:所有轨迹 + 自车,外扩 margin,再补到目标长宽比。

    逐帧重算会让 BEV 在视频里晃,所以整段只算一次。
    """
    hs, vs = [0.0], [-EGO_BACK, EGO_L - EGO_BACK]
    for fd in fdatas:
        for arr in (fd["gt"], fd["ours"], fd["base"]):
            hs += list(-arr[:, 1])
            vs += list(arr[:, 0])
    hmin, hmax = min(hs) - a.margin, max(hs) + a.margin
    vmin, vmax = min(vs) - a.margin, max(vs) + a.margin
    aspect = a.height / a.width
    hspan, vspan = hmax - hmin, vmax - vmin
    if vspan / hspan < aspect:
        need = hspan * aspect
        c = (vmin + vmax) / 2
        vmin, vmax = c - need / 2, c + need / 2
    else:
        need = vspan / aspect
        c = (hmin + hmax) / 2
        hmin, hmax = c - need / 2, c + need / 2
    return (hmin, hmax), (vmin, vmax)


def draw_map(ax, log, token, a):
    """路面场景。要素从 viz_case_study.map_layers 取(nuplan 地图 API),画法是这里自己的。

    只画"面"和车道边界,不画中心线 —— 路口处几十条虚线交织,轨迹会被埋掉。
    """
    if a.map == "none":
        return
    from viz_case_study import map_layers, drivable_polys
    s = a.scale
    L = map_layers(log, token, a.map_radius) if a.map == "rich" else {}
    if not L:
        # 退回 roadblock 级的粗多边形(PDMS 打 DAC 分用的同一批)
        for poly in drivable_polys(log, token):
            ax.fill(-poly[:, 1], poly[:, 0], facecolor=M_ROAD,
                    edgecolor=M_BOUND, lw=0.7 * s, zorder=0)
        return
    for key, fc in (("road", M_ROAD), ("inter", M_INTER), ("park", M_PARK)):
        for poly in L[key]:
            ax.fill(-poly[:, 1], poly[:, 0], facecolor=fc, edgecolor="none", zorder=0)
    for poly in L["walk"] + L["cross"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=M_WALK, edgecolor="none", zorder=1)
    for poly in L["stop"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=M_STOP, edgecolor="none",
                alpha=0.75, zorder=1)
    # 只用普通车道的边界:lane_connector 的边界在路口会扇形交织成一团网,盖掉轨迹
    for ln in (L["bound"] if a.lane_connectors else L["bound_lane"]):
        ax.plot(-ln[:, 1], ln[:, 0], color=M_BOUND, lw=0.9 * s, zorder=1,
                solid_capstyle="round")
    if a.centerlines:
        for ln in L["center"] + L["center_route"]:
            ax.plot(-ln[:, 1], ln[:, 0], color=M_CENTER, lw=0.7 * s,
                    ls=(0, (2.5, 2.5)), zorder=1)


def draw(ax, fd, log, token, anns, xlim, ylim, a):
    ax.set_facecolor(OFFROAD)
    s = a.scale

    draw_map(ax, log, token, a)

    if a.grid:                                   # 没有底图时唯一的比例尺参照
        for v in np.arange(np.ceil(xlim[0] / 5) * 5, xlim[1], 5):
            ax.axvline(v, color=GRID, lw=0.8 * s, zorder=0)
        for v in np.arange(np.ceil(ylim[0] / 5) * 5, ylim[1], 5):
            ax.axhline(v, color=GRID, lw=0.8 * s, zorder=0)

    # --- 周围目标:按【当前窗口】筛,不写死范围 ---
    boxes, names = np.array(anns["gt_boxes"]), np.array(anns["gt_names"])
    for bx, nm in zip(boxes, names):
        x, y, _, l, w, _, yaw = bx[:7]
        if not (xlim[0] < -y < xlim[1] and ylim[0] < x < ylim[1]):
            continue
        if nm == "vehicle":
            box(ax, x, y, l, w, yaw, VEH_FC, VEH_EC, 1.0 * s, 2)
        elif a.agents == "all":
            if nm == "bicycle":
                box(ax, x, y, max(l, 1.6), max(w, 0.6), yaw, "#cdbfa8", VEH_EC, 0.7 * s, 3)
            elif nm == "pedestrian":
                ax.plot(-y, x, "o", ms=4.5 * s, color=PED_C, zorder=3,
                        markeredgecolor=M_ROAD, markeredgewidth=0.8 * s)
            elif nm in ("traffic_cone", "czone_sign", "barrier"):
                ax.plot(-y, x, "^", ms=4 * s, color=CONE_C, zorder=3,
                        markeredgecolor=M_ROAD, markeredgewidth=0.6 * s)

    # --- 自车(后轴在原点) ---
    pose_box(ax, 0.0, 0.0, 0.0, EGO_FC, EGO_EC, 1.6 * s, 4)

    # --- 车身扫掠框:模型"不动"时轨迹缩成自车框里的一个点,只有这十个位姿的车身
    #     整整齐齐叠在原地才看得出来。按预测位姿直接摆,不等于 PDMS 的判定几何
    #     (PDMS 判的是 LQR 闭环仿真后的位姿),只作示意。 ---
    if a.footprints:
        for key, colour in (("base", C_BASE), ("ours", C_OURS)):
            for x, y, h in fd[key][2::3]:
                pose_box(ax, x, y, h, "none", colour, 0.9 * s, 5, alpha=0.55)

    # --- 三条轨迹。GT 画成更粗的浅色底带,两个预测叠在上面。
    #     两个预测谁在上按【5s 位移】定:短的画在上面。模型"该走不走"时轨迹缩成
    #     自车原点上的一个点,固定次序的话它会被另一条整段盖死,一帧都看不见。---
    order = sorted(("ours", "base"), key=lambda k: -float(np.hypot(*fd[k][-1, :2])))
    layers = [("gt", C_GT, 5.0, 0.0, 6, 0.55)]
    layers += [(k, C_OURS if k == "ours" else C_BASE, 2.4, 5.0, 7 + i, 1.0)
               for i, k in enumerate(order)]
    for key, colour, lw, ms, z, alpha in layers:
        p = np.vstack([[0, 0], fd[key][:, :2]])
        ax.plot(-p[:, 1], p[:, 0], color=colour, lw=lw * s, zorder=z, alpha=alpha,
                solid_capstyle="round")
        if ms:
            ax.plot(-p[1:, 1], p[1:, 0], linestyle="none", marker="o", ms=ms * s,
                    color=colour, zorder=z, markeredgecolor="#ffffff", markeredgewidth=1.0 * s)

    # --- 碰撞:只认 PDMS 的 NC=0 ---
    for key, colour in (("base", C_BASE), ("ours", C_OURS)):
        if "collision" in fd[f"flags_{key}"]:
            ax.plot(-fd[key][-1, 1], fd[key][-1, 0], "X", ms=14 * s, color=colour,
                    markeredgecolor="#ffffff", markeredgewidth=1.6 * s, zorder=10)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d4d3cd")


def frame_data(token, dumps, metrics):
    j = json.load(open(f"{JSON_DIR}/{token}.json"))
    fd = {"gt": np.array(j["gt_trajectory"], dtype=float)}
    for key in SIDE:
        fd[key] = dumps[key][token]
        m = metrics[key].get(token, {})
        flags = []
        if m.get("no_at_fault_collisions") == 0:
            flags.append("collision")
        if m.get("drivable_area_compliance") == 0:
            flags.append("off-road")
        fd[f"flags_{key}"] = flags
    return fd


def main(a):
    clips = json.load(open(a.clips))
    if a.clips_filter:
        want = set(a.clips_filter.split(","))
        clips = [c for c in clips if c["tag"] in want]
    dumps = {k: load_dumps(k) for k in SIDE}
    metrics = {k: load_metrics(k) for k in SIDE}

    dpi = 100
    for c in clips:
        log_frames = pickle.load(open(f"{LOGS}/{c['log']}.pkl", "rb"))
        fdatas = [frame_data(t, dumps, metrics) for t in c["tokens"]]
        xlim, ylim = window(fdatas, a)
        dst = os.path.join(a.out, c["tag"], "bev")
        os.makedirs(dst, exist_ok=True)
        n_veh = 0
        for fr, fd in zip(c["frames"], fdatas):
            fig = plt.figure(figsize=(a.width / dpi, a.height / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            anns = log_frames[fr["fi"]]["anns"]
            draw(ax, fd, c["log"], fr["token"], anns, xlim, ylim, a)
            n_veh += sum(1 for bx, nm in zip(np.array(anns["gt_boxes"]),
                                             np.array(anns["gt_names"]))
                         if nm == "vehicle" and xlim[0] < -bx[1] < xlim[1]
                         and ylim[0] < bx[0] < ylim[1])
            fig.savefig(os.path.join(dst, f"f{fr['fi']:05d}.png"), facecolor=OFFROAD, dpi=dpi)
            plt.close(fig)
        print(f"  {c['tag']:20s} {c['n']:2d} 帧  窗口 {xlim[1]-xlim[0]:4.0f}×{ylim[1]-ylim[0]:4.0f} m"
              f"  共画车辆 {n_veh:4d} 个框  -> {dst}", flush=True)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default=f"{REPO}/logs/0920_nvidia/video_clips.json")
    p.add_argument("--clips-filter", default="", help="只渲染这些 tag,逗号分隔")
    p.add_argument("--out", default=f"{REPO}/logs/0920_nvidia/video_sources")
    p.add_argument("--width", type=int, default=480, help="输出像素宽")
    p.add_argument("--height", type=int, default=540, help="输出像素高(默认与相机条带等高)")
    p.add_argument("--scale", type=float, default=1.4, help="线宽/marker 缩放")
    p.add_argument("--margin", type=float, default=4.0, help="窗口在轨迹外留的边距(m)")
    p.add_argument("--agents", choices=["vehicle", "all"], default="vehicle",
                   help="vehicle=只画车;all=加上行人/自行车/锥桶")
    p.add_argument("--map", choices=["rich", "plain", "none"], default="rich",
                   help="rich=车道级底图; plain=roadblock 级粗多边形; none=不画")
    p.add_argument("--map-radius", type=float, default=60.0, help="底图取多大范围(m)")
    p.add_argument("--centerlines", action="store_true",
                   help="加上车道中心线虚线(路口处会很乱,默认不画)")
    p.add_argument("--lane-connectors", action="store_true",
                   help="连 lane_connector 的边界一起画(路口会交织成一团网,默认不画)")
    p.add_argument("--grid", action="store_true", help="画 5 m 方格(没有底图时当比例尺)")
    p.add_argument("--footprints", action="store_true",
                   help="画预测位姿的车身扫掠框(一帧多 6 个框,默认不画)")
    main(p.parse_args())
