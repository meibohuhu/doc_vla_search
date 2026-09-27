#!/usr/bin/env python
"""
Case-study 图:同一场景连续 3 帧，好 / 差两个 CoT checkpoint 的 reasoning 与轨迹对比。

  好 ckpt = 103k CoT ep4  (navtest PDMS 80.45)
  差 ckpt = 166k CoT ep2  (navtest PDMS 74.65)

三行:
  A 3 路相机输入(front_left | front | front_right，当前帧)  —— AutoVLA 实际输入的视图
  B BEV:GT / best / worse 轨迹 + 周围目标框
  C 两个模型的 <think> 原文 + 该帧 PDMS 子分

用法: python scripts/0914/viz_case_study.py
"""
import os, re, json, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.transforms import Affine2D
import pandas as pd
from PIL import Image

# ---- 配色:dataviz 参考调色板前三槽(已验证 all-pairs) ----
C_BEST, C_WORSE, C_GT = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84"
BOX_OTHER, BOX_LEAD = "#dedcd6", "#a8a59b"

JSON_DIR = "/data/autovla_data/nuplan/navtest_nocot"
CACHE_ROOT = "/data/autovla_data/nuplan/navtest_metric_cache"

# ---- case 注册表（与 run_case_study_3frames.sh 的 CASE 同名）----
CASES = {
    "leadstop": dict(
        tag="casestudy3f",
        log="2021.05.25.14.16.10_veh-35_00083_00485",
        frames=[("135ae32b6edc55c5", 656), ("974a70027f8f593c", 657), ("ed98a4566ea95092", 658)],
        title="Same scene, same perception — the reasoning step decides the outcome",
        sub1="nuPlan navtest · log 2021.05.25.14.16.10_veh-35_00083_00485 · three consecutive frames (0.5 s apart).  "
             "Ego is nearly stopped (2.0 m/s) behind a slow lead vehicle in a crowded construction zone.",
        sub2="Both checkpoints are CoT-trained and perceive the same lead vehicle. "
             "The better checkpoint keeps the causal step “I must stay behind it → remain stopped”; "
             "the weaker one drops it and decides to keep speed — and rear-ends the lead vehicle.",
        xlim=(-10, 10), ylim=(-3.5, 22)),
    "leftturn_ped": dict(
        tag="casestudy3f_leftturn_ped",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("260f5d5245015db6", 130), ("abb9477dd3305951", 131), ("2a94741039ad566d", 132)],
        title="Unprotected left turn with pedestrians — same observation, different plan token",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · three consecutive frames (0.5 s apart).  "
             "Ego is stopped (0.1 m/s) at an intersection with 9 pedestrians and 4 vehicles within 25 m; the expert turns left.",
        sub2="Both checkpoints report the same lead vehicle. The weaker one first refuses to move at all (progress 0.02), "
             "then commits to ACCELERATE,LEFT one frame too early and cuts outside the drivable area.",
        xlim=None, ylim=None),
    "leftturn_ped129": dict(   # 同一场景往前挪一帧:129/130/131
        tag="casestudy3f_leftturn_ped129",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("c672f1584cb75697", 129), ("260f5d5245015db6", 130), ("abb9477dd3305951", 131)],
        title="Left turn at an intersection — the weaker checkpoint stalls for 1 s, then lurches left off the road",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · three consecutive frames (0.5 s apart).  "
             "Ego is stopped (0.1 m/s) at an intersection with 9 pedestrians and 4 vehicles within 25 m; the expert pulls away and turns left.",
        sub2="Both checkpoints state the same observation on every frame — “the vehicle 9 m ahead in my lane is stopped”. "
             "The better one concludes it should accelerate, and tracks the expert. The weaker one concludes it must stay behind and remain stopped, "
             "makes no progress for two frames, then over-corrects with ACCELERATE,LEFT and PDMS scores it off-road (DAC = 0).  "
             "Thin outlines are the ego footprint every 1.5 s — DAC is judged on the vehicle body, not the waypoints.",
        xlim=None, ylim=None),
    "leftturn_ped129_132": dict(   # 129/130/132（跳过 131）
        tag="casestudy3f_leftturn_ped129_132",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("c672f1584cb75697", 129), ("260f5d5245015db6", 130), ("2a94741039ad566d", 132)],
        title="Left turn at an intersection — the weaker checkpoint stalls, then catches up",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · frames 129, 130 and 132 (frame 131 omitted).  "
             "Ego is stopped (0.1 m/s) at an intersection with 9 pedestrians and 4 vehicles within 25 m; the expert pulls away and turns left.",
        sub2="Both checkpoints state the same observation — “the vehicle 9 m ahead in my lane is stopped”. "
             "The better one concludes it should accelerate and tracks the expert throughout. The weaker one concludes it must remain stopped and makes "
             "no progress on frames 129-130; by frame 132 it reaches the same decision and scores PDMS 100.  "
             "Thin outlines are the ego footprint every 1.5 s — DAC is judged on the vehicle body, not the waypoints.",
        xlim=None, ylim=None),
    "leftturn_ped131_133": dict(   # 131/132/133，数据来自 leftturn_ped_seq(129-134 全跑)
        tag="casestudy3f_leftturn_ped_seq",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("abb9477dd3305951", 131), ("2a94741039ad566d", 132), ("adc1a3a3dd1c501e", 133)],
        title="Left turn — the baseline over-corrects, then both converge",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · frames 131-133 (0.5 s apart).",
        sub2="Baseline commits to ACCELERATE,LEFT one frame early and PDMS scores it off-road; "
             "from frame 132 on both checkpoints agree and both score 100.",
        xlim=None, ylim=None),
    "leftturn_ped130_133": dict(   # 130/132/133
        tag="casestudy3f_leftturn_ped_seq",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("260f5d5245015db6", 130), ("2a94741039ad566d", 132), ("adc1a3a3dd1c501e", 133)],
        title="Left turn — baseline stalls, then catches up",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · frames 130, 132, 133.",
        sub2="Baseline makes no progress at frame 130; by 132-133 both checkpoints agree.",
        xlim=None, ylim=None),
    "leftturn_ped129_130_133": dict(   # 129/130/133，数据来自 leftturn_ped_seq(129-134 全跑)
        tag="casestudy3f_leftturn_ped_seq",
        log="2021.06.28.16.57.59_veh-26_00016_00484",
        frames=[("c672f1584cb75697", 129), ("260f5d5245015db6", 130), ("adc1a3a3dd1c501e", 133)],
        title="Left turn at an intersection — the baseline stalls, then catches up",
        sub1="nuPlan navtest · log 2021.06.28.16.57.59_veh-26_00016_00484 · frames 129, 130 and 133.",
        sub2="Both checkpoints report the same observation. The baseline concludes it must remain stopped and makes "
             "no progress on the first two frames; by frame 133 it has caught up and both score 100.",
        xlim=None, ylim=None),
    "rightturn_offroad": dict(
        tag="casestudy3f_rightturn_offroad",
        log="2021.06.03.13.55.17_veh-35_02572_02855",
        frames=[("c520f76d99f359f2", 549), ("154d0d1b363f5501", 550), ("655c3f17ee2d5683", 551)],
        title="Intersection right turn — identical reasoning, but the weaker model under-steers off the road",
        sub1="nuPlan navtest · log 2021.06.03.13.55.17_veh-35_02572_02855 · three consecutive frames (0.5 s apart).  "
             "Ego enters a 90° right turn at 6.0 m/s; the expert tracks the turn through the intersection.",
        sub2="Note both models emit nearly the same plan (“slow down and bear right”). Here the failure is NOT in the reasoning text "
             "but in the action tokens: the weaker checkpoint under-steers (−43° vs −79° heading) and PDMS scores it off-road (DAC = 0) on all three frames.  Thin outlines are the ego footprint every 1.5 s — DAC is judged on the vehicle body, not the waypoints.",
        xlim=None, ylim=None),
}


# nuPlan ego 车辆几何(m)
EGO_L, EGO_W, EGO_REAR2FRONT = 5.176, 2.297, 4.049


def load_dumps(work):
    d = {}
    for m in ("best", "worse"):
        for f in glob.glob(f"{work}/{m}/dump/*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                d[(m, r["token"])] = r
    return d


def load_scores(work):
    s = {}
    for m in ("best", "worse"):
        csvs = glob.glob(os.path.join(work, m, "**", "*.csv"), recursive=True)
        df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        df = df[df.token.notna()].drop_duplicates("token").set_index("token")
        s[m] = df
    return s


def build_case_data(case):
    """
    把一个 case 的所有【绘图需要的数值】收进一个 dict：三帧 x 两个 ckpt 的
    trajectory / think / plan / PDMS，加 GT 轨迹和相机图路径。
    这是从 eval dump + PDMS CSV 里如实导出的模型输出，导完后绘图不再碰原始产物。
    """
    cfg = CASES[case]
    work = f"/data/autovla_data/eval/nuplan/case_study/{cfg['tag']}"
    dumps, scores = load_dumps(work), load_scores(work)
    t0 = cfg["frames"][0][1]
    labels = {"best": "best ckpt · 103k CoT ep4", "worse": "worse ckpt · 166k CoT ep2"}
    # figure_label 只是图上的展示名(论文里 best=我们的方法, worse=SFT 基线)，可在 JSON 里改；
    # 它不是数据，改它不影响任何数值。
    fig_labels = {"best": "Ours", "worse": "SFT baseline"}
    frames = []
    for tok, fi in cfg["frames"]:
        j = json.load(open(os.path.join(JSON_DIR, f"{tok}.json")))
        fr = dict(token=tok, frame_idx=fi,
                  time_label=f"t = {'+' if fi > t0 else ''}{(fi - t0) * 0.5:.1f} s",
                  cameras={k: j[f"{k}_camera_paths"][-1]
                           for k in ("front_left", "front", "front_right")},
                  gt_trajectory=[list(map(float, p)) for p in j["gt_trajectory"]],
                  speed_mps=float(np.hypot(*j["velocity"][:2])),
                  models={})
        for m in ("best", "worse"):
            body, plan = split_plan(think_of(dumps[(m, tok)]["raw_output"]))
            row = scores[m].loc[tok]
            flags = []
            if row["no_at_fault_collisions"] == 0: flags.append("collision")
            if row["time_to_collision_within_bound"] == 0: flags.append("TTC violation")
            if row["drivable_area_compliance"] == 0: flags.append("off-road")
            fr["models"][m] = dict(
                label=labels[m], figure_label=fig_labels[m], think=body, plan=plan,
                trajectory=[list(map(float, p)) for p in dumps[(m, tok)]["trajectory"]],
                pdms=float(row["score"]) * 100, flags=flags)
        frames.append(fr)
    return dict(case=case, log=cfg["log"], tag=cfg["tag"], title=cfg["title"],
                sub1=cfg["sub1"], sub2=cfg["sub2"],
                xlim=cfg["xlim"], ylim=cfg["ylim"], frames=frames)


def drivable_polys(log, token):
    """
    从 navtest metric cache 取【PDMS 打 DAC 分时真正用的】可行驶区域多边形，
    转到自车系。自己按地图重画会和评分口径对不上，所以直接用 cache 里的。
    图层与 pdm_scorer._calculate_ego_area 一致:ROADBLOCK / INTERSECTION /
    DRIVABLE_AREA / CARPARK_AREA。
    """
    import lzma
    hit = glob.glob(os.path.join(CACHE_ROOT, log, "*", token, "metric_cache.pkl"))
    if not hit:
        return []
    with lzma.open(hit[0], "rb") as fh:
        mc = pickle.load(fh)
    dm = mc.drivable_area_map
    keep = {"ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA"}
    es = mc.ego_state.rear_axle
    ox, oy, oh = es.x, es.y, es.heading
    c, s = np.cos(-oh), np.sin(-oh)
    out = []
    for geom, mt in zip(dm._geometries, dm._map_types):
        if str(mt).split(".")[-1] not in keep:
            continue
        xy = np.asarray(geom.exterior.coords)
        dx, dy = xy[:, 0] - ox, xy[:, 1] - oy
        ex, ey = c * dx - s * dy, s * dx + c * dy       # 全局 -> 自车系
        if (np.abs(ex) < 80).any() and (np.abs(ey) < 80).any():
            out.append(np.stack([ex, ey], 1))
    return out


def think_of(raw):
    raw = raw if isinstance(raw, str) else str(raw)
    m = re.search(r"<think>(.*?)</think>", raw, re.S)
    return (m.group(1) if m else raw[:300]).strip()


def split_plan(txt):
    """把 '<PLAN>STOP,STRAIGHT</PLAN>' 拆出来单独强调。"""
    m = re.search(r"<PLAN>(.*?)</PLAN>", txt)
    plan = m.group(1) if m else ""
    body = re.sub(r"\s*:?\s*<PLAN>.*?</PLAN>", "", txt).strip().rstrip(":").strip()
    return body, plan


def wrap(s, n=54):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return "\n".join(out)


def box_patch(ax, x, y, l, w, yaw, fc, ec, lw=1.0, z=2, alpha=1.0):
    """BEV:横轴 = -y(左为正), 纵轴 = x(前)。车框中心 (x,y)、朝向 yaw。"""
    r = Rectangle((-l / 2, -w / 2), l, w, facecolor=fc, edgecolor=ec,
                  linewidth=lw, zorder=z, alpha=alpha)
    # 先在车体系画(长沿 +x)，再旋转 yaw，再映射到显示系 (h,v)=(-y,x)
    t = (Affine2D().rotate(yaw).translate(x, y)
         .scale(1, 1))
    # 显示系变换: h = -y, v = x  →  等价于旋转 -90° 后翻转
    disp = Affine2D().from_values(0, 1, -1, 0, 0, 0)   # (x,y) -> (-y, x)
    r.set_transform(t + disp + ax.transData)
    ax.add_patch(r)
    return r


# ---- 环境底图配色(整体压得很淡，保证轨迹是画面里最亮的东西) ----
MAP_ROAD, MAP_INTER, MAP_PARK = "#e9e7df", "#e0ddd2", "#f1efe9"
MAP_BOUND, MAP_CENTER, MAP_ROUTE = "#cbc7b9", "#d5d1c3", "#cdbf93"
MAP_STOP = "#b8b4a6"
AGENT_COL = {"vehicle": "#c3bfb4", "bicycle": "#b9c6c9", "pedestrian": "#8a8a84",
             "traffic_cone": "#c9a227", "barrier": "#c9a227", "czone_sign": "#c9a227",
             "generic_object": "#dedcd6"}

_MAP_CACHE = {}


def map_layers(log, token, radius=60.0):
    """
    取自车周边的地图要素并转到自车系(原点与 drivable_polys 一致 = metric cache 的 rear_axle)。
    返回 dict:road/inter/park 多边形、lane 左右边界与中心线、停止线、route 车道多边形。
    比 drivable_area_map 细得多 —— 那份只有 roadblock 级别的面，画出来是一整块灰。
    """
    key = (log, token, radius)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    import lzma
    from navsim.common.dataclasses import Scene
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as SL
    from nuplan.common.actor_state.state_representation import Point2D

    hit = glob.glob(os.path.join(CACHE_ROOT, log, "*", token, "metric_cache.pkl"))
    if not hit:
        return {}
    with lzma.open(hit[0], "rb") as fh:
        mc = pickle.load(fh)
    es = mc.ego_state.rear_axle
    ox, oy, oh = es.x, es.y, es.heading
    c, s_ = np.cos(-oh), np.sin(-oh)

    def to_ego(xy):
        xy = np.asarray(xy, dtype=float)
        dx, dy = xy[:, 0] - ox, xy[:, 1] - oy
        return np.stack([c * dx - s_ * dy, s_ * dx + c * dy], 1)

    frame = pickle.load(open(f"/data/autovla_data/nuplan/navsim_logs/test/{log}.pkl", "rb"))[0]
    api = Scene._build_map_api(frame["map_location"])
    pt = Point2D(ox, oy)
    want = [SL.ROADBLOCK, SL.ROADBLOCK_CONNECTOR, SL.INTERSECTION, SL.CARPARK_AREA,
            SL.LANE, SL.LANE_CONNECTOR, SL.STOP_LINE, SL.CROSSWALK, SL.WALKWAYS]
    got = {}
    for lay in want:
        try:
            got[lay] = api.get_proximal_map_objects(pt, radius, [lay])[lay]
        except Exception:
            got[lay] = []

    route = set(str(i) for i in getattr(mc, "route_lane_ids", []) or [])
    # bound      = LANE + LANE_CONNECTOR 的车道边界(原样保留,draw_map_rich 用的就是它)
    # bound_lane = 只有 LANE 的。路口处几十条 connector 边界会扇形交织成一团网,
    #              做视频底图时得把它们摘掉,所以额外给一份。
    out = dict(road=[], inter=[], park=[], walk=[], cross=[],
               bound=[], bound_lane=[], center=[], center_route=[], stop=[], route_poly=[])
    for lay, key_ in ((SL.ROADBLOCK, "road"), (SL.ROADBLOCK_CONNECTOR, "road"),
                      (SL.INTERSECTION, "inter"), (SL.CARPARK_AREA, "park"),
                      (SL.WALKWAYS, "walk"), (SL.CROSSWALK, "cross"),
                      (SL.STOP_LINE, "stop")):
        for o in got.get(lay, []):
            try:
                out[key_].append(to_ego(np.asarray(o.polygon.exterior.coords)))
            except Exception:
                pass
    for lay in (SL.LANE, SL.LANE_CONNECTOR):
        for o in got.get(lay, []):
            on_route = str(o.id) in route
            try:
                if on_route:
                    out["route_poly"].append(to_ego(np.asarray(o.polygon.exterior.coords)))
            except Exception:
                pass
            for side in ("left_boundary", "right_boundary"):
                b = getattr(o, side, None)
                if b is None:
                    continue
                try:
                    xy = to_ego([[q.x, q.y] for q in b.discrete_path])
                    out["bound"].append(xy)
                    if lay is SL.LANE:
                        out["bound_lane"].append(xy)
                except Exception:
                    pass
            try:
                cl = to_ego([[q.x, q.y] for q in o.baseline_path.discrete_path])
                out["center_route" if on_route else "center"].append(cl)
            except Exception:
                pass
    _MAP_CACHE[key] = out
    return out


def draw_map_rich(ax, log, token, s=1.0, radius=60.0):
    """画细粒度底图:路面/路口/停车场 -> 车道面(route 高亮) -> 车道线 -> 中心线 -> 停止线。"""
    L = map_layers(log, token, radius)
    if not L:
        return False
    for poly in L["road"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=MAP_ROAD, edgecolor="none", zorder=0)
    for poly in L["inter"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=MAP_INTER, edgecolor="none", zorder=0)
    for poly in L["park"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=MAP_PARK, edgecolor="none", zorder=0)
    for poly in L["walk"] + L["cross"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor="#e6e3d9", edgecolor="#d3cfc2",
                lw=0.3 * s, zorder=0)
    for poly in L["route_poly"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=MAP_ROUTE, alpha=0.22,
                edgecolor="none", zorder=1)
    for ln in L["bound"]:
        ax.plot(-ln[:, 1], ln[:, 0], color=MAP_BOUND, lw=0.75 * s, zorder=1,
                solid_capstyle="round")
    for ln in L["center"]:
        ax.plot(-ln[:, 1], ln[:, 0], color=MAP_CENTER, lw=0.55 * s,
                ls=(0, (2.5, 2.5)), zorder=1)
    for ln in L["center_route"]:
        ax.plot(-ln[:, 1], ln[:, 0], color="#b9a97a", lw=0.55 * s,
                ls=(0, (2.5, 2.5)), zorder=1)
    for poly in L["stop"]:
        ax.fill(-poly[:, 1], poly[:, 0], facecolor=MAP_STOP, edgecolor="none",
                alpha=0.7, zorder=1)
    return True


def draw_bev(ax, log, fdata, anns, xlim, ylim, s=1.0, ticks=True, labels=True,
             axis_label=True, ylabel=False, footprints=True, top="best", rich=False, radius=60.0):
    """
    画一格 BEV —— 大图和半栏图共用这一套画法，保证两处视觉一致。
      s          : 尺寸缩放(半栏图传 ~0.45，线宽/字号/marker 一起缩)
      ticks      : 是否显示刻度数字
      labels     : 是否标注三条轨迹端点 + 前车尾线文字
      axis_label : 是否写 "lateral (m)"
    """
    gt = np.array(fdata["gt_trajectory"])[:, :2]
    tb = np.array(fdata["models"]["best"]["trajectory"])[:, :2]
    tw = np.array(fdata["models"]["worse"]["trajectory"])[:, :2]
    ax.set_facecolor(SURFACE)
    drew = draw_map_rich(ax, log, fdata["token"], s=s, radius=radius) if rich else False
    if not drew:
        # 退回 roadblock 级的可行驶区域(PDMS 打 DAC 分用的同一批多边形)
        for poly in drivable_polys(log, fdata["token"]):
            ax.fill(-poly[:, 1], poly[:, 0], facecolor="#eceae2",
                    edgecolor="#cfccc0", lw=0.7 * s, zorder=0)
    # 周围目标
    boxes, names = np.array(anns["gt_boxes"]), np.array(anns["gt_names"])
    lead_i = None
    for i, (bx, nm) in enumerate(zip(boxes, names)):
        x0, y0, _, l, w, _, yaw = bx[:7]
        if not (-4 < x0 < 32 and abs(y0) < 12):
            continue
        if nm == "vehicle":
            is_lead = (x0 > 2 and abs(y0) < 2.2)
            if is_lead and (lead_i is None or x0 < boxes[lead_i][0]):
                lead_i = i
            box_patch(ax, x0, y0, l, w, yaw,
                      fc=BOX_LEAD if is_lead else BOX_OTHER, ec=INK3, lw=0.8 * s, z=2)
        elif nm == "bicycle":
            box_patch(ax, x0, y0, max(l, 1.6), max(w, 0.6), yaw,
                      fc=AGENT_COL["bicycle"], ec="#7e7b72", lw=0.5 * s, z=3)
        elif nm == "pedestrian":
            ax.plot(-y0, x0, "o", ms=4.5 * s, color=INK3, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.8 * s)
        elif nm in ("traffic_cone", "czone_sign", "barrier"):
            ax.plot(-y0, x0, "^", ms=4 * s, color="#c9a227", zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.6 * s)
        elif nm == "generic_object" and rich:
            ax.plot(-y0, x0, "s", ms=2.2 * s, color=AGENT_COL["generic_object"],
                    zorder=2, markeredgecolor="#c0bdb3", markeredgewidth=0.3 * s)
    # 自车
    box_patch(ax, 0, 0, EGO_L, EGO_W, 0.0, fc="#ffffff", ec=INK, lw=1.4 * s, z=4)
    # 轨迹（含起点 0,0）。best 与 GT 几乎重合 —— 这正是要展示的，
    # 所以 GT 画成更粗的浅色底带，best 叠在上面，读作“best 贴合专家”。
    p_gt = np.vstack([[0, 0], gt])
    ax.plot(-p_gt[:, 1], p_gt[:, 0], color=C_GT, lw=6.0 * s, alpha=0.45, zorder=5,
            solid_capstyle="round")
    # 车身扫掠框(每隔一个 pose)：DAC 判的是车身不是路点，画出来才看得出切角。
    # 注意 PDMS 实际是对 LQR 闭环仿真后的位姿判定，这里只是示意，不等同于评分几何。
    if footprints:
        for traj_full, c in ((np.array(fdata["models"]["best"]["trajectory"]), C_BEST),
                             (np.array(fdata["models"]["worse"]["trajectory"]), C_WORSE)):
            for x_, y_, h_ in traj_full[2::3]:
                box_patch(ax, x_, y_, EGO_L, EGO_W, h_, fc="none", ec=c,
                          lw=0.7 * s, z=5, alpha=0.5)
    zb, zw = (7, 6) if top == "best" else (6, 7)
    for traj, c, zo in ((tb, C_BEST, zb), (tw, C_WORSE, zw)):
        p = np.vstack([[0, 0], traj])
        ax.plot(-p[:, 1], p[:, 0], color=c, lw=2.0 * s, zorder=zo,
                solid_capstyle="round")
        ax.plot(-p[1:, 1], p[1:, 0], linestyle="none", marker="o", ms=4.5 * s,
                color=c, zorder=zo, markeredgecolor=SURFACE, markeredgewidth=0.9 * s)
    if labels:
        # 三条端点常常挨得很近(best≈GT)，按纵向位置排序后错开标注，避免叠字
        ends = [(-tb[-1, 1], tb[-1, 0], "best ckpt", INK2, "bold"),
                (-tw[-1, 1], tw[-1, 0], "worse ckpt", INK2, "bold"),
                (-p_gt[-1, 1], p_gt[-1, 0], "expert (GT)", "#0f7d57", "normal")]
        ends.sort(key=lambda e: e[1])
        for k, (hx, vy, lab, lab_c, fw) in enumerate(ends):
            ax.annotate(lab, (hx, vy), textcoords="offset points",
                        xytext=(10 * s, (k - 1) * 15 * s), fontsize=9.5 * s, color=lab_c,
                        zorder=9, fontweight=fw,
                        bbox=dict(boxstyle="square,pad=0.12", fc=SURFACE,
                                  ec="none", alpha=0.75))
    # 前车尾部
    if lead_i is not None:
        lx, ly, _, ll, lw_, _, _ = boxes[lead_i][:7]
        rear = lx - ll / 2
        if labels:
            ax.axhline(rear, color=INK3, lw=0.8 * s, ls=(0, (2, 3)), zorder=1)
            ax.text(xlim[1] - 0.4, rear - 1.5, f"lead vehicle rear  {rear:.1f} m",
                    fontsize=8.5 * s, color=INK3, zorder=9, ha="right",
                    bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    # 碰撞标记只认 PDMS 的判定(NC=0)，不要自己按静态框几何推断:
    # PDMS 是闭环仿真，前车也在动，静态几何会在“只超 TTC、未碰撞”的帧上误报。
    if "collision" in fdata["models"]["worse"]["flags"]:
        ax.plot(-tw[-1, 1], tw[-1, 0], "X", ms=13 * s, color=C_WORSE,
                markeredgecolor=SURFACE, markeredgewidth=1.4 * s, zorder=10)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if axis_label:
        ax.set_xlabel("lateral (m)", fontsize=9 * s, color=INK2)
        if ylabel:
            ax.set_ylabel("longitudinal (m)", fontsize=9 * s, color=INK2)
    ax.grid(True, color="#e9e8e2", lw=0.7 * s, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#d9d8d2")
    if ticks:
        ax.tick_params(colors=INK3, labelsize=8.5 * s)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return ax


def main(case=None, from_json=None, out_dir=None):
    """case=<名字> 从 eval 产物现场取数；from_json=<路径> 则只认 JSON 里的数值。"""
    if from_json:
        data = json.load(open(from_json))
        case = data.get("case", os.path.basename(from_json).split(".")[0])
    else:
        data = build_case_data(case)
    cfg = data
    LOG = data["log"]
    WORK = f"/data/autovla_data/eval/nuplan/case_study/{data['tag']}"
    OUT = out_dir or os.path.join(WORK, "fig"); os.makedirs(OUT, exist_ok=True)
    FRAMES = [(f["token"], f["frame_idx"], f["time_label"]) for f in data["frames"]]
    FR = {f["token"]: f for f in data["frames"]}

    log_frames = pickle.load(open(f"/data/autovla_data/nuplan/navsim_logs/test/{LOG}.pkl", "rb"))

    # BEV 范围:未指定则按三帧所有轨迹自动取，等比留边
    xlim, ylim = cfg["xlim"], cfg["ylim"]
    if xlim is None or ylim is None:
        pts = []
        for f in data["frames"]:
            pts.append(np.array(f["gt_trajectory"])[:, :2])
            for m in ("best", "worse"):
                pts.append(np.array(f["models"][m]["trajectory"])[:, :2])
        p = np.vstack(pts + [np.zeros((1, 2))])
        hmin, hmax = (-p[:, 1]).min() - 7, (-p[:, 1]).max() + 7
        vmin, vmax = p[:, 0].min() - 7, p[:, 0].max() + 7
        # 补齐到同一尺度，保证 aspect=equal 时三列一致
        span = max(hmax - hmin, vmax - vmin)
        hc, vc = (hmin + hmax) / 2, (vmin + vmax) / 2
        xlim = (hc - span / 2, hc + span / 2)
        ylim = (vc - span / 2, vc + span / 2)

    # 头部(标题+副标题+图例)先算好占多高，再定网格上沿，避免图例压住列标题。
    import textwrap
    sub_lines = [ln for sub in (cfg["sub1"], cfg["sub2"]) for ln in textwrap.wrap(sub, 190)]
    head_bottom = 0.955 - len(sub_lines) * 0.019      # 副标题最后一行的下沿
    gs_top = head_bottom - 0.062                       # 让出图例 + 列标题

    fig = plt.figure(figsize=(16.5, 11.6), facecolor=SURFACE)
    # 相机条按图像真实宽高比定行高(否则 imshow 等比缩放后下方留大片空白)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.46, 2.25, 1.30],
                          hspace=0.13, wspace=0.10,
                          left=0.035, right=0.985, top=gs_top, bottom=0.025)

    for col, (tok, fi, tlabel) in enumerate(FRAMES):
        fdata = FR[tok]
        anns = log_frames[fi]["anns"]                 # 周围目标:真值标注，不从 JSON 取
        gt = np.array(fdata["gt_trajectory"])[:, :2]
        tb = np.array(fdata["models"]["best"]["trajectory"])[:, :2]
        tw = np.array(fdata["models"]["worse"]["trajectory"])[:, :2]

        # ---------- Row A: 3 路相机（当前帧）----------
        axc = fig.add_subplot(gs[0, col])
        imgs = [Image.open(fdata["cameras"][k]) for k in
                ("front_left", "front", "front_right")]
        h = min(im.height for im in imgs)
        imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
        pad = 6
        W = sum(im.width for im in imgs) + pad * 2
        strip = Image.new("RGB", (W, h), (252, 252, 251))
        x = 0
        for im in imgs:
            strip.paste(im, (x, 0)); x += im.width + pad
        axc.imshow(strip); axc.axis("off"); axc.set_anchor("N")
        axc.set_title(f"{tlabel}    frame {fi}", fontsize=13, color=INK,
                      fontweight="bold", pad=6)
        for frac, name in zip((0.165, 0.5, 0.835), ("front_left", "front", "front_right")):
            axc.text(frac, -0.03, name, transform=axc.transAxes, ha="center",
                     va="top", fontsize=8.5, color=INK3)

        # ---------- Row B: BEV ----------
        ax = fig.add_subplot(gs[1, col])
        draw_bev(ax, LOG, fdata, anns, xlim, ylim, s=1.0,
                 ticks=True, labels=True, axis_label=True, ylabel=(col == 0))

        # ---------- Row C: reasoning + 分数 ----------
        axt = fig.add_subplot(gs[2, col]); axt.axis("off")
        y = 1.0
        for m, c in (("best", C_BEST), ("worse", C_WORSE)):
            md = fdata["models"][m]
            title, body, plan = md["label"], md["think"], md["plan"]
            pdms, flags = md["pdms"], md["flags"]
            tag = ("   " + " · ".join(flags)) if flags else ""
            axt.add_patch(FancyBboxPatch((0.0, y - 0.455), 1.0, 0.42,
                                         boxstyle="round,pad=0.008,rounding_size=0.02",
                                         transform=axt.transAxes, facecolor="#f4f3ee",
                                         edgecolor=c, linewidth=1.6, zorder=0))
            axt.text(0.022, y - 0.055, title, transform=axt.transAxes, fontsize=9.2,
                     color=c, fontweight="bold", va="top")
            axt.text(0.978, y - 0.055, f"PDMS {pdms:.0f}{tag}", transform=axt.transAxes,
                     fontsize=9.2, color=INK if not flags else C_WORSE,
                     fontweight="bold", va="top", ha="right")
            axt.text(0.022, y - 0.145, wrap(body, 62), transform=axt.transAxes,
                     fontsize=9.2, color=INK, va="top", linespacing=1.45)
            axt.text(0.022, y - 0.375, plan, transform=axt.transAxes, fontsize=10.5,
                     color=c, va="top", family="monospace", fontweight="bold")
            y -= 0.5

    # ---------- 标题 & 图例 ----------
    fig.text(0.035, 0.985, cfg["title"],
             fontsize=19, color=INK, fontweight="bold", va="top")
    # 副标题已在上面折行(超长单行会被 bbox_inches="tight" 横向撑开画布)
    yy = 0.955
    for ln in sub_lines:
        fig.text(0.035, yy, ln, fontsize=10.5, color=INK2, va="top")
        yy -= 0.019

    handles = [plt.Line2D([], [], color=C_GT, lw=5.0, alpha=0.45, label="expert (GT)"),
               plt.Line2D([], [], color=C_BEST, lw=2.0, marker="o", ms=5,
                          label="best ckpt · 103k CoT ep4 (navtest PDMS 80.45)"),
               plt.Line2D([], [], color=C_WORSE, lw=2.0, marker="o", ms=5,
                          label="worse ckpt · 166k CoT ep2 (navtest PDMS 74.65)"),
               plt.Line2D([], [], color=C_WORSE, marker="X", ls="", ms=10,
                          markeredgecolor=SURFACE, label="collision"),
               plt.Line2D([], [], color=BOX_LEAD, marker="s", ls="", ms=9,
                          markeredgecolor=INK3, label="lead vehicle"),
               plt.Line2D([], [], color=BOX_OTHER, marker="s", ls="", ms=9,
                          markeredgecolor=INK3, label="other vehicle"),
               plt.Line2D([], [], color=INK3, marker="o", ls="", ms=6, label="pedestrian"),
               plt.Line2D([], [], color="#c9a227", marker="^", ls="", ms=6,
                          label="cone / work-zone sign"),
               plt.Line2D([], [], color="#eceae2", marker="s", ls="", ms=10,
                          markeredgecolor="#cfccc0", label="drivable area (PDMS)")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.033, head_bottom - 0.002),
                     frameon=True, fontsize=9.2, ncol=9, labelcolor=INK2,
                     columnspacing=1.4, handlelength=1.8, borderpad=0.55)
    leg.get_frame().set_facecolor("#f4f3ee"); leg.get_frame().set_edgecolor("#d9d8d2")

    p = os.path.join(OUT, f"case_study_3frames_{case}.png")
    fig.savefig(p, dpi=155, facecolor=SURFACE, bbox_inches="tight")
    print("saved:", p)

    # 文字版摘要
    lines = []
    for tok, fi, tl in FRAMES:
        lines.append(f"\n=== {tl}  frame {fi}  token {tok} ===")
        for m, nm in (("best", "best  (103k CoT ep4)"), ("worse", "worse (166k CoT ep2)")):
            md = FR[tok]["models"][m]
            tr = np.array(md["trajectory"])
            fl = (" [" + ", ".join(md["flags"]) + "]") if md["flags"] else ""
            lines.append(f"  {nm}  PDMS={md['pdms']:5.1f}{fl}  | 5s 位移 {tr[-1, 0]:.2f} m")
            lines.append(f"      think: {md['think']}  {md['plan']}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, f"case_study_3frames_{case}.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=None, choices=list(CASES),
                    help="从 eval 产物现场取数并绘图")
    ap.add_argument("--export-json", metavar="PATH",
                    help="把该 case 的数值(轨迹/reasoning/PDMS)如实导出成 JSON，不绘图")
    ap.add_argument("--from-json", metavar="PATH",
                    help="只用这个 JSON 里的数值绘图(不再读 eval 产物)")
    ap.add_argument("--out-dir", metavar="DIR", help="图片输出目录")
    a = ap.parse_args()
    if a.export_json:
        assert a.case, "--export-json 需要同时给 --case"
        data = build_case_data(a.case)
        os.makedirs(os.path.dirname(os.path.abspath(a.export_json)), exist_ok=True)
        with open(a.export_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        n = sum(len(fr["models"]) for fr in data["frames"])
        print(f"exported: {a.export_json}  ({len(data['frames'])} frames x {n // len(data['frames'])} ckpt)")
    else:
        main(case=a.case or ("leadstop" if not a.from_json else None),
             from_json=a.from_json, out_dir=a.out_dir)
