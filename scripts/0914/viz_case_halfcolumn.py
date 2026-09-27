#!/usr/bin/env python
"""
半栏(3.5in)定性图，版式对齐 paper_draft/ICRA/media/qualitative_b2d.pdf：

  (a) SFT baseline   [t] [t+0.5s] [t+1s]   前视图 + 投影的预测轨迹 + 速度徽标
                     [reasoning 引文 x3]
                     一行斜体结论
  (b) Ours           同上

数值全部来自 case JSON(scripts/0914/viz_case_study.py --export-json 导出)。
轨迹投影用 navsim log 里的相机内外参(cam_intrinsic + sensor2lidar_*)，属于标定真值，不在 JSON 里。

  python scripts/0914/viz_case_halfcolumn.py \
      --from-json logs/0914_nvidia/case_json/leadstop.json \
      --out paper_draft/ICRA/media/qualitative_navsim.pdf
"""
import os, json, pickle, textwrap, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

C_OURS, C_BASE, C_GT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a84"
SURFACE = "#ffffff"
TINT = {"worse": "#fdf0ec", "best": "#eef3fc"}      # 块底色
LOGS = "/data/autovla_data/nuplan/navsim_logs/test"
EGO_BACK = 1.127          # 后轴到车尾(m)


def _mix(hexc, frac):
    """把颜色按 frac 兑白(frac=0 原色, 1 纯白)。"""
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(v + (255 - v) * frac) for v in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)



def quat2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def projector(log, frame_idx, cam="CAM_F0"):
    """返回把自车系 (x前,y左,z上) 点列投到该帧前视图像素的函数。"""
    fr = pickle.load(open(f"{LOGS}/{log}.pkl", "rb"))[frame_idx]
    c = fr["cams"][cam]
    K = np.array(c["cam_intrinsic"])
    Rsl, tsl = np.array(c["sensor2lidar_rotation"]), np.array(c["sensor2lidar_translation"])
    l2e_r, l2e_t = np.array(fr["lidar2ego_rotation"]), np.array(fr["lidar2ego_translation"])
    Rle = quat2R(l2e_r) if l2e_r.shape == (4,) else l2e_r

    def f(pts_xy, z=0.0):
        P = np.array([[p[0], p[1], z] for p in pts_xy], dtype=float)
        Pl = (Rle.T @ (P - l2e_t).T).T                 # ego -> lidar
        Pc = (Rsl.T @ (Pl - tsl).T).T                  # lidar -> camera
        ok = Pc[:, 2] > 0.5                            # 只取相机前方的点
        uv = np.full((len(P), 2), np.nan)
        if ok.any():
            q = (K @ Pc[ok].T).T
            uv[ok] = q[:, :2] / q[:, 2:3]
        return uv
    return f


def quote(txt, width=21, max_lines=5):
    """按 b2d 的做法:截断加省略号，控制在 max_lines 行内。"""
    lines = textwrap.wrap(txt, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1].rstrip(" ,.") + "…"
    return "\n".join(lines)


def layout_bev(a, data):
    """
    NAVSIM 半栏版式(开环:两个模型看同一批帧)：
      row1  每帧一条相机带 —— front 放大、front_left/front_right 缩小
      row2  每帧一个 BEV(与大图同一套 draw_bev 画法，窗口按轨迹范围收紧)
      row3  每帧 baseline 的 reasoning 框
      row4  每帧 ours 的 reasoning 框
    尺寸按英寸排版，图高自适应。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import viz_case_study as vcs
    from viz_case_study import draw_bev
    # 主色(红/蓝)统一从参数走；draw_bev 画 BEV 轨迹用的是 vcs 里的常量，一并覆盖，
    # 否则 BEV 里的线和下面文字框会是两种深浅。
    c_ours, c_base = a.color_ours, a.color_base
    vcs.C_BEST, vcs.C_WORSE = c_ours, c_base
    tint = {"worse": _mix(c_base, a.tint_mix), "best": _mix(c_ours, a.tint_mix)}

    log, frames = data["log"], data["frames"]
    W = a.width
    MARG, GAPX = 0.10, 0.045
    pw = (W - 2 * MARG - 2 * GAPX) / 3

    # ---- 相机带:front 大、两侧小，合成一张 ----
    def cam_strip(f):
        ims = [Image.open(f["cameras"][k]) for k in ("front_left", "front", "front_right")]
        Hf = 360
        Hs = int(Hf * a.side_scale)
        # 两侧横向裁掉外缘(保留靠近 front 的一侧)，让 front 在条子里占更大宽度 ->
        # 列宽固定时，这是唯一能把 front 画大的办法(单纯抬高坐标轴只会多出留白)。
        if a.front_crop < 1.0:            # front 居中裁切 = 真正的放大(视野换尺寸)
            fim = ims[1]; fw = int(fim.width * a.front_crop)
            x0 = (fim.width - fw) // 2
            ims[1] = fim.crop((x0, 0, x0 + fw, fim.height))
        cr = a.side_crop
        if cr < 1.0:
            l_, r_ = ims[0], ims[2]
            wl = int(l_.width * cr); wr = int(r_.width * cr)
            ims[0] = l_.crop((l_.width - wl, 0, l_.width, l_.height))   # 左视图保留右半
            ims[2] = r_.crop((0, 0, wr, r_.height))                     # 右视图保留左半
        out = []
        for im, h in zip(ims, (Hs, Hf, Hs)):
            out.append(im.resize((max(1, int(im.width * h / im.height)), h)))
        pad = 4
        Wc = sum(i.width for i in out) + pad * 2
        strip = Image.new("RGB", (Wc, Hf), (255, 255, 255))
        ox = 0
        for im in out:
            strip.paste(im, (ox, (Hf - im.height) // 2))
            ox += im.width + pad
        return strip

    strips = [cam_strip(f) for f in frames]
    cam_h = pw * strips[0].height / strips[0].width   # 精确贴合，多余的 zoom 只会变留白

    # ---- BEV 窗口:按三帧所有轨迹收紧，再补到面板长宽比 ----
    hs, vs = [], []
    for f in frames:
        arrs = [np.array(f["gt_trajectory"])[:, :2]]
        arrs += [np.array(f["models"][m]["trajectory"])[:, :2] for m in ("best", "worse")]
        for arr in arrs:
            hs += list(-arr[:, 1])
            vs += list(arr[:, 0])
    hs.append(0.0)
    vs += [-EGO_BACK, 0.0]
    m = a.bev_margin
    hmin, hmax = min(hs) - m, max(hs) + m
    vmin, vmax = min(vs) - m, max(vs) + m
    hspan, vspan = hmax - hmin, vmax - vmin
    if vspan / hspan < a.bev_aspect:
        need = hspan * a.bev_aspect
        c = (vmin + vmax) / 2
        vmin, vmax = c - need / 2, c + need / 2
    else:
        need = vspan / a.bev_aspect
        c = (hmin + hmax) / 2
        hmin, hmax = c - need / 2, c + need / 2
    xlim, ylim = (hmin, hmax), (vmin, vmax)
    bev_h = pw * a.bev_aspect

    # 框高按【实际最长引文的行数】算，避免文字出框(写死高度换个例子就溢出)
    nmax = max(len(quote(f["models"][mk]["think"], a.box_wrap, a.box_lines).split("\n"))
               for f in frames for mk in ("best", "worse"))
    line_h = a.box_fs * 1.22 / 72.0                 # pt -> in
    box_h = max(a.box_h, 0.145 + nmax * line_h + 0.055)
    if a.show_plan:
        box_h += 0.075
    gapy = 0.045
    leg_h = 0.0 if a.legend == "none" else 0.135
    tk_h = 0.115 * sum(1 for k in ("worse", "best")
                       if (data.get("halfcol_takeaway") or {}).get(k))
    H = (MARG + 0.135 + cam_h + gapy + bev_h + leg_h
         + gapy + 2 * box_h + tk_h + gapy + MARG)
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fx = lambda v: v / W
    fy = lambda v: v / H

    # 外框:贴着画布留一点边，灰色实线/虚线
    if a.frame != "none":
        pad = 0.030
        fig.patches.append(Rectangle(
            (fx(pad), fy(pad)), fx(W - 2 * pad), fy(H - 2 * pad),
            transform=fig.transFigure, facecolor="none", edgecolor=a.frame_color,
            linewidth=a.frame_lw, linestyle="--" if a.frame == "dashed" else "-",
            zorder=20, joinstyle="round"))

    y = H - MARG - 0.135
    # ---- row 1: 相机 ----
    for k, f in enumerate(frames):
        x = MARG + k * (pw + GAPX)
        ax = fig.add_axes([fx(x), fy(y - cam_h), fx(pw), fy(cam_h)])
        ax.imshow(strips[k])
        ax.axis("off")
        tl = f["time_label"].replace("t = +", "t+").replace("t = ", "t")
        tl = tl.replace("0.0 s", "").strip() or "t"
        badge = dict(boxstyle="round,pad=0.30,rounding_size=0.35",
                     facecolor=a.badge_fc, edgecolor=a.badge_ec,
                     linewidth=0.0 if a.badge_ec == "none" else 0.5)
        ax.text(0.0, 1.10, tl, transform=ax.transAxes, fontsize=5.4, color=a.badge_tc,
                va="bottom", ha="left", fontweight="bold", bbox=badge)
        ax.text(1.0, 1.10, "%.1f m/s" % f["speed_mps"], transform=ax.transAxes,
                fontsize=5.4, color=a.badge_tc, va="bottom", ha="right",
                fontweight="bold", bbox=badge)
    y -= cam_h + gapy

    # ---- row 2: BEV ----
    log_frames = pickle.load(open("%s/%s.pkl" % (LOGS, log), "rb"))
    for k, f in enumerate(frames):
        x = MARG + k * (pw + GAPX)
        ax = fig.add_axes([fx(x), fy(y - bev_h), fx(pw), fy(bev_h)])
        draw_bev(ax, log, f, log_frames[f["frame_idx"]]["anns"], xlim, ylim,
                 s=a.bev_scale, ticks=False, labels=False, axis_label=False,
                 footprints=a.footprints, top="worse",
                 rich=not a.plain_map, radius=a.map_radius)
    y -= bev_h

    # ---- 图例:绿=专家(GT)、蓝=Ours、红/橙=SFT baseline ----
    if a.legend != "none":
        axl = fig.add_axes([fx(MARG), fy(y - leg_h), fx(W - 2 * MARG), fy(leg_h)])
        axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis("off"); axl.patch.set_alpha(0)
        entries = [(C_GT, a.gt_label, "band"),
                   (c_ours, "Ours", "line"),
                   (c_base, "SFT baseline", "line")]
        # 按文字长度粗略分配水平位置，整体居中
        widths = [0.012 * len(t) + 0.075 for _, t, _ in entries]
        total = sum(widths) + 0.06 * (len(entries) - 1)
        x0 = (1.0 - total) / 2
        for (col, txt, kind), wgt in zip(entries, widths):
            xs = [x0, x0 + 0.048]
            if kind == "band":
                axl.plot(xs, [0.5, 0.5], color=col, lw=3.2, alpha=0.55,
                         solid_capstyle="round", clip_on=False)
            else:
                axl.plot(xs, [0.5, 0.5], color=col, lw=1.3, solid_capstyle="round",
                         clip_on=False)
                axl.plot([sum(xs) / 2], [0.5], marker="o", ms=2.4, color=col,
                         markeredgecolor="white", markeredgewidth=0.4, clip_on=False)
            axl.text(x0 + 0.058, 0.5, txt, fontsize=5.2, color=INK,
                     va="center", ha="left")
            x0 += wgt + 0.06
    y -= leg_h + gapy

    # ---- row 3/4: 每帧两个 reasoning 框(baseline / ours) ----
    short = {"worse": "Baseline", "best": "Ours"}
    for mk, c in (("worse", c_base), ("best", c_ours)):
        for k, f in enumerate(frames):
            x = MARG + k * (pw + GAPX)
            md = f["models"][mk]
            fig.patches.append(FancyBboxPatch(
                (fx(x), fy(y - box_h)), fx(pw), fy(box_h),
                boxstyle="round,pad=0.002,rounding_size=0.010",
                transform=fig.transFigure, facecolor=(a.box_fill if a.box_fill != "auto" else tint[mk]), edgecolor=c,
                lw=0.7, zorder=0))
            # 有违规标记时标题会变长，退用短名，避免和右侧数字撞在一起
            lab = md.get("figure_label", short[mk])
            if a.show_pdms and md["flags"]:     # 有分数+违规标记时标题退用短名，免得撞字
                lab = short[mk]
            fig.text(fx(x + 0.045), fy(y - 0.075), lab, fontsize=5.2, color=c,
                     fontweight="bold", va="center", zorder=3)
            flag = md["flags"][0] if md["flags"] else ""
            if a.show_pdms:                     # 右上角 PDMS 分数，默认不显示
                tail = ("%.0f" % md["pdms"]) + (("  " + flag) if flag else "")
                fig.text(fx(x + pw - 0.045), fy(y - 0.075), tail,
                         fontsize=5.2 if not flag else 4.7, color=c,
                         fontweight="bold", va="center", ha="right", zorder=3)
            fig.text(fx(x + 0.045), fy(y - 0.145),
                     "\u201c" + quote(md["think"], a.box_wrap, a.box_lines) + "\u201d",
                     fontsize=a.box_fs, color=INK, va="top", style="italic",
                     linespacing=1.2, zorder=3)
            if a.show_plan:                       # meta-action 默认不画
                fig.text(fx(x + 0.045), fy(y - box_h + 0.038), md["plan"],
                         fontsize=a.box_fs - 0.4, color=c, family="monospace",
                         va="bottom", zorder=3)
        # 该模型整体表现的一句话总结(JSON 里的 halfcol_takeaway)
        tk = (data.get("halfcol_takeaway") or {}).get(mk)
        if tk:
            fig.text(fx(W / 2), fy(y - box_h - 0.075), tk, fontsize=5.4, color=c,
                     style="italic", fontweight="bold", va="center", ha="center",
                     zorder=3)
            y -= 0.115
        y -= box_h + gapy

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE, dpi=400)
    print("saved: %s   (%.2f x %.2f in)" % (a.out, W, H))
    if a.png:
        q = a.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(q, facecolor=SURFACE, dpi=400)
        print("saved:", q)


def main(a):
    data = json.load(open(a.from_json))
    if a.layout == "bev":
        return layout_bev(a, data)
    log = data["log"]
    frames = data["frames"]
    order = [("worse", C_BASE), ("best", C_OURS)]      # baseline 在上、ours 在下，与 b2d 一致
    takeaway = data.get("halfcol_takeaway", {})

    fig = plt.figure(figsize=(a.width, a.height), facecolor=SURFACE)

    L, R = 0.012, 0.988
    block_h = 0.468                                    # 每块占的高度(figure 坐标)
    gap_top = 0.012
    for bi, (m, col) in enumerate(order):
        y1 = 1.0 - gap_top - bi * (block_h + 0.028)    # 块顶
        y0 = y1 - block_h                              # 块底
        # 块底板
        fig.patches.append(FancyBboxPatch(
            (L, y0), R - L, block_h, boxstyle="round,pad=0.004,rounding_size=0.016",
            transform=fig.transFigure, facecolor=TINT[m], edgecolor=col,
            linewidth=0.9, zorder=0))
        # 块标签
        lab = frames[0]["models"][m].get("figure_label", m)
        fig.text(L + 0.018, y1 - 0.018, f"({'ab'[bi]}) {lab}", fontsize=7.2,
                 color=col, fontweight="bold", va="top", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=col, lw=0.8))

        # --- 三张前视图 ---
        iw = (R - L - 0.036 - 2 * 0.012) / 3           # 单图宽度
        img_y = y1 - 0.075
        img_h = 0.150
        for k, fr in enumerate(frames):
            x = L + 0.018 + k * (iw + 0.012)
            ax = fig.add_axes([x, img_y - img_h, iw, img_h], zorder=3)
            im = Image.open(fr["cameras"]["front"])
            W, H = im.size
            ax.imshow(im)
            # 预测轨迹(投影)
            proj = projector(log, fr["frame_idx"])
            traj = np.array(fr["models"][m]["trajectory"])[:, :2]
            uv = proj(np.vstack([[0, 0], traj]))
            v = uv[~np.isnan(uv[:, 0])]
            if len(v) > 1:
                ax.plot(v[:, 0], v[:, 1], "-", color=col, lw=1.5, zorder=4,
                        solid_capstyle="round",
                        path_effects=None)
                ax.plot(v[:, 0], v[:, 1], "o", color=col, ms=2.0, zorder=5,
                        markeredgecolor="white", markeredgewidth=0.35)
            if a.show_expert:
                g = proj(np.vstack([[0, 0], np.array(fr["gt_trajectory"])[:, :2]]))
                g = g[~np.isnan(g[:, 0])]
                if len(g) > 1:
                    ax.plot(g[:, 0], g[:, 1], "--", color=C_GT, lw=1.0, zorder=3)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
            # 时间 & 速度徽标
            tl = fr["time_label"].replace("t = +", "t+").replace("t = ", "t")
            tl = tl.replace("0.0 s", "").strip() or "t"
            ax.text(0.03, 0.06, tl, transform=ax.transAxes, fontsize=5.0, color=INK,
                    va="top", ha="left", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.9))
            ax.text(0.97, 0.06, f"{fr['speed_mps']:.1f} m/s", transform=ax.transAxes,
                    fontsize=5.0, color=col, fontweight="bold", va="top", ha="right",
                    zorder=6,
                    bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none", alpha=0.9))
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(0.8)
            ax.set_frame_on(True)

        # --- 三个 reasoning 引文框 ---
        q_y = img_y - img_h - 0.016
        q_h = 0.185
        for k, fr in enumerate(frames):
            x = L + 0.018 + k * (iw + 0.012)
            md = fr["models"][m]
            fig.patches.append(FancyBboxPatch(
                (x, q_y - q_h), iw, q_h, boxstyle="round,pad=0.003,rounding_size=0.012",
                transform=fig.transFigure, facecolor="white", edgecolor=col,
                linewidth=0.7, zorder=3))
            fig.text(x + iw / 2, q_y - 0.012, "“" + quote(md["think"], a.wrap) + "”",
                     fontsize=a.quote_fs, color=INK, va="top", ha="center",
                     style="italic", linespacing=1.28, zorder=4)
            fig.text(x + iw / 2, q_y - q_h + 0.030, md["plan"], fontsize=a.quote_fs - 0.3,
                     color=col, va="bottom", ha="center", family="monospace",
                     fontweight="bold", zorder=4)

        # --- 块尾结论 ---
        tk = takeaway.get(m)
        if tk:
            fig.text((L + R) / 2, y0 + 0.012, tk, fontsize=6.0, color=col,
                     style="italic", fontweight="bold", va="bottom", ha="center", zorder=4)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE, dpi=400)
    print("saved:", a.out)
    if a.png:
        fig.savefig(a.out.rsplit(".", 1)[0] + ".png", facecolor=SURFACE, dpi=400)
        print("saved:", a.out.rsplit(".", 1)[0] + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=float, default=3.5)    # ieeeconf 单栏 = 3.5in
    ap.add_argument("--height", type=float, default=3.55)
    ap.add_argument("--wrap", type=int, default=21)
    ap.add_argument("--quote-fs", type=float, default=5.2)
    ap.add_argument("--show-expert", action="store_true", help="叠一条虚线专家轨迹作参照")
    ap.add_argument("--png", action="store_true", help="同时存一份 png 方便预览")
    ap.add_argument("--bev-scale", type=float, default=0.42, help="BEV 内元素缩放")
    ap.add_argument("--plain-map", action="store_true", help="用回 roadblock 级的粗底图")
    ap.add_argument("--map-radius", type=float, default=60.0, help="底图取多大范围(m)")
    ap.add_argument("--bev-margin", type=float, default=2.5, help="BEV 窗口在轨迹外留的边距(m)")
    ap.add_argument("--side-scale", type=float, default=0.62, help="左右视图相对 front 的高度比")
    ap.add_argument("--box-fill", default="auto",
                    help="reasoning 框的填充色;auto=各自主色兑白")
    ap.add_argument("--gt-label", default="GT", help="绿线在图例里的名字")
    ap.add_argument("--legend", choices=["show", "none"], default="show",
                    help="是否画 GT/Ours/baseline 的颜色图例")
    ap.add_argument("--color-ours", default="#4a8ade", help="Ours 的主色(蓝)")
    ap.add_argument("--color-base", default="#ef7c4a", help="baseline 的主色(红/橙)")
    ap.add_argument("--tint-mix", type=float, default=0.94,
                    help="文字框底色 = 主色兑白的比例(越大越浅)")
    ap.add_argument("--badge-fc", default="#eaeef3", help="时间/速度小框底色")
    ap.add_argument("--badge-ec", default="#aebac7", help="小框边色(none=无边)")
    ap.add_argument("--badge-tc", default="#20303f", help="小框文字色")
    ap.add_argument("--frame-color", default="#c3b58e", help="外框颜色")
    ap.add_argument("--frame-lw", type=float, default=0.8, help="外框线宽")
    ap.add_argument("--frame", choices=["solid", "dashed", "none"], default="solid",
                    help="整图外框样式")
    ap.add_argument("--show-pdms", action="store_true", help="框右上角显示该帧 PDMS")
    ap.add_argument("--show-plan", action="store_true", help="在框里显示 meta-action(如 STOP,STRAIGHT)")
    ap.add_argument("--box-h", type=float, default=0.50, help="每个 reasoning 框高度(in)")
    ap.add_argument("--box-wrap", type=int, default=23, help="框内换行宽度(字符)")
    ap.add_argument("--box-lines", type=int, default=6, help="框内最多行数")
    ap.add_argument("--box-fs", type=float, default=4.6, help="框内字号")
    ap.add_argument("--footprints", action="store_true", help="BEV 里画车身扫掠框(半栏偏挤，默认关)")
    ap.add_argument("--front-crop", type=float, default=1.0,
                    help="front 视图保留的横向比例(<1 即居中裁切放大)")
    ap.add_argument("--side-crop", type=float, default=0.40,
                    help="左右视图保留的横向比例(越小 front 越大)")
    ap.add_argument("--bev-wrap", type=int, default=74, help="bev 版式文字行换行宽度(字符)")
    ap.add_argument("--bev-aspect", type=float, default=0.92, help="BEV 面板 高/宽")
    ap.add_argument("--bev-lat", type=float, default=14.0, help="BEV 横向视野(m)")
    ap.add_argument("--quote-frame", type=int, default=-1, help="文字行引用第几帧的 reasoning")
    ap.add_argument("--layout", choices=["camera", "bev"], default="bev",
                    help="camera=b2d 那种两块相机版式(需自车在动);bev=NAVSIM 开环用的版式")
    main(ap.parse_args())
