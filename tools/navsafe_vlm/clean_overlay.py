"""去掉 NavSafe 可视化前视图（scenarios/<token>/cam/cam_f0_*.jpg）上叠加的规划走廊、轨迹点和文字。

NavSafe 的 vis_utils.py:1148 用 cv2.addWeighted 叠了半透明走廊：out = 0.55 * (0, 200, 200)[BGR] + 0.45 * orig，
可以反算 orig = (out - 0.55 * c) / 0.45。走廊的区域按颜色找：从图片底部中间（走廊的锚点）开始的连通区域，
满足 R、G 都明显高于 B（叠加后 R、G 至少 110，B 至多 115）。走廊边线（0.7 倍颜色、不透明）、轨迹点、
角上的文字框反算不了，用 cv2.inpaint 补。JPEG 噪声在走廊里会被放大约 2.2 倍，看起来有点糊，不影响判断。

    python tools/navsafe_vlm/clean_overlay.py in.jpg out.jpg
"""
import sys

import cv2
import numpy as np

RIBBON_BGR = np.array([0, 200, 200], np.float32)
ALPHA = 0.55


def ribbon_mask(img: np.ndarray) -> np.ndarray:
    b, g, r = [img[..., k].astype(np.int16) for k in range(3)]
    cand = ((r - b) > 45) & ((g - b) > 45) & (np.abs(r - g) < 40) & (r >= 100) & (g >= 100)
    cand = cand.astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    H, W = cand.shape
    seeds = set(lab[H - 6:H, W // 2 - 60:W // 2 + 60].ravel().tolist()) - {0}
    mask = np.isin(lab, list(seeds)).astype(np.uint8)
    # 填掉里面的小洞（被轨迹点、车道线打断的地方）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def clean(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    mask = ribbon_mask(img)
    out = img.astype(np.float32)
    m = mask.astype(bool)
    out[m] = (out[m] - ALPHA * RIBBON_BGR) / (1 - ALPHA)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 要 inpaint 的：走廊边缘一圈（边线）、轨迹点 / 时间标签（绿、红的饱和小点）、左上和左下的文字框
    fix = cv2.dilate(mask, np.ones((7, 7), np.uint8)) - cv2.erode(mask, np.ones((7, 7), np.uint8))
    # 轨迹点 / 标签在反算后的图上找：叠在走廊里的绿点、红点反算后会变成很饱和的颜色，路面是灰的
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    dots = ((hsv[..., 1] > 110) & mask.astype(bool)).astype(np.uint8)
    fix |= cv2.dilate(dots, np.ones((5, 5), np.uint8))
    fix[:int(0.035 * H), :int(0.30 * W)] = 1                        # 左上 "Model | Frame k  v km/h"
    fix[int(0.955 * H):, :int(0.10 * W)] = 1                        # 左下 "Cmd: …"
    return cv2.inpaint(out, fix, 5, cv2.INPAINT_TELEA)


if __name__ == "__main__":
    cv2.imwrite(sys.argv[2], clean(cv2.imread(sys.argv[1])))
