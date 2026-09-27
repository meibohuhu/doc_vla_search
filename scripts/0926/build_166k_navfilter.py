"""按 navtest 口径过滤 166k，建混训用的净新增集（全 symlink，不动原始实体）。

分析与动机见 docs/0901/166k_vs_103k_distribution.md §6。
所有判据都用 GT 轨迹运动学 + instruction 字段，不用 CoT 标签（CoT 的 STOP ≠ 真停）。

  漏斗:166k 全量 → 去 navtrain 重合 → 去 navtrain_val 泄漏
       F1 去真停(5s 位移 < 1m)  F2 去 unknown 导航  F3 去 CV-easy(|disp5 - v0*5| < 1m)
  产出:
    trainval_166k_navfilt         = F1+F2+F3                  （不配平）
    trainval_166k_navfilt_bal     = 上面 + 直行下采样，使 navtrain+它 的转弯指令占比 = navtest (33.6%)
    trainval_166k_navfilt_turn40  = 同上，但转弯目标上调到 40%（转弯场景 PDMS 最低，见 doc §6.2）

  链接目标：trainval_cot_166k(_val)/<token>.json —— 带 cot_output，CoT / no-CoT 训练都能用
  （no-CoT 忽略 cot_output；与 trainval_mix166k_add 的做法一致）。

  PY=/data/autovla_data/envs/autovla/bin/python
  $PY scripts/0926/build_166k_navfilter.py [--dry-run]                                  # navfilt + navfilt_bal
  $PY scripts/0926/build_166k_navfilter.py --no-full --turn-target 0.40 \\
      --bal-name trainval_166k_navfilt_turn40                                           # 转弯 40% 版
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

R = '/data/autovla_data/nuplan'
TURN = ('turn left', 'turn right')


def feat(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    g = np.array(d['gt_trajectory'], float)
    v0 = float(np.hypot(*np.asarray(d['velocity'], float)[:2]))
    disp = float(np.hypot(*g[-1, :2]))
    return dict(token=d['token'], v0=v0, disp=disp, instr=d.get('instruction'), cv_err=abs(disp - v0 * 5.0))


def load(sub):
    fs = [f'{R}/{sub}/{f}' for f in os.listdir(f'{R}/{sub}') if f.endswith('.json')]
    with Pool(48) as p:
        return pd.DataFrame([r for r in p.imap_unordered(feat, fs, chunksize=256) if r])


def tokens(sub):
    return {f[:-5] for f in os.listdir(f'{R}/{sub}') if f.endswith('.json')}


def link(df, name, dry):
    out = f'{R}/{name}'
    if dry:
        print(f'  [dry-run] {out}: {len(df)}')
        return
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，先挪走再建')
    os.makedirs(out)
    for t in df.token:
        src = next(f'{R}/{d}/{t}.json' for d in ('trainval_cot_166k', 'trainval_cot_166k_val')
                   if os.path.exists(f'{R}/{d}/{t}.json'))
        os.symlink(src, f'{out}/{t}.json')
    df.token.to_csv(f'{R}/{name}_tokens.txt', index=False, header=False)
    print(f'  {out}: {len(df)} symlinks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--turn-target', type=float, default=None, help='混合后转弯指令占比；不给 = navtest 的占比')
    ap.add_argument('--bal-name', default='trainval_166k_navfilt_bal', help='配平集目录名')
    ap.add_argument('--no-full', action='store_true', help='不建不配平的 trainval_166k_navfilt')
    args = ap.parse_args()

    k, nav, test = load('trainval_nocot_166k'), load('navtrain_nocot'), load('navtest_nocot')
    only = k[~k.token.isin(tokens('navtrain_nocot')) & ~k.token.isin(tokens('navtrain_nocot_val'))]
    f1 = only[only.disp >= 1.0]
    f2 = f1[f1.instr != 'unknown']
    f3 = f2[f2.cv_err >= 1.0]
    for n, d in [('166k', k), ('去 navtrain 重合/val 泄漏', only), ('F1 去真停', f1), ('F2 去 unknown', f2), ('F3 去 CV-easy', f3)]:
        print(f'  {n:24s} {len(d):7d}')

    # 转弯配平：F3 的转弯全留，直行下采样到 (nav_turn + T) / (nav + T + S) = 目标转弯占比
    tgt = args.turn_target if args.turn_target is not None else test.instr.isin(TURN).mean()
    t_add, s_add = f3[f3.instr.isin(TURN)], f3[~f3.instr.isin(TURN)]
    n_s = int((nav.instr.isin(TURN).sum() + len(t_add)) / tgt - len(nav) - len(t_add))
    bal = pd.concat([t_add, s_add.sample(max(0, min(n_s, len(s_add))), random_state=0)])
    print(f'  配平(转弯目标 {tgt * 100:.1f}%)        {len(bal):7d}  (turn {len(t_add)} + straight {len(bal) - len(t_add)})')

    if not args.no_full:
        link(f3, 'trainval_166k_navfilt', args.dry_run)
    link(bal, args.bal_name, args.dry_run)


if __name__ == '__main__':
    main()
