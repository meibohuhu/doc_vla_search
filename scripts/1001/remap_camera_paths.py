"""把 per-frame JSON 里的相机绝对路径换个根（别的机器上生成的 JSON 拿到 tigris 上用）。

只改 key 以 `camera_paths` 结尾的列表，其他字段原样保留；原地写回。
幂等：已经是新前缀的路径不再动；既不是旧前缀也不是新前缀的路径会报出来。

    python scripts/1001/remap_camera_paths.py \
        --src /scratch/mh2803/vla/nuplan \
        --dst /shared/rc/llm-gen-agent/mhu/auto_vla/nuplan \
        /shared/rc/llm-gen-agent/mhu/auto_vla/nuplan/navtest_nocot ...
"""
import argparse
import json
import os

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="旧前缀，如 /data/autovla_data/nuplan")
    ap.add_argument("--dst", required=True, help="新前缀，如 /shared/rc/llm-gen-agent/mhu/auto_vla/nuplan")
    ap.add_argument("dirs", nargs="+", help="放 <token>.json 的目录")
    args = ap.parse_args()
    src, dst = args.src.rstrip("/") + "/", args.dst.rstrip("/") + "/"

    for d in args.dirs:
        names = [e.name for e in os.scandir(d) if e.name.endswith(".json")]
        changed = already = 0
        foreign = []
        for n in tqdm(names, desc=os.path.basename(d.rstrip("/")), mininterval=5):
            p = os.path.join(d, n)
            with open(p) as f:
                j = json.load(f)
            dirty = False
            for k, v in j.items():
                if not k.endswith("camera_paths"):
                    continue
                for i, img in enumerate(v):
                    if img.startswith(src):
                        v[i] = dst + img[len(src):]
                        dirty = True
                    elif not img.startswith(dst):
                        foreign.append(img)
            if dirty:
                with open(p, "w") as f:
                    json.dump(j, f, indent=2)
                changed += 1
            else:
                already += 1
        print(f"{d}: {len(names)} json, 改写 {changed}, 已是新前缀 {already}, 未知前缀路径 {len(foreign)}")
        for img in foreign[:5]:
            print(f"    未知前缀: {img}")


if __name__ == "__main__":
    main()
