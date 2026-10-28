import json, sys
from multiprocessing import Pool
import lc_render as L
R = [r for r in json.load(open('scan_navtrain.json')) if r['cmd'] == 'keep forward']
P = L.mc_paths()
jobs = [(r, P[r['token']]) for r in R if r['token'] in P]
with Pool(48) as p:
    rows = p.map(L.offsets_one, jobs, chunksize=64)
json.dump(rows, open('kf_offsets.json', 'w'))
print(len(R), 'keep forward,', len(jobs), 'with metric cache')
