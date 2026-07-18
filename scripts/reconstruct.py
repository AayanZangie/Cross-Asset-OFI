"""Batch reconstruct raw OKX L2 .data files into cached .npz books."""

import argparse
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ofi import load_l2_delta  # noqa: E402

NAME_RE = re.compile(r"([A-Za-z]+)-USDT-L2orderbook-\d+lv-(\d{4}-\d{2}-\d{2})\.data$")


def _one(task):
    f, out, levels = task
    if out.exists():
        return (f.name, "skip", 0, 0.0)
    t0 = time.time()
    sym = load_l2_delta(str(f), levels=levels)
    np.savez_compressed(out, **sym)
    return (f.name, "done", len(sym["ts"]), time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/historical")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--levels", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args()

    raw, out = Path(a.raw_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep_assets = {x.lower() for x in a.assets} if a.assets else None
    tasks = []
    for f in sorted(raw.glob("*.data")):
        m = NAME_RE.search(f.name)
        if not m:
            continue
        asset, date = m.group(1).lower(), m.group(2)
        if keep_assets and asset not in keep_assets:
            continue
        if a.start and date < a.start:
            continue
        if a.end and date > a.end:
            continue
        tasks.append((f, out / f"{asset}_{date}.npz", a.levels))

    if not tasks:
        print(f"No matching .data files in {raw}/")
        return

    cached = sum(1 for t in tasks if t[1].exists())
    print(f"{len(tasks)} files match | {cached} cached | {len(tasks) - cached} to do | jobs={a.jobs}")
    t0 = time.time()
    n = len(tasks)

    def log(i, r):
        name, status, ns, dt = r
        if status == "done":
            print(f"  [{i}/{n}] {name}: {ns} snaps ({dt:.0f}s)")

    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
                log(i, r)
    else:
        for i, task in enumerate(tasks, 1):
            log(i, _one(task))
    print(f"Finished in {(time.time() - t0) / 60:.1f} min. Cache -> {out}/")


if __name__ == "__main__":
    main()
