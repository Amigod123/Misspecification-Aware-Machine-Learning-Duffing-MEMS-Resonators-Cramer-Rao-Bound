"""Deterministic dataset generation with manifests.

    python gen_data.py --config configs/default.json

Writes data/<config>/seed<s>/<dataset>.npz and data/<config>/manifest.json (seeds, ranges, sizes,
software versions, generation time, feature definitions, SHA-256 of every file).
Every sample draws its own parameter vector; train / calibration / test sets come from independent
RNG streams, and held-out Q and N blocks never overlap the training ranges."""
import os, json, time, argparse
import numpy as np
from lib import load_config, generate, save_dataset, dataset_seed, software_versions, features

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.json")
args = ap.parse_args()
cfg = load_config(args.config)
root = os.path.join("data", cfg["name"]); os.makedirs(root, exist_ok=True)
S = cfg["sizes"]


def plan():
    """(dataset name, size, ranges key, world) for every dataset of one seed."""
    out = [("train", S["train"], "train", "nominal"), ("calib", S["calib"], "train", "nominal"),
           ("abstain_calib", S["abstain_calib"], "mismatch", "nominal")]
    for k, v in cfg["test_conditions"].items():
        out.append((f"test_{k}", S["test"], v["ranges"], v["world"]))
    for w in ("nominal", "feed", "quintic", "damp"):
        out.append((f"det_{w}", S["det_train"], "mismatch", w))
    return out


manifest = dict(config=cfg, versions=software_versions(), generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                seed_rule="dataset_seed = 100000*(seed+1) + DATASET_CODES[name] (see lib.py)", datasets={})
t_all = time.perf_counter()
for seed in cfg["seeds"]:
    os.makedirs(os.path.join(root, f"seed{seed}"), exist_ok=True)
    for name, n, rk, world in plan():
        t = time.perf_counter()
        d = generate(n, dataset_seed(seed, name), cfg, rk, world)
        path = os.path.join(root, f"seed{seed}", f"{name}.npz")
        h = save_dataset(d, path)
        manifest["datasets"][f"seed{seed}/{name}"] = dict(path=path, n=n, ranges=cfg["ranges"][rk], ranges_key=rk, world=world,
                                                          rng_seed=dataset_seed(seed, name), sha256=h,
                                                          seconds=round(time.perf_counter() - t, 3))
    print(f"seed {seed}: done")
_, names = features(generate(5, 0, cfg))
manifest["feature_names"] = names
manifest["n_features"] = len(names)
manifest["targets"] = ["log(Q)", "N", "shift_ppm"]
manifest["total_seconds"] = round(time.perf_counter() - t_all, 2)
json.dump(manifest, open(os.path.join(root, "manifest.json"), "w"), indent=1)
print(f"{len(manifest['datasets'])} datasets, {len(names)} features, {manifest['total_seconds']} s")
