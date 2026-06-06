"""Aggregate the per-seed dual_sourcing benchmark_full_suite instance JSONs into a
seed-robust (mean +/- std over optimizer seeds) gap-vs-CDI table per Gijsbrechts Fig-9 row.

OBJECTIVE: turn the single-seed "learned beats CDI on 2 rows by -0.009%/-0.041%" claim into a
mean +/- cross-seed-std verdict. CDI (capped_dual_index) is the strongest Gijsbrechts heuristic
and the ~0% optimality proxy, so we report the learned policy's relative gap vs the BEST
heuristic (= CDI on every row) as mean +/- std over the >=5 optimizer seeds, and the count of
seeds that strictly beat CDI.

USAGE: python scripts/dual_sourcing/aggregate_seed_robust_cdi.py --tags ds_seedrobust_s9001 ...
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SPEC = "soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets"
ROWS = ["dual_l2_ce105", "dual_l2_ce110", "dual_l3_ce105", "dual_l3_ce110",
        "dual_l4_ce105", "dual_l4_ce110"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--spec", default=SPEC)
    parsed = p.parse_args()

    per_row = {r: [] for r in ROWS}
    for tag in parsed.tags:
        root = PACKAGE_ROOT / "outputs" / "benchmarks" / tag / "instances"
        for r in ROWS:
            f = root / f"{r}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            cs = d["comparative_summary"]
            bh = cs["best_heuristic_cost"]
            bhn = cs["best_heuristic_name"]
            cdi = d["heuristics"].get("capped_dual_index", {}).get("mean_cost")
            g = cs["policy_gaps"].get(parsed.spec)
            if g is None:
                continue
            learned = d["learned_policies"][parsed.spec]["evaluation"]["learned_policy"]["mean_cost"]
            relgap_bh = g["relative_gap_pct_vs_best_heuristic"]
            relgap_cdi = 100.0 * (learned - cdi) / cdi if cdi else None
            per_row[r].append({"tag": tag, "learned": learned, "best_heur": bh,
                               "best_heur_name": bhn, "cdi": cdi,
                               "relgap_vs_best_heur": relgap_bh, "relgap_vs_cdi": relgap_cdi})

    print(f"{'row':14s} {'N':>2s} {'learned mean':>13s} {'CDI':>9s} "
          f"{'gap%vsCDI mean':>15s} {'std':>7s} {'beats':>7s}  verdict")
    out = {}
    for r in ROWS:
        rows = per_row[r]
        if not rows:
            print(f"{r:14s}  0  (no data)")
            continue
        gaps = [x["relgap_vs_cdi"] for x in rows if x["relgap_vs_cdi"] is not None]
        learned = [x["learned"] for x in rows]
        cdi = rows[0]["cdi"]
        n = len(gaps)
        m = statistics.mean(gaps); s = statistics.stdev(gaps) if n > 1 else 0.0
        beats = sum(1 for g in gaps if g < 0)
        if m < -s and beats == n and s >= 0:
            verdict = "ROBUST_BEAT_CDI"
        elif abs(m) <= max(s, 1e-9):
            verdict = "PARITY"
        elif m > 0:
            verdict = "ROBUST_ABOVE_CDI"
        else:
            verdict = "BEAT_WITHIN_STD"
        print(f"{r:14s} {n:>2d} {statistics.mean(learned):>13.4f} {cdi:>9.4f} "
              f"{m:>+15.4f} {s:>7.4f} {beats:>5d}/{n}  {verdict}")
        out[r] = {"n": n, "learned_mean": statistics.mean(learned),
                  "learned_std": statistics.stdev(learned) if n > 1 else 0.0,
                  "cdi": cdi, "gap_vs_cdi_mean_pct": m, "gap_vs_cdi_std_pct": s,
                  "seeds_beating_cdi": f"{beats}/{n}", "verdict": verdict,
                  "per_seed": rows}
    outpath = PACKAGE_ROOT / "outputs" / "benchmarks" / "ds_seed_robust_cdi_summary.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"WROTE {outpath}")


if __name__ == "__main__":
    main()
