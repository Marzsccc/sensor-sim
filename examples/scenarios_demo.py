"""
Batch regression demo (v0.15.1).

Runs the ENTIRE default scenario library through the standard ESKF
pipeline:
  - stable scenarios -> one run + per-run gates
  - stochastic corners (parking-garage etc.) -> Monte-Carlo over seeds,
    gates on the across-seed p90 distribution

Prints one pass/fail matrix -- the "is my fusion stack healthy today"
overnight view.

    .venv/bin/python examples/scenarios_demo.py [--seed N]
        [--mc-seeds 10] [--verbose]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensor_sim.scenarios import (default_library, run_scenario,
                                  run_scenario_mc, batch_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="override every scenario's default seed")
    ap.add_argument("--mc-seeds", type=int, default=10,
                    help="number of seeds for Monte-Carlo-gated scenarios")
    ap.add_argument("--verbose", action="store_true",
                    help="print each full report / MC summary")
    args = ap.parse_args()

    rows = []           # (display_report, mc_gate_or_None)
    for sc in default_library():
        if sc.mc_gates:
            mc = run_scenario_mc(sc, seeds=range(42, 42 + args.mc_seeds))
            # display row uses a representative single run (default seed)
            disp = run_scenario(sc) if args.seed is None else \
                run_scenario(sc, seed=args.seed)
            rows.append((disp, mc))
            if args.verbose:
                print(mc.summary(), "\n")
        else:
            rep = run_scenario(sc, seed=args.seed)
            rows.append((rep, None))
            if args.verbose:
                print(rep.summary(), "\n")

    reports = [r for r, _ in rows]

    print("=" * 64)
    print(" Batch scenario regression (v0.15.1)")
    print("=" * 64)
    print(batch_summary(reports, display_rows=rows))

    failed = []
    for rep, mc in rows:
        if mc is not None:
            v = mc.verdict()
            if not v["passed"]:
                failed.append((mc.name, "Monte-Carlo",
                               [g for g in v["gates"] if not g["passed"]]))
        elif not rep.all_gates_passed():
            failed.append((rep.name, "single-run",
                           [g for g in rep.gates if not g.passed]))

    if failed:
        print("\nFailed scenarios:")
        for name, kind, gates in failed:
            print(f"  - {name} ({kind}):")
            for g in gates:
                detail = g["detail"] if isinstance(g, dict) else g.detail
                gname = g["gate"] if isinstance(g, dict) else g.name
                print(f"      {gname}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
