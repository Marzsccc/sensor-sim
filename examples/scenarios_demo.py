"""
Batch regression demo (v0.15.0).

Runs the ENTIRE default scenario library through the standard ESKF
pipeline and prints one pass/fail matrix -- the "is my fusion stack
healthy today" overnight view.

    .venv/bin/python examples/scenarios_demo.py [--seed N] [--verbose]

Each scenario pins its own sensor grades, outages and regression gates,
so a future estimator swap keeps every gate fixed and comparable.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensor_sim.scenarios import default_library, run_scenario, batch_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="override every scenario's seed")
    ap.add_argument("--verbose", action="store_true",
                    help="print each full report")
    args = ap.parse_args()

    reports = []
    for sc in default_library():
        rep = run_scenario(sc, seed=args.seed)
        reports.append(rep)
        if args.verbose:
            print(rep.summary(), "\n")

    print("=" * 64)
    print(" Batch scenario regression (v0.15.0)")
    print("=" * 64)
    print(batch_summary(reports))

    failed = [r.name for r in reports if not r.all_gates_passed()]
    if failed:
        print("\nFailed scenarios:")
        for name in failed:
            r = [x for x in reports if x.name == name][0]
            for g in r.gates:
                if not g.passed:
                    print(f"  - {name}: {g.name} {g.detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
