"""Command-line entry point.

  python run.py --list                    list all versions
  python run.py iter1_baseline            run iter1: CV + submissions/sub_iter1_baseline.csv + update log
  python run.py iter1_baseline --cv-only  CV score only, no submission file
  python run.py --all                     run every version (breakdown versions are CV only)
  python run.py iter3_features --ablate   ablation: remove iter3 feature steps one at a time
"""
import argparse

from src.pipeline import ablation, run_version
from src.versions import VERSIONS


def main():
    p = argparse.ArgumentParser(description="House Prices experiment runner")
    p.add_argument("version", nargs="?", help="version name, e.g. iter1_baseline")
    p.add_argument("--list", action="store_true", help="list all versions")
    p.add_argument("--all", action="store_true", help="run every version in order")
    p.add_argument("--cv-only", action="store_true", help="CV only, do not write a submission file")
    p.add_argument("--ablate", action="store_true", help="run an ablation for the given version")
    args = p.parse_args()

    if args.list or not (args.version or args.all):
        for v, cfg in VERSIONS.items():
            tag = "CV    " if cfg["cv_only"] else "SUBMIT"
            print(f"[{tag}] {v:<22} {cfg['desc']}")
        return
    if args.all:
        for v in VERSIONS:
            run_version(v, submit=not args.cv_only)
        return
    if args.version not in VERSIONS:
        p.error(f"No version named {args.version}; use --list to see available versions")
    if args.ablate:
        ablation(args.version)
    else:
        run_version(args.version, submit=not args.cv_only)


if __name__ == "__main__":
    main()
