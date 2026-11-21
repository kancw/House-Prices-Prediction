"""命令列入口。

  python run.py --list                    列出所有版本
  python run.py iter1_baseline            跑 iter1：CV + 產生 submissions/sub_iter1_baseline.csv + 更新紀錄表
  python run.py iter1_baseline --cv-only  只看 CV 分數，不產生提交檔
  python run.py --all                     依序跑所有版本（細分版本只跑 CV）
  python run.py iter3_features --ablate   ablation：逐一拿掉 iter3 的特徵步驟
"""
import argparse

from src.pipeline import ablation, run_version
from src.versions import VERSIONS


def main():
    p = argparse.ArgumentParser(description="House Prices 實驗執行器")
    p.add_argument("version", nargs="?", help="版本名稱，例如 iter1_baseline")
    p.add_argument("--list", action="store_true", help="列出所有版本")
    p.add_argument("--all", action="store_true", help="依序跑所有版本")
    p.add_argument("--cv-only", action="store_true", help="只跑 CV，不寫提交檔")
    p.add_argument("--ablate", action="store_true", help="對指定版本做 ablation")
    args = p.parse_args()

    if args.list or not (args.version or args.all):
        for v, cfg in VERSIONS.items():
            tag = "  CV" if cfg["cv_only"] else "提交"
            print(f"[{tag}] {v:<22} {cfg['desc']}")
        return
    if args.all:
        for v in VERSIONS:
            run_version(v, submit=not args.cv_only)
        return
    if args.version not in VERSIONS:
        p.error(f"沒有版本 {args.version}，用 --list 看可用版本")
    if args.ablate:
        ablation(args.version)
    else:
        run_version(args.version, submit=not args.cv_only)


if __name__ == "__main__":
    main()
