import argparse, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026,2027,2028])
    args = ap.parse_args()
    for seed in args.seeds:
        cmd = [sys.executable, "scripts/train.py", "--config", args.config, "--seed", str(seed)]
        print(" ".join(cmd))
        subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
