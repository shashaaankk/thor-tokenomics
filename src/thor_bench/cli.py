"""thor-bench command line: probe, run, analyze."""

import argparse
import json
import sys

from thor_bench.analyze import analyzeRun, printSummary
from thor_bench.bench import runBench
from thor_bench.config import loadConfig
from thor_bench.probe import probe
from thor_bench.workloads import loadWorkload


def main() -> int:
    ap = argparse.ArgumentParser(prog="thor-bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pProbe = sub.add_parser("probe", help="pre-measurement server checks")
    pProbe.add_argument("-c", "--config", required=True)
    pProbe.add_argument("--osl", type=int, default=64)

    pRun = sub.add_parser("run", help="run one measurement cell")
    pRun.add_argument("-c", "--config", required=True)
    pRun.add_argument("-w", "--workload", required=True)

    pAn = sub.add_parser("analyze", help="summarize a run directory")
    pAn.add_argument("runDir")

    args = ap.parse_args()

    if args.cmd == "probe":
        cfg = loadConfig(args.config)
        result = probe(cfg.server, osl=args.osl, timeoutS=cfg.run.timeoutS)
        print(json.dumps(result, indent=2))
        return 0 if result["pass"] else 1

    if args.cmd == "run":
        cfg = loadConfig(args.config)
        wl = loadWorkload(args.workload)
        runDir = runBench(cfg, wl)
        printSummary(analyzeRun(runDir))
        return 0

    if args.cmd == "analyze":
        printSummary(analyzeRun(args.runDir))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
