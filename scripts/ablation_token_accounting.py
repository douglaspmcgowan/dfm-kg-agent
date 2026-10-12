"""Aggregate per-configuration token and latency totals for the v2 ablation.

Sources (repository-relative by default):
  benchmark_ablation_v2_AB.json      configs A_llm_only and B_llm_rag
  benchmark_ablation_v2_C_only.json  config C_full_pipeline

Every number is a sum or mean over the stored records. The stored records
carry one completion_tokens count per query, which is reasoning plus visible
output combined. They carry no visible-versus-reasoning split and no per-call
overhead, so those components are listed as unavailable rather than estimated.

Usage
  python scripts/ablation_token_accounting.py
  python scripts/ablation_token_accounting.py --output ablation_token_accounting.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    REPO_ROOT / "benchmark_ablation_v2_AB.json",
    REPO_ROOT / "benchmark_ablation_v2_C_only.json",
]
DEFAULT_OUTPUT = REPO_ROOT / "ablation_token_accounting.json"
UNAVAILABLE = [
    "visible_completion_tokens",
    "reasoning_completion_tokens",
    "per_call_overhead_tokens",
]


def summarize_config(records):
    def column(key):
        return [r[key] for r in records if isinstance(r.get(key), (int, float))]

    completion = column("completion_tokens")
    prompt = column("prompt_tokens")
    elapsed = column("elapsed_s")
    return {
        "queries": len(records),
        "prompt_tokens_total": sum(prompt),
        "prompt_tokens_mean": sum(prompt) / len(prompt) if prompt else None,
        "completion_tokens_total": sum(completion),
        "completion_tokens_mean": sum(completion) / len(completion) if completion else None,
        "completion_tokens_per_query": completion,
        "elapsed_s_mean": sum(elapsed) / len(elapsed) if elapsed else None,
        "refusals": sum(1 for r in records if r.get("refusal_flag")),
        "unavailable_components": UNAVAILABLE,
    }


def load_configs(paths):
    by_config = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["records"]:
            by_config.setdefault(r["config"], []).append(r)
    return by_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, nargs="+", default=SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Aggregate JSON for the paper builder; pass '-' to skip writing.")
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.sources if not p.is_file()]
    if missing:
        print(f"error: missing source file(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    by_config = load_configs(args.sources)
    result = {
        "sources": [p.name for p in args.sources],
        "configs": {name: summarize_config(rs) for name, rs in sorted(by_config.items())},
    }

    print(f"{'config':<20}{'n':>4}{'prompt_mean':>14}{'compl_mean':>14}{'compl_total':>14}{'elapsed_mean':>14}")
    for name, s in result["configs"].items():
        print(f"{name:<20}{s['queries']:>4}{s['prompt_tokens_mean']:>14.3f}"
              f"{s['completion_tokens_mean']:>14.3f}{s['completion_tokens_total']:>14}"
              f"{s['elapsed_s_mean']:>14.3f}")
    print(f"\nUnavailable in the source records: {', '.join(UNAVAILABLE)}")

    if str(args.output) != "-":
        out = args.output.resolve()
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
