"""Run the full pipeline: generate synthetic data -> clean & validate -> analyse -> charts -> report.

Usage:
    python run_pipeline.py              # everything
    python run_pipeline.py --skip-generate   # reuse the existing raw CSV
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nps_analysis import analysis, clean, generate_data, report  # noqa: E402
from nps_analysis import config as C  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-generate", action="store_true", help="reuse data/raw instead of regenerating it")
    args = parser.parse_args()

    if not args.skip_generate:
        print("1/4 Generating synthetic survey data")
        generate_data.main()
    print("2/4 Cleaning and validating")
    clean.main()
    print("3/4 Running analysis, tables and charts")
    findings = analysis.run()
    print("4/4 Writing report")
    md, pdf = report.write_reports(findings)
    o = findings["overall"]
    print(f"\nOverall NPS {o.nps:+.1f} (95% CI {o.ci_low:+.1f} to {o.ci_high:+.1f}), n={o.n:,}")
    print(f"Tables:  {C.TABLES.relative_to(C.ROOT)}/  Figures: {C.FIGURES.relative_to(C.ROOT)}/")
    print(f"Report:  {md.relative_to(C.ROOT)} and {pdf.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
