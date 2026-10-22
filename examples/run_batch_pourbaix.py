"""Run a configured batch of Pourbaix diagram calculations."""

import sys
from pathlib import Path

from generate_batch_pourbaix import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main_wrapper():
    """Generate collage-ready direct-label plots at baseline conditions."""
    common_args = [
        "generate_batch_pourbaix.py",
        "--metals", "Cu",# "Au", "Ni", "Pt", "Pd", "Ti",
        "--activities", "1e-4",
        "--temperatures", "298.15",
        "--ligand-sets", "baseline",
        "--ph-range", "-2", "16",
        "--potential-range", "-2", "3",
        "--box-ph-range", "10.5", "13.5",
        "--box-potential-range", "-2", "2.3",
        "--formats", "png", "pdf", "tif",
        "--dpi", "600",
        "--grid-size", "400",
    ]
    for label_style in ("legend",):#("direct",):
        sys.argv = common_args + [
            "--label-style", label_style,
            "--output-root", str(
                PROJECT_ROOT / "examples" / "figures" / "figures"
                / f"batch_pourbaix_baseline_{label_style}"
            ),
        ]
        main()


if __name__ == "__main__":
    main_wrapper()
