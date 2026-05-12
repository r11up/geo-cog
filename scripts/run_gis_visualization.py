#!/usr/bin/env python3
"""Generate GIS figures from live Singapore API data."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate GIS visualization figures with real API data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--gee_project", type=str, default=None,
                   help="GEE cloud project ID (e.g. 'my-gee-project')")
    p.add_argument("--openaq_key",  type=str, default=None,
                   help="OpenAQ v3 API key (recommended; may be required)")
    p.add_argument("--year",        type=int, default=2024,
                   help="Calendar year for satellite composites (default 2024)")
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Override output directory")
    p.add_argument("--accuracy",    type=float, default=0.768,
                   help="Accuracy for proxy risk calibration (default 0.768)")
    p.add_argument("--log_level",   type=str, default="INFO",
                   choices=["DEBUG","INFO","WARNING"],
                   help="Logging verbosity (default INFO)")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  [%(levelname)s]  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("gis_viz")

    from config import CFG
    if args.output_dir:
        CFG.paths.output_dir = Path(args.output_dir)
    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  NEURO-GEOSPATIAL  ·  GIS Visualization Suite")
    logger.info("  Year: %d   GEE: enabled   Cache: disabled", args.year)
    logger.info("=" * 65)

    from data.geo_data import SingaporeGridData

    logger.info("Fetching environmental indicators from live APIs …")
    grids, df = SingaporeGridData.build(
        cfg            = CFG,
        use_gee        = True,
        gee_project    = args.gee_project,
        openaq_api_key = args.openaq_key,
        year           = args.year,
        cache          = False,
    )

    logger.info("Grid data assembled:")
    for key, grid in grids.items():
        logger.info("  %-22s min=%7.3f  mean=%7.3f  max=%7.3f",
                    key, grid.min(), grid.mean(), grid.max())
    logger.info("DataFrame: %d rows × %d columns", len(df), len(df.columns))

    from evaluation.gis_visualization import run_all_figures
    run_all_figures(CFG, grids, df, model_accuracy=args.accuracy)

    output_dir = CFG.paths.output_dir / "gis_figures"
    logger.info("=" * 65)
    logger.info("All figures saved to: %s", output_dir)
    logger.info("=" * 65)
    logger.info("")
    logger.info("Output files:")
    outputs = [
        ("figure3_study_area_map.png/.pdf",     "Paper Figure 3 — NDVI / PM₂.₅ / FAR"),
        ("figure_all_indicators.png/.pdf",      "All 6 indicators (6-panel)"),
        ("figure_grid_heatmap.png/.pdf",        "Annotated per-cell values"),
        ("figure_correlation_matrix.png/.pdf",  "Pearson r — all 6 indicators"),
        ("figure_buffer_analysis.png/.pdf",     "100/250/500 m buffer (MAUP)"),
        ("figure_exposure_zones.png/.pdf",      "PECM zone classification map"),
        ("figure_singapore_basemap.png/.pdf",   "Live OSM/satellite basemap"),
        ("figure8_spatial_error_map.png/.pdf",  "Proxy environmental error-risk figure"),
        ("figure8_spatial_error_clean.png/.pdf","Clean proxy error-risk map"),
        ("singapore_grid_indicators.csv",       "100-cell data table"),
    ]
    for fname, desc in outputs:
        p = output_dir / fname.split("/")[0]
        exists = "✓" if p.exists() else "✗"
        logger.info("  %s  %-46s  %s", exists, fname, desc)


if __name__ == "__main__":
    main()
