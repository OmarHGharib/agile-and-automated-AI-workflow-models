"""Stage 08 -- assemble morphometry + mechanics into CSV / Excel / JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ribpipe.report")

MORPH_COLS = ["label", "side", "rib", "length_mm", "max_width_mm",
              "min_width_mm", "max_height_mm", "min_height_mm",
              "max_depth_mm", "curvature_max", "n_voxels"]


def morphometry_dataframe(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in MORPH_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[MORPH_COLS].sort_values(["side", "rib"]).reset_index(drop=True)
    return df.round(2)


def write_reports(morph_rows, mechanics_rows, out_dir, cfg,
                  case_id: str = "case", qc: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    morph = morphometry_dataframe(morph_rows)
    mech = pd.DataFrame(mechanics_rows)

    if cfg.write_csv:
        p = out_dir / f"{case_id}_rib_morphometry.csv"; morph.to_csv(p, index=False)
        paths["morph_csv"] = str(p)
        p = out_dir / f"{case_id}_mechanics_report.csv"; mech.to_csv(p, index=False)
        paths["mech_csv"] = str(p)

    if cfg.write_excel:
        p = out_dir / f"{case_id}_report.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as xl:
            morph.to_excel(xl, sheet_name="rib_morphometry", index=False)
            mech.to_excel(xl, sheet_name="mechanics", index=False)
            if qc:
                pd.DataFrame([qc]).to_excel(xl, sheet_name="qc", index=False)
        paths["excel"] = str(p)

    if cfg.write_json:
        n_fail = int((mech.get("status") == "FAIL").sum()) if not mech.empty else 0
        n_warn = int((mech.get("status") == "WARN").sum()) if not mech.empty else 0
        summary = {
            "case_id": case_id,
            "n_ribs_measured": int(len(morph)),
            "n_mechanics_checks": int(len(mech)),
            "n_fail": n_fail, "n_warn": n_warn,
            "qc": qc or {},
            "morphometry": morph.to_dict(orient="records"),
            "mechanics": mech.to_dict(orient="records"),
        }
        p = out_dir / f"{case_id}_summary.json"
        p.write_text(json.dumps(summary, indent=2))
        paths["json"] = str(p)

    log.info("Reports written: %s", ", ".join(paths))
    return paths
