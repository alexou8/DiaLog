"""Integration test for scripts/export_findings.py: the report a human
reviewer reads to decide whether to port a method to the production
TypeScript analytics engine.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_export_findings_module():
    spec = importlib.util.spec_from_file_location("export_findings", SCRIPTS_DIR / "export_findings.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_findings"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
class TestExportFindings:
    def test_infers_synthetic_provenance_from_filename(self):
        module = _load_export_findings_module()
        provenance = module.infer_data_provenance(Path("data/sample_glucose_data.csv"), override=None)
        assert "SYNTHETIC" in provenance

    def test_explicit_provenance_overrides_heuristic(self):
        module = _load_export_findings_module()
        provenance = module.infer_data_provenance(Path("data/real_export.csv"), override="Real clinic export, IRB #123")
        assert provenance == "Real clinic export, IRB #123"

    def test_end_to_end_writes_valid_findings_json(self, tmp_path, synthetic_event_rows):
        module = _load_export_findings_module()

        events_csv = tmp_path / "sample_events.csv"
        raw = pd.DataFrame(synthetic_event_rows, columns=module.EVENT_COLUMNS)
        raw.to_csv(events_csv, index=False)

        output_path = tmp_path / "findings.json"
        args = [
            "export_findings.py",
            "--events-csv",
            str(events_csv),
            "--output",
            str(output_path),
            "--n-estimators",
            "20",
            "--cv-splits",
            "3",
        ]
        old_argv = sys.argv
        sys.argv = args
        try:
            module.main()
        finally:
            sys.argv = old_argv

        assert output_path.exists()
        with open(output_path) as f:
            findings = json.load(f)

        assert "SYNTHETIC" in findings["data_provenance"]
        assert findings["split_config"]["method"] == "chronological"
        for baseline in ("majority_class", "prevalence", "previous_reading"):
            assert baseline in findings["baseline_metrics_on_test"]
            assert baseline in findings["lift_over_baselines"]
        assert set(findings["sample_sizes"].keys()) == {"train", "val", "test"}
        assert "feature_importances" in findings
        assert len(findings["feature_importances"]) > 0
