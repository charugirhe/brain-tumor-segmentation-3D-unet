# test_langgraph_workflow.py
# Tests for the LangGraph workflow.
# All tests use mocked agents/DB/model -- no real MRI files, DB, or Gemini key needed.

import sys
import unittest
import unittest.mock as mock
from typing import Any, Dict

# --- Patch heavy modules before importing anything that touches them ----------
_MOCK_CONFIG = mock.MagicMock()
_MOCK_CONFIG.BRATS_DATASET_PATH = "./dataset"
_MOCK_CONFIG.MODEL_PATH = "./models/best_model.pth"

_PATCH_MODULES = {
    "model_inference":  mock.MagicMock(),
    "database":         mock.MagicMock(),
    "config":           _MOCK_CONFIG,
    "report_generator": mock.MagicMock(),
    "torch":            mock.MagicMock(),
    "monai":            mock.MagicMock(),
    "reportlab":        mock.MagicMock(),
    "google":           mock.MagicMock(),
}

for mod, m in _PATCH_MODULES.items():
    sys.modules.setdefault(mod, m)

# --- Minimal agent stubs -----------------------------------------------------
_GOOD_SCAN = {
    "tumor_detected":    True,
    "total_volume":      25.0,
    "ncr_volume":        8.0,
    "ed_volume":         12.0,
    "et_volume":         5.0,
    "ncr_percent":       32.0,
    "ed_percent":        48.0,
    "et_percent":        20.0,
    "flair_image":       "b64flair",
    "predicted_image":   "b64pred",
    "overlay_image":     "b64overlay",
    "gt_image":          "",
    "gradcam_image":     "b64cam",
    "confidence_image":  "b64conf",
    "modality_attribution": {"FLAIR": 30.0, "T1": 25.0, "T1CE": 35.0, "T2": 10.0},
    "top_modality":      "T1CE",
    "class_attentions":  {},
}

_LARGE_SCAN = {
    **_GOOD_SCAN,
    "total_volume": 45.0,
    "et_volume":    12.0,
}

_RETURNING_PREVIOUS = {
    "scan_date":      "2026-01-01",
    "tumor_detected": True,
    "total_volume":   20.0,
    "ncr_volume":     6.0,
    "ed_volume":      10.0,
    "et_volume":      4.0,
    "ncr_percent":    30.0,
    "ed_percent":     50.0,
    "et_percent":     20.0,
}

_GROWING_PREVIOUS = {
    **_RETURNING_PREVIOUS,
    "total_volume": 10.0,  # current 25 cm3 -> 150% growth -> must flag review
}


def _make_reception(is_returning: bool, previous_scan=None):
    def _agent(patient_id, name, age, gender):
        return {
            "patient_id":    patient_id,
            "name":          name,
            "age":           age,
            "gender":        gender,
            "is_returning":  is_returning,
            "previous_scan": previous_scan,
        }
    return _agent


def _make_comparison():
    """Stub that mirrors comparison_agent logic for predictable output."""
    def _agent(current_scan, previous_scan):
        cur = current_scan["total_volume"]
        prev = previous_scan["total_volume"]
        abs_change = round(cur - prev, 2)
        pct = round((abs_change / prev) * 100, 1) if prev else 0.0
        trend = "GREW" if abs_change > 0 else ("SHRANK" if abs_change < 0 else "STABLE")
        return {
            "previous_date":     previous_scan.get("scan_date", ""),
            "prev_total_volume": prev,
            "prev_ncr_volume":   previous_scan["ncr_volume"],
            "prev_ed_volume":    previous_scan["ed_volume"],
            "prev_et_volume":    previous_scan["et_volume"],
            "total_change":  {"absolute": abs_change, "percent": pct, "trend": trend},
            "ncr_change":    {"absolute": 0, "percent": 0, "trend": "STABLE"},
            "ed_change":     {"absolute": 0, "percent": 0, "trend": "STABLE"},
            "et_change":     {"absolute": 0, "percent": 0, "trend": "STABLE"},
        }
    return _agent


# --- Base state helpers -------------------------------------------------------
_VALID_NII_BYTES = b"FAKE_NII_CONTENT"


def _upload_state(patient_id="PT001", name="Test Patient", age=45, gender="Male"):
    return {
        "patient_id":  patient_id,
        "name":        name,
        "age":         age,
        "gender":      gender,
        "use_dataset": False,
        "flair_bytes": _VALID_NII_BYTES,
        "t1_bytes":    _VALID_NII_BYTES,
        "t1ce_bytes":  _VALID_NII_BYTES,
        "t2_bytes":    _VALID_NII_BYTES,
        "flair_name":  "flair.nii",
        "t1_name":     "t1.nii",
        "t1ce_name":   "t1ce.nii",
        "t2_name":     "t2.nii",
        "has_seg":     False,
    }


# --- Test Cases ---------------------------------------------------------------
class TestWorkflowRouting(unittest.TestCase):

    def _run(self, state, agents_mock):
        """Patch agents module and run the workflow."""
        with mock.patch.dict("sys.modules", {"agents": agents_mock}):
            import importlib
            import langgraph_workflow as wf
            importlib.reload(wf)
            return wf.run_workflow(state)

    # 1. Validation failure -- missing modality
    def test_validation_fails_missing_modality(self):
        """State with missing t2_bytes must fail at scan_validation."""
        state = _upload_state()
        state["t2_bytes"] = None

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)

        result = self._run(state, agents_mock)

        self.assertIn("Missing modality: T2", result["validation_errors"])
        self.assertEqual(result["workflow_status"], "validation_failed")
        agents_mock.scan_analysis_agent.assert_not_called()
        agents_mock.dataset_scan_agent.assert_not_called()

    # 2. Validation failure -- bad file extension
    def test_validation_fails_bad_extension(self):
        """Uploading a .jpg instead of .nii must fail validation."""
        state = _upload_state()
        state["flair_name"] = "flair.jpg"

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)

        result = self._run(state, agents_mock)

        errors = result["validation_errors"]
        self.assertTrue(any("FLAIR" in e and ".nii" in e for e in errors), errors)
        self.assertEqual(result["workflow_status"], "validation_failed")

    # 3. Happy path -- new patient, no review flags
    def test_happy_path_new_patient(self):
        """Valid upload, new patient, small tumor -> complete, no review."""
        state = _upload_state()

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)
        agents_mock.scan_analysis_agent.return_value = _GOOD_SCAN
        agents_mock.report_generator_agent.return_value = "AI report text."

        result = self._run(state, agents_mock)

        self.assertEqual(result["workflow_status"], "complete")
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["scan_result"]["total_volume"], 25.0)
        self.assertIn("AI report text.", result["gemini_report"])
        self.assertNotIn("RADIOLOGIST REVIEW REQUIRED", result["gemini_report"])  # noqa: E501
        agents_mock.scan_analysis_agent.assert_called_once()

    # 4. Large tumor volume completes cleanly without arbitrary review triggers
    def test_large_tumor_volume_completes_cleanly(self):
        """Large tumor volume must complete without false threshold review flags."""
        state = _upload_state()

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)
        agents_mock.scan_analysis_agent.return_value = _LARGE_SCAN
        agents_mock.report_generator_agent.return_value = "AI report text."

        result = self._run(state, agents_mock)

        self.assertEqual(result["workflow_status"], "complete")
        self.assertFalse(result["needs_review"])
        self.assertEqual(len(result["review_reasons"]), 0)
        self.assertNotIn("RADIOLOGIST REVIEW REQUIRED", result["gemini_report"])
        self.assertIn("research purposes only", result["gemini_report"])

    # 5. Returning patient -- significant growth compared cleanly
    def test_returning_patient_growth_comparison(self):
        """150% growth (10->25 cm3) executes longitudinal comparison without threshold alert."""
        state = _upload_state(patient_id="PT002")

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(
            True, _GROWING_PREVIOUS
        )
        agents_mock.scan_analysis_agent.return_value = _GOOD_SCAN  # 25 cm3
        agents_mock.comparison_agent.side_effect = _make_comparison()
        agents_mock.report_generator_agent.return_value = "AI report text."

        result = self._run(state, agents_mock)

        self.assertEqual(result["workflow_status"], "complete")
        self.assertFalse(result["needs_review"])
        comp = result["comparison_result"]
        self.assertIsNotNone(comp)
        self.assertEqual(comp["total_change"]["trend"], "GREW")
        self.assertEqual(comp["total_change"]["percent"], 150.0)
        self.assertNotIn("RADIOLOGIST REVIEW REQUIRED", result["gemini_report"])

    # 6. Returning patient -- comparison runs
    def test_returning_patient_comparison(self):
        """Returning patient comparison runs and calculates changes cleanly."""
        state = _upload_state(patient_id="PT003")

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(
            True, _RETURNING_PREVIOUS  # 20 cm3 -> 25 cm3 = 25% growth
        )
        agents_mock.scan_analysis_agent.return_value = _GOOD_SCAN
        agents_mock.comparison_agent.side_effect = _make_comparison()
        agents_mock.report_generator_agent.return_value = "Report."

        result = self._run(state, agents_mock)

        self.assertEqual(result["workflow_status"], "complete")
        self.assertIsNotNone(result.get("comparison_result"))
        self.assertFalse(result["needs_review"])
        self.assertNotIn("RADIOLOGIST REVIEW REQUIRED", result["gemini_report"])

    # 7. Dataset mode -- validation passes without byte fields
    def test_dataset_mode_validation_passes(self):
        """Dataset mode should skip upload-field checks."""
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_name = os.path.basename(tmpdir)
            _MOCK_CONFIG.BRATS_DATASET_PATH = os.path.dirname(tmpdir)

            state = {
                "patient_id":     "PT004",
                "name":           "Dataset Patient",
                "age":            50,
                "gender":         "Female",
                "use_dataset":    True,
                "dataset_folder": folder_name,
            }

            agents_mock = mock.MagicMock()
            agents_mock.reception_agent.side_effect = _make_reception(False)
            agents_mock.dataset_scan_agent.return_value = _GOOD_SCAN
            agents_mock.report_generator_agent.return_value = "Dataset report."

            result = self._run(state, agents_mock)

            self.assertEqual(result["validation_errors"], [])
            self.assertEqual(result["workflow_status"], "complete")
            agents_mock.dataset_scan_agent.assert_called_once_with(folder_name)

        _MOCK_CONFIG.BRATS_DATASET_PATH = "./dataset"

    # 8. Segmentation failure -> workflow ends gracefully
    def test_segmentation_failure_ends_gracefully(self):
        """If inference raises, scan_result=None, workflow_status=validation_failed."""
        state = _upload_state()

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)
        agents_mock.scan_analysis_agent.side_effect = RuntimeError("GPU OOM")

        result = self._run(state, agents_mock)

        self.assertIsNone(result.get("scan_result"))
        self.assertEqual(result["workflow_status"], "validation_failed")
        self.assertIn("GPU OOM", result.get("error_message", ""))
        agents_mock.report_generator_agent.assert_not_called()

    # 9. .nii.gz extensions accepted
    def test_nii_gz_extension_accepted(self):
        """.nii.gz files must pass validation."""
        state = _upload_state()
        state["flair_name"] = "flair.nii.gz"
        state["t1_name"]    = "t1.nii.gz"
        state["t1ce_name"]  = "t1ce.nii.gz"
        state["t2_name"]    = "t2.nii.gz"

        agents_mock = mock.MagicMock()
        agents_mock.reception_agent.side_effect = _make_reception(False)
        agents_mock.scan_analysis_agent.return_value = _GOOD_SCAN
        agents_mock.report_generator_agent.return_value = "Report."

        result = self._run(state, agents_mock)

        self.assertEqual(result["validation_errors"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
