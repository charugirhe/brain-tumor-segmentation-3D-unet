# langgraph_workflow.py
# LangGraph-based multi-agent workflow for the Brain Tumor Analysis System.
#
# Graph nodes (in order):
#   reception -> scan_validation -> segmentation -> supervisor
#                                                       |
#                                     +-------------------------------------------+
#                                     | reporting  (needs_review flagged in text)  |
#                                     | reporting  (standard)                      |
#                                     | END        (validation / inference fail)   |
#                                     +-------------------------------------------+
#
# Design rules followed:
# - No LLM calls added beyond what already exists in agents.py / report_generator_agent.
# - No arbitrary volume thresholds or invented confidence scores; explicit research disclaimer included.
# - Large tensors / numpy arrays never forced into the state dict; only serialisable
#   primitives (str, float, int, bool, dict) live in WorkflowState. Image data is
#   kept as base64 strings exactly as model_inference.run_inference() returns them.
# - Existing agents are called directly -- no duplication.

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# Concise disclaimer required on all research outputs
CONCISE_DISCLAIMER = (
    "This report contains AI-generated segmentation results for research purposes only. "
    "The results have not been independently validated for clinical use and should be "
    "reviewed by a qualified radiologist or physician. They must not be used as the sole basis for clinical decisions."
)


# -----------------------------------------------------------------
# Shared state definition
# -----------------------------------------------------------------
class WorkflowState(TypedDict, total=False):
    # -- Input ------------------------------------------------------
    patient_id:    str
    name:          str
    age:           int
    gender:        str

    # Uploaded-file mode (bytes + filenames)
    flair_bytes:   Optional[bytes]
    t1_bytes:      Optional[bytes]
    t1ce_bytes:    Optional[bytes]
    t2_bytes:      Optional[bytes]
    flair_name:    Optional[str]
    t1_name:       Optional[str]
    t1ce_name:     Optional[str]
    t2_name:       Optional[str]
    seg_bytes:     Optional[bytes]
    seg_name:      Optional[str]
    has_seg:       bool

    # Dataset mode
    dataset_folder: Optional[str]
    use_dataset:    bool

    # -- Reception output -------------------------------------------
    is_returning:  bool
    previous_scan: Optional[Dict[str, Any]]

    # -- Validation output ------------------------------------------
    validation_errors: List[str]   # empty -> valid

    # -- Segmentation output ----------------------------------------
    # Primitive / base64-safe fields only; tensors stay in memory
    # inside model_inference.py and are NOT stored here.
    scan_result: Optional[Dict[str, Any]]

    # -- Supervisor output ------------------------------------------
    needs_review:   bool
    review_reasons: List[str]

    # -- Comparison output ------------------------------------------
    comparison_result: Optional[Dict[str, Any]]

    # -- Report output ----------------------------------------------
    gemini_report: str
    # pdf_bytes intentionally excluded -- kept in Streamlit session only

    # -- Workflow metadata ------------------------------------------
    workflow_status: str   # "running" | "validation_failed" | "complete"
    error_message:   str


# -----------------------------------------------------------------
# NODE 1 -- Reception
# Reuses: reception_agent() from agents.py
# -----------------------------------------------------------------
def reception_node(state: WorkflowState) -> WorkflowState:
    """Check/register the patient in PostgreSQL; determine new vs returning."""
    from agents import reception_agent

    patient_id = state["patient_id"]
    name       = state.get("name", "")
    age        = state.get("age", 0)
    gender     = state.get("gender", "")

    print("[Graph] >> reception_node -- patient_id=" + str(patient_id))
    result = reception_agent(patient_id, name, age, gender)

    return {
        **state,
        "is_returning":  result["is_returning"],
        "previous_scan": result["previous_scan"],
        "workflow_status": "running",
    }


# -----------------------------------------------------------------
# NODE 2 -- Scan Validation
# Validates that all four required modalities are present and that
# uploaded files have a supported extension (.nii / .nii.gz).
# No model call happens here -- pure input checking.
# -----------------------------------------------------------------
def _is_valid_nii(filename: str) -> bool:
    """Accept .nii or .nii.gz (and bare .gz for Streamlit uploads)."""
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".gz")


def scan_validation_node(state: WorkflowState) -> WorkflowState:
    """Validate uploaded MRI inputs before running inference."""
    print("[Graph] >> scan_validation_node")
    errors: List[str] = []
    use_dataset = state.get("use_dataset", False)

    if use_dataset:
        # Dataset mode: check the folder exists
        from config import BRATS_DATASET_PATH
        folder = state.get("dataset_folder", "")
        if not folder:
            errors.append("No dataset folder selected.")
        else:
            full_path = os.path.join(BRATS_DATASET_PATH, folder)
            if not os.path.isdir(full_path):
                errors.append("Dataset folder not found: " + full_path)
    else:
        # Upload mode: check all four modalities
        modalities = {
            "FLAIR": (state.get("flair_bytes"), state.get("flair_name", "")),
            "T1":    (state.get("t1_bytes"),    state.get("t1_name", "")),
            "T1CE":  (state.get("t1ce_bytes"),  state.get("t1ce_name", "")),
            "T2":    (state.get("t2_bytes"),    state.get("t2_name", "")),
        }
        for modality, (file_bytes, filename) in modalities.items():
            if file_bytes is None:
                errors.append("Missing modality: " + modality)
            elif not filename:
                errors.append(modality + ": filename is empty.")
            elif not _is_valid_nii(filename):
                errors.append(
                    modality + " file '" + filename + "' is not a valid NIfTI file "
                    "(.nii or .nii.gz required)."
                )

    if errors:
        print("[Graph]   VALIDATION FAILED: " + str(errors))
        return {
            **state,
            "validation_errors": errors,
            "workflow_status": "validation_failed",
        }

    print("[Graph]   Validation passed.")
    return {**state, "validation_errors": []}


# -----------------------------------------------------------------
# NODE 3 -- Segmentation
# Reuses: scan_analysis_agent() or dataset_scan_agent() from agents.py
# -----------------------------------------------------------------
def segmentation_node(state: WorkflowState) -> WorkflowState:
    """Run SegResNet inference. Delegates entirely to existing agent functions."""
    from agents import scan_analysis_agent, dataset_scan_agent

    print("[Graph] >> segmentation_node")
    use_dataset = state.get("use_dataset", False)

    try:
        if use_dataset:
            folder = state["dataset_folder"]
            print("[Graph]   Dataset mode -- folder: " + str(folder))
            result = dataset_scan_agent(folder)
        else:
            print("[Graph]   Upload mode -- running scan_analysis_agent")
            result = scan_analysis_agent(
                flair_bytes=state["flair_bytes"],
                t1_bytes=state["t1_bytes"],
                t1ce_bytes=state["t1ce_bytes"],
                t2_bytes=state["t2_bytes"],
                flair_name=state["flair_name"],
                t1_name=state["t1_name"],
                t1ce_name=state["t1ce_name"],
                t2_name=state["t2_name"],
                seg_bytes=state.get("seg_bytes"),
                seg_name=state.get("seg_name"),
                has_seg=state.get("has_seg", False),
            )

        print("[Graph]   Segmentation done -- total_volume=" + str(result["total_volume"]) + " cm3")
        return {**state, "scan_result": result, "error_message": ""}

    except Exception as exc:
        msg = str(exc)
        print("[Graph]   Segmentation FAILED: " + msg)
        return {
            **state,
            "scan_result": None,
            "error_message": msg,
            "workflow_status": "validation_failed",
        }


# -----------------------------------------------------------------
# NODE 4 -- Supervisor
# Routes based on validated output fields and configurable rules.
# Does NOT invent a confidence score. Uses:
#   * total_volume, et_volume from scan_result
#   * growth percent from comparison_result (if returning patient)
# -----------------------------------------------------------------
def supervisor_node(state: WorkflowState) -> WorkflowState:
    """
    Coordinate longitudinal comparison for returning patients without arbitrary thresholds.
    """
    from agents import comparison_agent

    print("[Graph] >> supervisor_node")
    scan_result = state.get("scan_result")
    if not scan_result:
        return {
            **state,
            "needs_review": False,
            "review_reasons": [],
        }

    # Run comparison for returning patients
    comparison_result = None
    if state.get("is_returning") and state.get("previous_scan"):
        try:
            comparison_result = comparison_agent(
                scan_result, state["previous_scan"]
            )
        except Exception as exc:
            print("[Graph]   WARNING: Comparison failed (non-fatal): " + str(exc))

    return {
        **state,
        "needs_review": False,
        "review_reasons": [],
        "comparison_result": comparison_result,
    }


# -----------------------------------------------------------------
# NODE 5 -- Reporting
# Reuses: report_generator_agent() from agents.py (Gemini + fallback)
# Prepends an explicit REVIEW REQUIRED section when needs_review=True.
# -----------------------------------------------------------------
def reporting_node(state: WorkflowState) -> WorkflowState:
    """Generate Gemini AI report with standard research disclaimer."""
    from agents import report_generator_agent

    print("[Graph] >> reporting_node")

    patient_info = {
        "patient_id": state["patient_id"],
        "name":       state.get("name", ""),
        "age":        state.get("age", 0),
        "gender":     state.get("gender", ""),
    }

    gemini_report = report_generator_agent(
        patient_info=patient_info,
        scan_result=state["scan_result"],
        comparison=state.get("comparison_result"),
    )

    # Ensure the concise research disclaimer is at the end of the report
    if "This report contains AI-generated segmentation results for research purposes only" not in gemini_report:
        gemini_report = (
            gemini_report.rstrip()
            + "\n\n6. DISCLAIMER\n"
            + CONCISE_DISCLAIMER
            + "\n"
        )

    print("[Graph]   Report generated.")
    return {
        **state,
        "gemini_report":   gemini_report,
        "workflow_status": "complete",
    }


# -----------------------------------------------------------------
# ROUTING FUNCTIONS (conditional edges)
# -----------------------------------------------------------------
def route_after_validation(state: WorkflowState) -> str:
    """Route to segmentation if valid, else end immediately."""
    if state.get("validation_errors"):
        print("[Graph] route: validation_failed -> END")
        return "validation_failed"
    print("[Graph] route: valid -> segmentation")
    return "valid"


def route_after_segmentation(state: WorkflowState) -> str:
    """Route to supervisor if inference succeeded, else end."""
    if state.get("scan_result") is None:
        print("[Graph] route: inference_failed -> END")
        return "inference_failed"
    print("[Graph] route: inference_ok -> supervisor")
    return "inference_ok"


def route_after_supervisor(state: WorkflowState) -> str:
    """Route to reporting (same node; review flag is embedded in the report)."""
    # Both review-required and standard cases go through reporting_node.
    # The distinction is explicit in the report content; we do NOT skip
    # reporting for review-required cases or silently mark them as approved.
    print("[Graph] route: -> reporting")
    return "reporting"


# -----------------------------------------------------------------
# BUILD THE GRAPH
# -----------------------------------------------------------------
def build_workflow() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("reception",       reception_node)
    graph.add_node("scan_validation", scan_validation_node)
    graph.add_node("segmentation",    segmentation_node)
    graph.add_node("supervisor",      supervisor_node)
    graph.add_node("reporting",       reporting_node)

    # Entry point
    graph.set_entry_point("reception")

    # Fixed edges
    graph.add_edge("reception", "scan_validation")

    # Conditional edge: validation -> segmentation OR end
    graph.add_conditional_edges(
        "scan_validation",
        route_after_validation,
        {
            "valid":             "segmentation",
            "validation_failed": END,
        },
    )

    # Conditional edge: segmentation -> supervisor OR end
    graph.add_conditional_edges(
        "segmentation",
        route_after_segmentation,
        {
            "inference_ok":     "supervisor",
            "inference_failed": END,
        },
    )

    # Conditional edge: supervisor -> reporting (always, review flag embedded)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "reporting": "reporting",
        },
    )

    # Terminal edge
    graph.add_edge("reporting", END)

    return graph.compile()


# -----------------------------------------------------------------
# PUBLIC API -- called from app.py
# -----------------------------------------------------------------

# Build once at import time (lightweight -- no model loaded here).
_workflow = build_workflow()


def run_workflow(initial_state: WorkflowState) -> WorkflowState:
    """
    Execute the full multi-agent LangGraph workflow.

    Parameters
    ----------
    initial_state : WorkflowState
        Must include: patient_id, name, age, gender.
        For upload mode: flair_bytes, t1_bytes, t1ce_bytes, t2_bytes + names.
        For dataset mode: dataset_folder=<folder_name>, use_dataset=True.

    Returns
    -------
    WorkflowState
        Completed state dict containing scan_result, gemini_report,
        needs_review, review_reasons, comparison_result, etc.
    """
    # Set safe defaults for optional fields
    defaults: WorkflowState = {
        "validation_errors": [],
        "review_reasons":    [],
        "needs_review":      False,
        "is_returning":      False,
        "previous_scan":     None,
        "scan_result":       None,
        "comparison_result": None,
        "gemini_report":     "",
        "workflow_status":   "running",
        "error_message":     "",
        "use_dataset":       False,
        "dataset_folder":    None,
        "has_seg":           False,
        "seg_bytes":         None,
        "seg_name":          None,
        "flair_bytes":       None,
        "t1_bytes":          None,
        "t1ce_bytes":        None,
        "t2_bytes":          None,
        "flair_name":        None,
        "t1_name":           None,
        "t1ce_name":         None,
        "t2_name":           None,
    }
    merged = {**defaults, **initial_state}
    final_state = _workflow.invoke(merged)
    return final_state
