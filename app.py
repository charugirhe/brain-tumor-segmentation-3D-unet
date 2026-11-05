# app.py
# This is the MAIN UI of the project built with Streamlit
# Think of it as the FACE of the hospital system!

import streamlit as st
import pandas as pd
import base64
import os
from datetime import datetime

# ─────────────────────────────────────────
# PAGE CONFIG — Must be first Streamlit command!
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Analysis System",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        padding: 14px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0d9488;
    }
    .grew { color: #b91c1c; font-weight: 600; }
    .shrunk { color: #047857; font-weight: 600; }
    .stable { color: #b45309; font-weight: 600; }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 30px;
        padding: 12px;
        border-top: 1px solid #e2e8f0;
    }
    /* Red Analyze MRI Scans Button */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #dc2626 !important;
        border-color: #b91c1c !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #b91c1c !important;
        border-color: #991b1b !important;
    }
    /* Subtle blue/teal accents */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #0d9488 !important;
        color: #0d9488 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🧠 Brain Tumor Analysis")
st.caption("AI-Powered MRI Segmentation & Analysis")

# ─────────────────────────────────────────
# LAZY IMPORTS — Load heavy modules only when needed
# This prevents black screen on startup!
# ─────────────────────────────────────────
@st.cache_resource
def load_database():
    from database import init_database
    init_database()
    return True

@st.cache_resource
def load_agents():
    from agents import (
        reception_agent,
        scan_analysis_agent,
        dataset_scan_agent,
        comparison_agent,
        report_generator_agent,
    )
    return (
        reception_agent,
        scan_analysis_agent,
        dataset_scan_agent,
        comparison_agent,
        report_generator_agent,
    )

# Load database in background
with st.spinner("🔄 Initializing system..."):
    try:
        load_database()
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        st.stop()

# Load config
from config import BRATS_DATASET_PATH
from database import save_scan, get_all_scans
from report_generator import generate_pdf_report

# ─────────────────────────────────────────
# SIDEBAR — Patient Information
# ─────────────────────────────────────────
with st.sidebar:
    st.header("👤 Patient Information")

    patient_id = st.text_input(
        "Patient ID *",
        placeholder="e.g. PT001",
        help="Required — unique identifier for the patient"
    )

    patient_name = st.text_input(
        "Patient Name *",
        placeholder="e.g. John Doe",
        help="Required"
    )

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.divider()

    # Color legend
    st.markdown("### 🎨 Color Legend")
    st.markdown("🔴 **NCR** — Necrotic Core")
    st.markdown("🟢 **ED** — Edema")
    st.markdown("🔵 **ET** — Enhancing Tumor")

# ─────────────────────────────────────────
# MAIN AREA — Two Tabs
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload MRI Files", "📁 Select from BraTS Dataset"])

# Variables to track what input method is used
uploaded_files_ready = False
dataset_patient_ready = False
selected_folder = None
seg_file = None

# ── TAB 1 — Upload MRI Files ──
with tab1:
    st.subheader("Upload MRI Modalities")

    col1, col2 = st.columns(2)

    with col1:
        flair_file = st.file_uploader(
            "FLAIR Scan *",
            type=["nii", "gz"],
            key="flair"
        )
        t1ce_file = st.file_uploader(
            "T1CE Scan *",
            type=["nii", "gz"],
            key="t1ce"
        )

    with col2:
        t1_file = st.file_uploader(
            "T1 Scan *",
            type=["nii", "gz"],
            key="t1"
        )
        t2_file = st.file_uploader(
            "T2 Scan *",
            type=["nii", "gz"],
            key="t2"
        )

    # Optional ground truth segmentation
    with st.expander("📎 Optional: Upload Ground Truth Segmentation"):
        seg_file = st.file_uploader(
            "Segmentation File (.nii or .nii.gz)",
            type=["nii", "gz"],
            key="seg"
        )
        if seg_file:
            st.success("✅ Ground truth segmentation uploaded!")

    # Show which files are missing
    missing = []
    if not flair_file: missing.append("FLAIR")
    if not t1_file:    missing.append("T1")
    if not t1ce_file:  missing.append("T1CE")
    if not t2_file:    missing.append("T2")

    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)}")
    else:
        st.success("✅ All 4 MRI modalities uploaded!")
        uploaded_files_ready = True

# ── TAB 2 — BraTS Dataset ──
with tab2:
    st.subheader("Select from BraTS Dataset")

    if not BRATS_DATASET_PATH or not os.path.exists(BRATS_DATASET_PATH):
        st.warning("""
        ⚠️ BraTS dataset path not configured!

        Please set **BRATS_DATASET_PATH** in your `.env` file:
```
        BRATS_DATASET_PATH=C:/Users/Dell/your_brats_folder
```
        Then restart the app.
        """)
    else:
        # List all patient folders
        folders = sorted([
            f for f in os.listdir(BRATS_DATASET_PATH)
            if os.path.isdir(os.path.join(BRATS_DATASET_PATH, f))
        ])

        if not folders:
            st.error("❌ No patient folders found in the dataset path!")
        else:
            selected_folder = st.selectbox(
                "Select Patient Folder",
                folders,
                help="Select a BraTS patient folder to analyze"
            )
            st.info(
                "ℹ️ Ground truth segmentation will be "
                "loaded automatically from the dataset!"
            )
            dataset_patient_ready = True

# ─────────────────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────────────────
st.divider()

# Button is disabled until all required inputs are filled
button_disabled = not (
    patient_id and
    patient_name and
    (uploaded_files_ready or dataset_patient_ready)
)

analyze_clicked = st.button(
    "🔬 Analyze MRI Scans",
    type="primary",
    disabled=button_disabled,
    use_container_width=True
)

if button_disabled:
    st.caption(
        "⚠️ Please fill Patient ID, Patient Name, "
        "and upload all 4 MRI files (or select a dataset patient)"
    )

# ─────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────
if analyze_clicked:

    # LangGraph multi-agent workflow
    from langgraph_workflow import run_workflow

    patient_info_result = None
    scan_result = None
    comparison_result = None
    gemini_report = ""

    with st.status("Analyzing MRI scans...", expanded=True) as status:
        try:
            # Build the initial state for the graph
            if dataset_patient_ready and selected_folder:
                initial_state = {
                    "patient_id":     patient_id,
                    "name":           patient_name,
                    "age":            age,
                    "gender":         gender,
                    "use_dataset":    True,
                    "dataset_folder": selected_folder,
                }
            else:
                initial_state = {
                    "patient_id":  patient_id,
                    "name":        patient_name,
                    "age":         age,
                    "gender":      gender,
                    "use_dataset": False,
                    "flair_bytes": flair_file.read(),
                    "t1_bytes":    t1_file.read(),
                    "t1ce_bytes":  t1ce_file.read(),
                    "t2_bytes":    t2_file.read(),
                    "flair_name":  flair_file.name,
                    "t1_name":     t1_file.name,
                    "t1ce_name":   t1ce_file.name,
                    "t2_name":     t2_file.name,
                    "seg_bytes":   seg_file.read() if seg_file else None,
                    "seg_name":    seg_file.name  if seg_file else None,
                    "has_seg":     seg_file is not None,
                }

            wf_result = run_workflow(initial_state)

            # Handle validation failure
            if wf_result.get("validation_errors"):
                errs = wf_result["validation_errors"]
                st.error("Validation error: " + "; ".join(errs))
                status.update(label="Scan validation failed", state="error")
                st.stop()

            # Handle inference failure
            if wf_result.get("scan_result") is None:
                st.error("Analysis could not proceed: " + wf_result.get("error_message", "unknown error"))
                status.update(label="Inference failed", state="error")
                st.stop()

            # Extract results from completed workflow state
            patient_info_result = {
                "patient_id":    patient_id,
                "name":          patient_name,
                "age":           age,
                "gender":        gender,
                "is_returning":  wf_result.get("is_returning", False),
                "previous_scan": wf_result.get("previous_scan"),
            }
            scan_result       = wf_result["scan_result"]
            comparison_result = wf_result.get("comparison_result")
            gemini_report     = wf_result.get("gemini_report", "")

            # Workflow execution details (rendered directly inside st.status without nested expander)
            if patient_info_result["is_returning"]:
                st.write("• Patient record: Returning patient found")
            else:
                st.write("• Patient record: New patient registered")
            st.write(f"• Segmentation: SegResNet complete ({scan_result['total_volume']} cm³)")
            if comparison_result:
                st.write("• Comparison: Longitudinal analysis complete")

            # Workflow completed successfully

            status.update(label="Analysis complete", state="complete")

        except Exception as e:
            st.error("Workflow error: " + str(e))
            status.update(label="Workflow failed!", state="error")
            st.stop()

    # ── Save to Database ──
    try:
        save_scan(
            patient_id=patient_id,
            scan_data={**scan_result, "gemini_report": gemini_report},
            file_paths={
             "flair": flair_file.name if (not dataset_patient_ready and flair_file) else "",
             "t1":    t1_file.name if (not dataset_patient_ready and t1_file) else "",
             "t1ce":  t1ce_file.name if (not dataset_patient_ready and t1ce_file) else "",
             "t2":    t2_file.name if (not dataset_patient_ready and t2_file) else "",
}
        )
    except Exception as e:
        st.warning(f"⚠️ Could not save to database: {e}")

    # Store results in session state so user interactions (e.g. XAI toggles, PDF download) don't trigger re-analysis!
    st.session_state["analysis_data"] = {
        "scan_result": scan_result,
        "patient_info_result": patient_info_result,
        "comparison_result": comparison_result,
        "gemini_report": gemini_report,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
    }

# ─────────────────────────────────────────
# RESULTS SECTION (PERSISTENT VIA SESSION STATE)
# ─────────────────────────────────────────
if "analysis_data" in st.session_state:
    res_data = st.session_state["analysis_data"]
    scan_result = res_data["scan_result"]
    patient_info_result = res_data["patient_info_result"]
    comparison_result = res_data["comparison_result"]
    gemini_report = res_data["gemini_report"]
    res_patient_id = res_data["patient_id"]
    res_patient_name = res_data["patient_name"]
    res_age = res_data["age"]
    res_gender = res_data["gender"]

    st.divider()
    st.header("📊 Analysis Results")

    # Tumor detected banner
    if scan_result["tumor_detected"]:
        st.error("🔴 TUMOR DETECTED")
    else:
        st.success("🟢 NO TUMOR DETECTED")

    # ── 4 Metric Cards ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔘 Total Volume", f"{scan_result['total_volume']} cm³")
    with col2:
        st.metric("🔴 NCR Volume",
                  f"{scan_result['ncr_volume']} cm³",
                  f"{scan_result['ncr_percent']}%")
    with col3:
        st.metric("🟢 ED Volume",
                  f"{scan_result['ed_volume']} cm³",
                  f"{scan_result['ed_percent']}%")
    with col4:
        st.metric("🔵 ET Volume",
                  f"{scan_result['et_volume']} cm³",
                  f"{scan_result['et_percent']}%")

    st.divider()

    # ── MRI Images ──
    st.subheader("🖼️ MRI Scan Visualization")

    image_cols = []
    if scan_result.get("flair_image"):
        image_cols.append(("1. Original FLAIR", scan_result["flair_image"]))
    if scan_result.get("gt_image"):
        image_cols.append(("2. Ground Truth Mask", scan_result["gt_image"]))
    if scan_result.get("predicted_image"):
        image_cols.append(("3. Predicted Mask", scan_result["predicted_image"]))
    if scan_result.get("overlay_image"):
        image_cols.append(("4. MRI + Overlay", scan_result["overlay_image"]))

    if image_cols:
        cols = st.columns(len(image_cols))
        for i, (caption, b64_img) in enumerate(image_cols):
            with cols[i]:
                st.image(
                    f"data:image/png;base64,{b64_img}",
                    caption=caption,
                    use_column_width=True
                )

    # Color legend below images
    st.caption("**Legend:** 🔴 Red = NCR | 🟢 Green = ED | 🔵 Blue = ET")

    # ── Explainable AI (XAI) Section ──
    if scan_result.get("gradcam_image") or scan_result.get("modality_attribution"):
        st.divider()
        st.subheader("🔍 Explainable AI (XAI) & Model Interpretability")
        st.caption(
            "Visualizing 3D SegResNet spatial attention via 3D Grad-CAM, "
            "input sequence importance via Input × Gradient Attribution, and prediction confidence."
        )

        xai_col1, xai_col2 = st.columns([1.3, 1])

        with xai_col1:
            st.markdown("#### 🎯 3D Grad-CAM Attention Focus")

            class_options = ["Overall Tumor Focus"]
            if scan_result.get("class_attentions"):
                class_options.extend(list(scan_result["class_attentions"].keys()))

            selected_cam_type = st.radio(
                "Select Model Attention Layer / Sub-Region:",
                options=class_options,
                horizontal=True,
                key="xai_cam_select"
            )

            display_cam_b64 = scan_result.get("gradcam_image")
            if selected_cam_type != "Overall Tumor Focus" and scan_result.get("class_attentions"):
                display_cam_b64 = scan_result["class_attentions"].get(
                    selected_cam_type, scan_result.get("gradcam_image")
                )

            cam_subcol1, cam_subcol2 = st.columns(2)
            with cam_subcol1:
                st.image(
                    f"data:image/png;base64,{display_cam_b64}",
                    caption=f"Grad-CAM Heatmap ({selected_cam_type})",
                    use_column_width=True
                )
            with cam_subcol2:
                if scan_result.get("confidence_image"):
                    st.image(
                        f"data:image/png;base64,{scan_result['confidence_image']}",
                        caption="Voxel Confidence / Certainty Map",
                        use_column_width=True
                    )
            st.caption(
                "🔥 **Heatmap Legend:** Warmer colors (Red/Yellow) signify peak neural network focus. "
                "Cold/Dark areas had negligible influence on the classification."
            )

        with xai_col2:
            st.markdown("#### 📊 MRI Sequence Importance")
            attrib = scan_result.get("modality_attribution", {})
            if attrib:
                attrib_df = pd.DataFrame(
                    list(attrib.items()),
                    columns=["Modality", "Contribution (%)"]
                )
                st.bar_chart(attrib_df.set_index("Modality"), color="#3949ab")

                top_m = scan_result.get("top_modality", "T1CE")
                top_pct = attrib.get(top_m, 0)

                clinical_notes = {
                    "T1CE": "T1-weighted Contrast Enhanced (T1CE) is the primary driver. Strong gadolinium uptake indicates breakdown of the blood-brain barrier characteristic of high-grade active tumor.",
                    "FLAIR": "FLAIR is the primary driver. Hyperintensity on fluid-attenuated inversion recovery highlights vasogenic edema surrounding the tumor core.",
                    "T1": "T1-weighted sequence contributed anatomical structural boundaries and necrotic core differentiation.",
                    "T2": "T2-weighted sequence contributed water/cystic fluid contrast."
                }
                note = clinical_notes.get(
                    top_m,
                    "Multi-sequence feature attribution verified by Input × Gradient."
                )

                st.info(f"💡 **Clinical Driver ({top_m}: {top_pct}%):** {note}")

    # ── Comparison Section ──
    if comparison_result:
        st.divider()
        st.subheader(
            f"📈 Comparison with Previous Scan "
            f"({comparison_result['previous_date']})"
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tc = comparison_result["total_change"]
            st.metric(
                "Total Volume",
                f"{scan_result['total_volume']} cm³",
                f"{tc['absolute']} cm³ ({tc['trend']})"
            )
        with col2:
            nc = comparison_result["ncr_change"]
            st.metric(
                "NCR Volume",
                f"{scan_result['ncr_volume']} cm³",
                f"{nc['absolute']} cm³ ({nc['trend']})"
            )
        with col3:
            ec = comparison_result["ed_change"]
            st.metric(
                "ED Volume",
                f"{scan_result['ed_volume']} cm³",
                f"{ec['absolute']} cm³ ({ec['trend']})"
            )
        with col4:
            etc = comparison_result["et_change"]
            st.metric(
                "ET Volume",
                f"{scan_result['et_volume']} cm³",
                f"{etc['absolute']} cm³ ({etc['trend']})"
            )

    # ── AI Report ──
    st.divider()
    with st.expander("📝 AI Medical Analysis Report", expanded=True):
        # Format the markdown cleanly: remove raw separator lines and excessive decorative characters
        clean_display_lines = []
        for line in gemini_report.split('\n'):
            stripped = line.strip()
            # Omit ugly decorative lines like '=======' or '-------'
            if stripped and len(stripped) > 3 and set(stripped).issubset({'-', '=', '*', '_', '#'}):
                continue
            if '[RADIOLOGIST REVIEW REQUIRED]' in stripped or 'RADIOLOGIST REVIEW REQUIRED' in stripped:
                continue
            clean_display_lines.append(line)

        final_display_text = "\n".join(clean_display_lines).strip()
        st.markdown(final_display_text)

    # ── PDF Download ──
    st.divider()
    try:
        pdf_bytes = generate_pdf_report(
            patient_info={
                "patient_id": res_patient_id,
                "name": res_patient_name,
                "age": res_age,
                "gender": res_gender,
            },
            scan_result=scan_result,
            comparison=comparison_result,
            gemini_report=gemini_report,
            overlay_b64=scan_result.get("overlay_image", ""),
            predicted_b64=scan_result.get("predicted_image", ""),
            gt_b64=scan_result.get("gt_image", ""),
            gradcam_b64=scan_result.get("gradcam_image", ""),
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"brain_tumor_report_{res_patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"❌ Could not generate PDF: {e}")

# ─────────────────────────────────────────
# SCAN HISTORY SECTION
# ─────────────────────────────────────────
if patient_id:
    st.divider()
    st.header("📋 Scan History")

    try:
        all_scans = get_all_scans(patient_id)

        if all_scans:
            df = pd.DataFrame(all_scans)
            df.columns = [
                "Date", "Total (cm³)",
                "NCR (cm³)", "ED (cm³)", "ET (cm³)"
            ]
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Tumor Volume Over Time")
            chart_df = df.set_index("Date")
            st.line_chart(chart_df)
        else:
            st.info("No previous scans found for this patient.")
    except Exception as e:
        st.warning(f"Could not load scan history: {e}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.caption("🧠 Brain Tumor Analysis System | AI-Powered Medical Imaging | ⚠️ For Research Purposes Only — Not for Clinical Use")
