# agents.py
# This file contains all 4 AI agents that work together
# Each agent has one specific job — like a team of doctors!

from email.mime import base
import os
import uuid
import tempfile
from datetime import datetime

import model_inference
from database import (
    patient_exists,
    get_latest_scan,
    get_patient_info,
    save_patient,
    save_scan,
)
from config import BRATS_DATASET_PATH

# ─────────────────────────────────────────
# AGENT 1 — Reception Agent
# Job: Check if patient is new or returning
# ─────────────────────────────────────────
def reception_agent(patient_id, name, age, gender):
    print(f"\n[Agent 1] [Hospital] Checking records for patient: {patient_id}")

    # Check if patient exists in database
    is_returning = patient_exists(patient_id)

    previous_scan = None
    if is_returning:
        print(f"[Agent 1] [OK] Returning patient found!")
        previous_scan = get_latest_scan(patient_id)
        if previous_scan:
            print(f"[Agent 1] Last scan date: {previous_scan.get('scan_date')}")
        else:
            is_returning = False
            print(f"[Agent 1] No previous scans found, treating as new patient!")
    else:
        print(f"[Agent 1] [NEW] New patient -- saving to database...")
        save_patient(patient_id, name, age, gender)
        print(f"[Agent 1] [OK] Patient saved!")

    return {
        "patient_id":    patient_id,
        "name":          name,
        "age":           age,
        "gender":        gender,
        "is_returning":  is_returning,
        "previous_scan": previous_scan,
    }

# ─────────────────────────────────────────
# AGENT 2 — Scan Analysis Agent
# Job: Take uploaded MRI files and run inference
# ─────────────────────────────────────────
def scan_analysis_agent(
    flair_bytes, t1_bytes, t1ce_bytes, t2_bytes,
    flair_name, t1_name, t1ce_name, t2_name,
    seg_bytes=None, seg_name=None, has_seg=False
):
    print(f"\n[Agent 2] [Scan] Starting scan analysis...")

    # We'll store temp file paths here so we can clean up later
    tmp_files = []

    try:
        # Save each uploaded file to a temp location in /tmp
        def save_tmp(file_bytes, filename):
            # Create unique filename to avoid conflicts
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            tmp_path = os.path.join(tempfile.gettempdir(), unique_name)
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)
            tmp_files.append(tmp_path)
            print(f"[Agent 2] Saved temp file: {tmp_path}")
            return tmp_path

        # Save all 4 MRI modalities
        flair_path = save_tmp(flair_bytes, flair_name)
        t1_path    = save_tmp(t1_bytes,    t1_name)
        t1ce_path  = save_tmp(t1ce_bytes,  t1ce_name)
        t2_path    = save_tmp(t2_bytes,    t2_name)

        # Build the data dictionary for inference
        data_dict = {
            "flair": flair_path,
            "t1":    t1_path,
            "t1ce": t1ce_path,
            "t2":    t2_path,
        }

        # Save seg file if provided
        if has_seg and seg_bytes is not None:
            seg_path = save_tmp(seg_bytes, seg_name)
            data_dict["seg"] = seg_path
            print(f"[Agent 2] Ground truth segmentation included!")

        # Run the actual model inference
        print(f"[Agent 2] Running model inference...")
        result = model_inference.run_inference(data_dict, has_seg=has_seg)
        print(f"[Agent 2] [OK] Inference complete!")

        return result

    finally:
        # ALWAYS clean up temp files even if inference fails
        for tmp_path in tmp_files:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"[Agent 2] Cleaned up: {tmp_path}")
            except Exception as e:
                print(f"[Agent 2] Warning: Could not delete {tmp_path}: {e}")

# ─────────────────────────────────────────
# AGENT 2B — Dataset Scan Agent
# Job: Read directly from BraTS dataset folder
# ─────────────────────────────────────────
def dataset_scan_agent(patient_folder):
    print(f"\n[Agent 2B] [Dataset] Loading from BraTS dataset: {patient_folder}")

    # Build full paths to all files
    base = os.path.join(BRATS_DATASET_PATH, patient_folder)

    # Auto-detect .nii or .nii.gz
    def find_file(base, name):
        for ext in [".nii.gz", ".nii"]:
            path = os.path.join(base, f"{name}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find {name} in {base}")

    flair_path = find_file(base, f"{patient_folder}_flair")
    t1_path    = find_file(base, f"{patient_folder}_t1")
    t1ce_path  = find_file(base, f"{patient_folder}_t1ce")
    t2_path    = find_file(base, f"{patient_folder}_t2")
    seg_path   = find_file(base, f"{patient_folder}_seg")

    # Check all files exist
    for path in [flair_path, t1_path, t1ce_path, t2_path, seg_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Agent 2B] ❌ File not found: {path}")
        print(f"[Agent 2B] [OK] Found: {path}")

    # Build data dict — no temp files needed, read directly!
    data_dict = {
        "flair": flair_path,
        "t1":    t1_path,
        "t1ce": t1ce_path,
        "t2":    t2_path,
        "seg":   seg_path,
    }

    print(f"[Agent 2B] Running model inference...")
    result = model_inference.run_inference(data_dict, has_seg=True)
    print(f"[Agent 2B] [OK] Dataset scan complete!")

    return result

# ─────────────────────────────────────────
# AGENT 3 — Comparison Agent
# Job: Compare current scan with previous scan
# ─────────────────────────────────────────
def comparison_agent(current_scan, previous_scan):
    print(f"\n[Agent 3] [Stats] Comparing current scan with previous scan...")

    def calculate_change(current_val, previous_val):
        """Calculate absolute change, percent change, and trend"""
        absolute = round(current_val - previous_val, 2)

        if previous_val == 0:
            trend = "NEW"
            percent = 0.0
        elif absolute > 0:
            trend = "GREW"
            percent = round((absolute / previous_val) * 100, 1)
        elif absolute < 0:
            trend = "SHRANK"
            percent = round((absolute / previous_val) * 100, 1)
        else:
            trend = "STABLE"
            percent = 0.0

        return {
            "absolute": absolute,
            "percent":  percent,
            "trend":    trend,
        }

    # Compare all 4 volume metrics
    total_change = calculate_change(
        current_scan["total_volume"],
        previous_scan["total_volume"]
    )
    ncr_change = calculate_change(
        current_scan["ncr_volume"],
        previous_scan["ncr_volume"]
    )
    ed_change = calculate_change(
        current_scan["ed_volume"],
        previous_scan["ed_volume"]
    )
    et_change = calculate_change(
        current_scan["et_volume"],
        previous_scan["et_volume"]
    )

    print(f"[Agent 3] Total volume change: {total_change['absolute']} cm3 ({total_change['trend']})")
    print(f"[Agent 3] [OK] Comparison complete!")

    return {
        "previous_date":      previous_scan.get("scan_date", "Unknown"),
        "prev_total_volume":  previous_scan["total_volume"],
        "prev_ncr_volume":    previous_scan["ncr_volume"],
        "prev_ed_volume":     previous_scan["ed_volume"],
        "prev_et_volume":     previous_scan["et_volume"],
        "total_change":       total_change,
        "ncr_change":         ncr_change,
        "ed_change":          ed_change,
        "et_change":          et_change,
    }

# ─────────────────────────────────────────
# AGENT 4 — Report Generator Agent
# Job: Generate AI report using Gemini
# ─────────────────────────────────────────
def report_generator_agent(patient_info, scan_result, comparison=None):
    print(f"\n[Agent 4] [Report] Generating AI medical report...")

    try:
        from google import genai
        from config import GEMINI_API_KEY

        # Build the prompt for Gemini
        prompt = f"""
You are an expert neuro-radiologist AI assistant. 
Write a detailed medical report for the following brain tumor scan.

PATIENT INFORMATION:
- Patient ID : {patient_info['patient_id']}
- Name       : {patient_info['name']}
- Age        : {patient_info['age']}
- Gender     : {patient_info['gender']}

CURRENT SCAN RESULTS:
- Tumor Detected : {'Yes' if scan_result['tumor_detected'] else 'No'}
- Total Volume   : {scan_result['total_volume']} cm³
- NCR Volume     : {scan_result['ncr_volume']} cm³ ({scan_result['ncr_percent']}%)
- ED Volume      : {scan_result['ed_volume']} cm³ ({scan_result['ed_percent']}%)
- ET Volume      : {scan_result['et_volume']} cm³ ({scan_result['et_percent']}%)
"""

        # Add Explainable AI attribution data if present
        if scan_result.get("modality_attribution"):
            prompt += f"""
EXPLAINABLE AI (XAI) & FEATURE ATTRIBUTION:
- Primary MRI Modality Driver: {scan_result.get('top_modality', 'N/A')}
- Modality Attribution Breakdown: {scan_result.get('modality_attribution', {})}
- 3D Grad-CAM Attention: Peak spatial attention centered on primary tumor lesion.
"""

        # Add comparison data if this is a returning patient
        if comparison:
            prompt += f"""
COMPARISON WITH PREVIOUS SCAN ({comparison['previous_date']}):
- Previous Total Volume : {comparison['prev_total_volume']} cm³
- Total Change          : {comparison['total_change']['absolute']} cm³ ({comparison['total_change']['trend']})
- NCR Change            : {comparison['ncr_change']['absolute']} cm³ ({comparison['ncr_change']['trend']})
- ED Change             : {comparison['ed_change']['absolute']} cm³ ({comparison['ed_change']['trend']})
- ET Change             : {comparison['et_change']['absolute']} cm³ ({comparison['et_change']['trend']})
"""

        prompt += """
Please write a concise, professional, structured medical report with these exact sections and headers:

1. ANALYSIS SUMMARY
- State patient details and core objective of scan analysis.
- Note whether a lesion/tumor was detected and total estimated volume.
- Clearly state that this is an AI-generated research output and has not been evaluated or confirmed by a physician.

2. SEGMENTATION MEASUREMENTS
- Present the measurements in a clean structured list or markdown table:
  * Total Tumor Volume (cm³)
  * Necrotic Core (NCR) Volume and percentage
  * Peritumoral Edema (ED) Volume and percentage
  * Enhancing Tumor (ET) Volume and percentage

3. PREVIOUS SCAN COMPARISON (Include only if comparison data is provided above)
- Present numerical changes across sub-regions concisely without redundant introductory text.
- Note volumetric differences objectively. Do not describe unchanged volume as proof that the tumor or patient is clinically stable.

4. MODEL INTERPRETATION & EXPLAINABILITY
- Discuss primary MRI modality contributions based on the XAI attribution above.
- Mention spatial focus patterns observed by 3D Grad-CAM.

5. RECOMMENDATIONS & CLINICAL NOTES
- Recommend clinical correlation and specialist consultation.

6. DISCLAIMER
- Conclude the report with this exact disclaimer text verbatim:
"This report contains AI-generated segmentation results for research purposes only. The results have not been independently validated for clinical use and should be reviewed by a qualified radiologist or physician. They must not be used as the sole basis for clinical decisions."

Style guidelines:
- Avoid excessive punctuation, decorative separator symbols, or conversational filler.
- Keep the report concise, objective, and professionally structured.
"""

        # Call Gemini API
        print(f"[Agent 4] Calling Gemini API...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        report = response.text
        print(f"[Agent 4] [OK] Gemini report generated!")
        return report

    except Exception as e:
        print(f"[Agent 4] [WARNING] Gemini failed: {e}")
        print(f"[Agent 4] Using fallback report...")
        return _fallback_report(patient_info, scan_result, comparison)

# ─────────────────────────────────────────
# FALLBACK — Local report if Gemini fails
# ─────────────────────────────────────────
def _fallback_report(patient_info, scan_result, comparison=None):
    """Generate a basic report locally without Gemini"""

    report = f"""
1. ANALYSIS SUMMARY
Patient ID: {patient_info['patient_id']} | Name: {patient_info['name']} | Age: {patient_info['age']} | Gender: {patient_info['gender']}
Finding: {'A lesion consistent with tumor tissue was segmented.' if scan_result['tumor_detected'] else 'No significant tumor volume detected.'}
Note: This is an AI-generated research analysis and has not been reviewed or confirmed by a physician.

2. SEGMENTATION MEASUREMENTS
- Total Tumor Volume    : {scan_result['total_volume']} cm³
- Necrotic Core (NCR)   : {scan_result['ncr_volume']} cm³ ({scan_result['ncr_percent']}%)
- Peritumoral Edema (ED): {scan_result['ed_volume']} cm³ ({scan_result['ed_percent']}%)
- Enhancing Tumor (ET)  : {scan_result['et_volume']} cm³ ({scan_result['et_percent']}%)
"""

    if comparison:
        report += f"""
3. PREVIOUS SCAN COMPARISON ({comparison['previous_date']})
- Previous Total Volume : {comparison['prev_total_volume']} cm³
- Current Total Volume  : {scan_result['total_volume']} cm³
- Volumetric Change     : {comparison['total_change']['absolute']} cm³ ({comparison['total_change']['trend']})
- Sub-region Differences: NCR {comparison['ncr_change']['absolute']} cm³ | ED {comparison['ed_change']['absolute']} cm³ | ET {comparison['et_change']['absolute']} cm³
(Note: Unchanged volumetric figures do not establish clinical stability.)
"""

    top_m = scan_result.get('top_modality', 'N/A')
    report += f"""
4. MODEL INTERPRETATION & EXPLAINABILITY
- Primary Sequence Contribution: {top_m}
- Multi-modality Attribution   : {scan_result.get('modality_attribution', {})}
- Spatial Attention Focus      : 3D Grad-CAM centered on segmented abnormal signal intensity.

5. RECOMMENDATIONS & CLINICAL NOTES
Clinical evaluation and formal radiological correlation by a licensed specialist are advised.

6. DISCLAIMER
This report contains AI-generated segmentation results for research purposes only. The results have not been independently validated for clinical use and should be reviewed by a qualified radiologist or physician. They must not be used as the sole basis for clinical decisions.
"""
    return report