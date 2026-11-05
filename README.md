# 🧠 Brain Tumor Analysis System

An AI-powered 3D MRI segmentation, explainability (XAI), and clinical report generation system. It combines a **MONAI SegResNet model** (fine-tuned via transfer learning on the BraTS 2020 dataset), **LangGraph multi-agent orchestration**, and **Google Gemini AI** to automate brain tumor detection, interpret model decisions, and generate detailed clinical reports.

---

## 🌟 Key Features

1. **AI-Powered 3D Segmentation:** Auto-detects and segments brain tumors into three clinical sub-regions:
   - 🔴 **NCR** (Necrotic Tumor Core)
   - 🟢 **ED** (Peritumoral Edema)
   - 🔵 **ET** (Enhancing Tumor)
2. **Transfer Learning Performance:** Fine-tuned `SegResNet` architecture with `DiceFocalLoss` and data augmentation, achieving **0.7105 Mean Dice** (outperforming the 0.6863 scratch baseline).
3. **Explainable AI (XAI) & Interpretability:** 
   - 🎯 **3D Grad-CAM:** Visualizes deep residual layer attention heatmaps for overall tumor and per-class sub-regions.
   - 📊 **MRI Modality Attribution:** Quantifies sequence importance (FLAIR vs T1 vs T1CE vs T2) using Input × Gradient saliency.
   - 🛡️ **Voxel Confidence Map:** Maps softmax entropy and model certainty across lesion margins.
4. **Multi-Modal Support:** Upload your own FLAIR, T1, T1CE, and T2 scans, or load directly from the BraTS dataset.
5. **LangGraph Multi-Agent Orchestration:** A structured state graph workflow (`reception` ➔ `scan_validation` ➔ `segmentation` ➔ `supervisor` ➔ `reporting`) that coordinates patient registration, scan validation, 3D inference, historical progression checks, and medical reporting.
6. **Historical Comparison:** Automatically tracks tumor volume changes (growth/shrinkage) for returning patients.
7. **AI Neuro-Radiologist Reports:** Generates professional medical reports with clinical explainability using Google's Gemini API.
8. **PDF Reports & Visualization:** View overlay images, Grad-CAM attention maps, and download a complete, clinical-style PDF report.

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python installed (Python 3.8+ recommended).

### 2. Install Dependencies
Open your terminal inside the project directory and install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root folder (this is already ignored by Git for security) and add:
```env
GEMINI_API_KEY=your_gemini_api_key_here
BRATS_DATASET_PATH=C:/path/to/your/brats_dataset  # (Optional)
```
> **Disclaimer:** This system is for research purposes only. AI-generated segmentations and reports have not been independently validated for clinical use and should be reviewed by a qualified radiologist or physician. They must not be used as the sole basis for clinical decisions.

### 4. Run the Streamlit Web Application
Launch the app with Streamlit:
```bash
streamlit run app.py
```
Open the local link (usually `http://localhost:8501`) in your browser to view the application!

### 5. Run Unit Tests (Optional)
Run the test suite to verify the LangGraph workflow and state transitions:
```bash
python -m unittest test_langgraph_workflow.py
```

---

## 🛠️ Project Structure

- `app.py`: Main Streamlit web application and interactive XAI dashboard.
- `langgraph_workflow.py`: LangGraph state machine orchestrating the multi-agent pipeline.
- `agents.py`: Agent implementations (Reception, Scan Analysis, Comparison, and Reporting).
- `model_inference.py`: Runs SegResNet inference and 3D Grad-CAM / Modality Attribution XAI engine.
- `database.py`: Handles patient registration and scan history using PostgreSQL/SQLite.
- `report_generator.py`: Generates printable PDF reports with multi-modal overlays and Grad-CAM visualizations.
- `config.py`: Project configuration and environment variables.
- `test_langgraph_workflow.py`: Unit tests for the LangGraph workflow and agent graph.

---

## 📐 System Architecture

Here is how data flows through the LangGraph multi-agent pipeline:

```mermaid
graph TD
    %% Define styles
    classDef agent fill:#1a237e,stroke:#3949ab,stroke-width:2px,color:#fff;
    classDef database fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;
    classDef ui fill:#e65100,stroke:#ef6c00,stroke-width:2px,color:#fff;
    classDef model fill:#006064,stroke:#00838f,stroke-width:2px,color:#fff;

    %% Elements
    UI["Streamlit UI (app.py)"]:::ui
    DB[("Patient Database")]:::database
    SegResNet["MONAI SegResNet"]:::model
    XAI["Explainable AI Engine (Grad-CAM)"]:::model
    Gemini["Google Gemini AI"]:::model
    PDF["PDF Clinical Report Generator"]:::ui

    subgraph LangGraph ["LangGraph Multi-Agent Pipeline"]
        N1["Reception Node"]:::agent
        N2["Scan Validation Node"]:::agent
        N3["Segmentation & Analysis Node"]:::agent
        N4["Supervisor Node"]:::agent
        N5["Reporting Node"]:::agent
    end

    %% Workflow
    UI -->|Patient Info & MRI Scans| N1
    N1 <-->|Query / Register Patient| DB
    N1 --> N2
    N2 -->|Valid Scans| N3
    N3 <-->|Sliding Window Inference| SegResNet
    N3 <-->|Feature Attribution & Grad-CAM| XAI
    N3 -->|Save Scan & Volume Metrics| DB
    N3 --> N4
    N4 --> N5
    N5 <-->|Generate Medical Draft| Gemini
    N5 -->|State & Results| UI
    UI -->|Compile Report & Visuals| PDF
```
