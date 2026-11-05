# report_generator.py
# This file generates a professional PDF report using ReportLab
# Think of it as the PRINTER of the project!

import io
import base64
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─────────────────────────────────────────
# MAIN FUNCTION — Generate PDF report
# Returns PDF as bytes (so Streamlit can download it)
# ─────────────────────────────────────────
def generate_pdf_report(
    patient_info,
    scan_result,
    comparison=None,
    gemini_report="",
    overlay_b64="",
    predicted_b64="",
    gt_b64="",
    gradcam_b64=""
):
    print("[PDF] Generating PDF report...")

    # Create PDF in memory (not saved to disk)
    buffer = io.BytesIO()

    # Setup document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # ─────────────────────────────────────────
    # STYLES
    # ─────────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=20,
        textColor=colors.HexColor('#1e3a5f'),
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#0d9488'),
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1e3a5f'),
        spaceBefore=12,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
    )

    bold_style = ParagraphStyle(
        'BoldStyle',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.red,
        alignment=TA_CENTER,
        spaceBefore=12,
    )

    # ─────────────────────────────────────────
    # BUILD CONTENT
    # ─────────────────────────────────────────
    content = []

    # ── Section 1: Header ──
    content.append(Paragraph("🧠 Brain Tumor Analysis Report", title_style))
    content.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    content.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor('#1e3a5f'), spaceAfter=12
    ))

    # ── Section 2: Patient Info Table ──
    content.append(Paragraph("Patient Information", section_style))

    patient_data = [
        ["Patient ID", patient_info.get("patient_id", "N/A"),
         "Name", patient_info.get("name", "N/A")],
        ["Age", str(patient_info.get("age", "N/A")),
         "Gender", patient_info.get("gender", "N/A")],
        ["Scan Date", datetime.now().strftime("%Y-%m-%d"),
         "Tumor Detected",
         "YES ⚠️" if scan_result.get("tumor_detected") else "NO ✅"],
    ]

    patient_table = Table(patient_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0fdfa')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0fdfa')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUND', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 12))

    # ── Section 3: Scan Results Table ──
    content.append(Paragraph("Segmentation Measurements", section_style))

    results_data = [
        ["Region", "Volume (cm³)", "Percentage", "Description"],
        ["🔴 NCR (Necrotic Core)",
         str(scan_result.get("ncr_volume", 0)),
         f"{scan_result.get('ncr_percent', 0)}%",
         "Dead tissue at tumor center"],
        ["🟢 ED (Edema)",
         str(scan_result.get("ed_volume", 0)),
         f"{scan_result.get('ed_percent', 0)}%",
         "Swelling around the tumor"],
        ["🔵 ET (Enhancing Tumor)",
         str(scan_result.get("et_volume", 0)),
         f"{scan_result.get('et_percent', 0)}%",
         "Active tumor region"],
        ["⬜ Total Tumor",
         str(scan_result.get("total_volume", 0)),
         "100%",
         "Combined tumor volume"],
    ]

    results_table = Table(
        results_data,
        colWidths=[4.5*cm, 3.5*cm, 3*cm, 6*cm]
    )
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUND', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f5f5f5')]),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#f0fdfa')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    content.append(results_table)
    content.append(Spacer(1, 12))

    # ── Section 4: MRI Images ──
    images_to_show = []

    if gt_b64:
        images_to_show.append(("Ground Truth", gt_b64))
    if predicted_b64:
        images_to_show.append(("Predicted Mask", predicted_b64))
    if overlay_b64:
        images_to_show.append(("MRI + Overlay", overlay_b64))
    if gradcam_b64:
        images_to_show.append(("Grad-CAM Attention", gradcam_b64))

    if images_to_show:
        content.append(Paragraph("MRI Scan Images & Explainability", section_style))

        img_cells = []
        caption_cells = []
        img_size = 4.0 * cm if len(images_to_show) >= 4 else 5.0 * cm
        cell_width = 17.0 * cm / len(images_to_show)

        for caption, b64_str in images_to_show:
            try:
                img_data = base64.b64decode(b64_str)
                img_buf = io.BytesIO(img_data)
                img = Image(img_buf, width=img_size, height=img_size)
                img_cells.append(img)
                caption_cells.append(
                    Paragraph(caption, ParagraphStyle(
                        'caption', fontSize=8,
                        alignment=TA_CENTER
                    ))
                )
            except Exception as e:
                print(f"[PDF] Could not add image: {e}")

        if img_cells:
            img_table = Table(
                [img_cells, caption_cells],
                colWidths=[cell_width] * len(img_cells)
            )
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            content.append(img_table)
            content.append(Spacer(1, 12))

    # ── Section 5: Comparison Table (only if returning patient) ──
    if comparison:
        content.append(Paragraph(
            f"Previous Scan Comparison ({comparison.get('previous_date', 'N/A')})",
            section_style
        ))

        comp_data = [
            ["Region", "Previous (cm³)", "Current (cm³)", "Change (cm³)", "Trend"],
            ["Total",
             str(comparison.get("prev_total_volume", 0)),
             str(scan_result.get("total_volume", 0)),
             str(comparison["total_change"]["absolute"]),
             comparison["total_change"]["trend"]],
            ["NCR",
             str(comparison.get("prev_ncr_volume", 0)),
             str(scan_result.get("ncr_volume", 0)),
             str(comparison["ncr_change"]["absolute"]),
             comparison["ncr_change"]["trend"]],
            ["ED",
             str(comparison.get("prev_ed_volume", 0)),
             str(scan_result.get("ed_volume", 0)),
             str(comparison["ed_change"]["absolute"]),
             comparison["ed_change"]["trend"]],
            ["ET",
             str(comparison.get("prev_et_volume", 0)),
             str(scan_result.get("et_volume", 0)),
             str(comparison["et_change"]["absolute"]),
             comparison["et_change"]["trend"]],
        ]

        comp_table = Table(
            comp_data,
            colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm]
        )
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUND', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(comp_table)
        content.append(Spacer(1, 12))

    # ── Section 6: AI Report Text ──
    if gemini_report:
        content.append(Paragraph("Medical Analysis & Clinical Findings", section_style))
        content.append(HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor('#0d9488'), spaceAfter=8
        ))

        for raw_line in gemini_report.split('\n'):
            line = raw_line.strip()
            # Clean pure separator lines
            if line and len(line) > 3 and set(line).issubset({'-', '=', '*', '_', '#'}):
                continue

            clean_line = line.lstrip('#').strip()
            clean_line = clean_line.replace('**', '').replace('__', '')
            clean_line = clean_line.replace('!!!', '').strip()

            if not clean_line:
                content.append(Spacer(1, 3))
                continue

            # Skip any residual review required banner text if present
            if '[RADIOLOGIST REVIEW REQUIRED]' in clean_line or 'RADIOLOGIST REVIEW REQUIRED' in clean_line:
                continue

            # Make numbered section headers bold
            if clean_line and (clean_line[0].isdigit() and '.' in clean_line[:3]):
                content.append(Spacer(1, 4))
                content.append(Paragraph(clean_line, bold_style))
            else:
                content.append(Paragraph(clean_line, normal_style))

    # ── Section 7: Disclaimer ──
    content.append(Spacer(1, 16))
    content.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#0d9488'), spaceAfter=8
    ))
    content.append(Paragraph(
        "This report contains AI-generated segmentation results for research purposes only. "
        "The results have not been independently validated for clinical use and should be "
        "reviewed by a qualified radiologist or physician. They must not be used as the sole basis for clinical decisions.",
        disclaimer_style
    ))

    # ─────────────────────────────────────────
    # BUILD PDF AND RETURN AS BYTES
    # ─────────────────────────────────────────
    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    print("[PDF] [OK] PDF report generated successfully!")
    return pdf_bytes