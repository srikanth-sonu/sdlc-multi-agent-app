"""Generate open healthcare domain PDFs for RAG."""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "Domain_documents"
OUT.mkdir(parents=True, exist_ok=True)

DOCS = {
    "hipaa_privacy_overview.pdf": [
        "HIPAA Privacy Rule Overview (Healthcare Domain Knowledge)",
        "",
        "Protected Health Information (PHI) includes names, addresses, dates,",
        "medical record numbers, and any data that can identify a patient.",
        "",
        "Covered entities must: obtain patient consent for uses beyond treatment,",
        "payment, and operations; implement administrative, physical, and technical",
        "safeguards; provide Notice of Privacy Practices; and report breaches.",
        "",
        "Minimum Necessary Rule: only the minimum PHI needed for a task should",
        "be accessed or disclosed. Access controls and audit logs are required.",
        "",
        "Patients have rights to access records, request amendments, and receive",
        "an accounting of disclosures. Systems must support these workflows.",
    ],
    "ehr_best_practices.pdf": [
        "Electronic Health Record (EHR) Best Practices",
        "",
        "EHR systems should support: patient demographics, problem lists,",
        "medication lists, allergies, vitals, clinical notes, lab results,",
        "imaging reports, care plans, and appointment scheduling.",
        "",
        "Interoperability: use FHIR R4 resources (Patient, Encounter, Observation,",
        "MedicationRequest) for exchange. Prefer REST APIs with OAuth2.",
        "",
        "Clinical decision support should alert on drug interactions and allergies",
        "without alert fatigue. Keep workflows clinician-friendly.",
        "",
        "Availability target for clinical systems is typically 99.9%+.",
        "Backup and disaster recovery must be tested regularly.",
    ],
    "clinical_workflow.pdf": [
        "Clinical Workflow and Patient Safety",
        "",
        "Typical outpatient flow: registration -> triage/vitals -> clinician",
        "encounter -> orders (labs/meds/imaging) -> checkout -> follow-up.",
        "",
        "Inpatient flow: admission -> assessments -> care plan -> medication",
        "administration (eMAR) -> discharge summary -> referrals.",
        "",
        "Patient safety: unique patient identifiers, barcode med admin,",
        "allergy checks, fall risk scoring, and sepsis early warning scores.",
        "",
        "Documentation must be contemporaneous, attributable, and legible.",
        "Audit trails must record who viewed or changed clinical data.",
    ],
}


def write_pdf(path: Path, lines: list[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    for line in lines:
        if y < 72:
            c.showPage()
            y = height - 72
        c.setFont("Helvetica-Bold" if line and not line.startswith(" ") and y > height - 100 else "Helvetica", 11)
        if lines.index(line) == 0:
            c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, line[:95])
        y -= 16
    c.save()


def main() -> None:
    for name, lines in DOCS.items():
        write_pdf(OUT / name, lines)
        print(f"Wrote {OUT / name}")


if __name__ == "__main__":
    main()
