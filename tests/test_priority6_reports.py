"""Priority 6 verification tests — Clinical Report Generation."""
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

print("=" * 60)
print("NeuroSynth v2 — Priority 6 Clinical Reports Verification")
print("=" * 60)

# 1. SOAP Report Generation
print("\n[1/5] SOAP-Structured Report Generation...")
from backend.report_generator_v2 import ClinicalReportGeneratorV2

gen = ClinicalReportGeneratorV2()
report = gen.generate_report(
    patient_data={
        "Age": 73, "Gender": 1, "MMSE": 22, "FunctionalAssessment": 5.5,
        "ADL": 6.0, "SleepQuality": 4, "PhysicalActivity": 3,
        "SystolicBP": 145, "DiastolicBP": 88, "BMI": 27.3,
        "CholesterolTotal": 220, "CholesterolLDL": 140, "CholesterolHDL": 45,
        "MemoryComplaints": 1, "BehavioralProblems": 1, "Confusion": 1,
        "Disorientation": 0, "FamilyHistoryAlzheimers": 1,
    },
    prediction={"probability": 0.72, "risk_level": "High", "confidence": "High", "prediction": 1},
    trajectory=[0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88],
    causal_graph={"top_causes_of_Diagnosis": [{"variable": "MMSE", "strength": 0.8}]},
    shap_values=[
        {"feature": "MMSE", "value": -0.18}, {"feature": "FunctionalAssessment", "value": 0.12},
        {"feature": "ADL", "value": -0.09}, {"feature": "Age", "value": 0.07},
        {"feature": "SleepQuality", "value": 0.05},
    ],
    patient_id="P-TEST-001",
    disease="Alzheimer's Disease",
)
assert report["format"] == "SOAP"
assert "subjective" in report["soap"]
assert "objective" in report["soap"]
assert "assessment" in report["soap"]
assert "plan" in report["soap"]
assert len(report["sections"]) == 4
print(f"  Format:      {report['format']}")
print(f"  Sections:    {list(report['sections'].keys())}")
print(f"  Word count:  {report['word_count']}")
print(f"  Report ID:   {report['report_id']}")
print("  PASSED")

# 2. ICD-10 Code Suggestions
print("\n[2/5] ICD-10 Code Suggestions...")
icd_codes = report["icd10_codes"]
assert len(icd_codes) > 0
print(f"  Codes suggested: {len(icd_codes)}")
for icd in icd_codes:
    print(f"    {icd['code']:>8}: {icd['description']:<55} (conf={icd['confidence']:.1%})")
assert icd_codes[0]["code"] == "G30.9", f"Expected G30.9, got {icd_codes[0]['code']}"
assert all(0 < c["confidence"] <= 1.0 for c in icd_codes)
print("  PASSED")

# 3. FHIR R4 DiagnosticReport
print("\n[3/5] FHIR R4 DiagnosticReport...")
fhir = gen.to_fhir(report)
assert fhir["resourceType"] == "DiagnosticReport"
assert fhir["status"] == "final"
assert "Patient/P-TEST-001" in fhir["subject"]["reference"]
assert len(fhir["conclusionCode"]) > 0
coding = fhir["conclusionCode"][0]["coding"]
assert coding[0]["system"] == "http://hl7.org/fhir/sid/icd-10-cm"
print(f"  Resource:       {fhir['resourceType']}")
print(f"  Status:         {fhir['status']}")
print(f"  Subject:        {fhir['subject']['reference']}")
print(f"  ICD-10 codes:   {len(coding)}")
print(f"  Conclusion:     {fhir['conclusion'][:80]}...")
print("  PASSED")

# 4. PDF Export
print("\n[4/5] PDF Export...")
pdf_bytes = gen.to_pdf(report)
assert isinstance(pdf_bytes, bytes)
assert len(pdf_bytes) > 100
print(f"  PDF size:    {len(pdf_bytes):,} bytes")
print(f"  Starts with: {pdf_bytes[:5]}")
print("  PASSED")

# 5. HTML Template Rendering
print("\n[5/5] HTML Template Rendering...")
html = report["html"]
assert "<!DOCTYPE html>" in html
assert "NeuroSynth Clinical Intelligence Report" in html
assert "P-TEST-001" in html
assert "SOAP" not in html or "Subjective" in html  # Template renders SOAP sections
assert "G30.9" in html  # ICD code present
assert "72.0%" in html  # Risk score
print(f"  HTML size:   {len(html):,} chars")
print(f"  Contains patient ID: {'P-TEST-001' in html}")
print(f"  Contains ICD codes:  {'G30.9' in html}")
print(f"  Contains risk score: {'72.0%' in html}")
print("  PASSED")

# 6. v2 Router Registration
print("\n[BONUS] v2 Reports Router...")
from backend.routers.reports_v2 import router as v2_router
routes = [r.path for r in v2_router.routes]
print(f"  Router prefix: {v2_router.prefix}")
print(f"  Routes: {routes}")
assert any("generate" in r for r in routes)
assert any("fhir" in r for r in routes)
assert any("pdf" in r for r in routes)
print("  PASSED")

print("\n" + "=" * 60)
print("ALL PRIORITY 6 VERIFICATION TESTS PASSED")
print("=" * 60)
