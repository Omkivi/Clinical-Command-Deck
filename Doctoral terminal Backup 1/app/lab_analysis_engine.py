import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class LabAnalysisEngine:
    """
    Comprehensive Lab Analysis and Diagnostic Testing Engine
    
    Features:
    - Define lab test panels and normal ranges
    - Analyze uploaded lab reports
    - Extract structured data from reports
    - Flag abnormal values
    - Provide clinical interpretation
    - Analyze radiological reports
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash') if GOOGLE_API_KEY else None
        
        # Comprehensive Lab Test Database
        self.lab_tests = {
            # ===== HEMATOLOGY =====
            "CBC": {
                "name": "Complete Blood Count",
                "category": "Hematology",
                "tests": {
                    "WBC": {"name": "White Blood Cells", "unit": "×10³/μL", "normal_range": [4.5, 11.0], "critical_low": 2.0, "critical_high": 30.0},
                    "RBC": {"name": "Red Blood Cells", "unit": "×10⁶/μL", "normal_range": [4.5, 5.9], "critical_low": 2.5, "critical_high": 7.0},
                    "Hemoglobin": {"name": "Hemoglobin", "unit": "g/dL", "normal_range": [13.5, 17.5], "critical_low": 7.0, "critical_high": 20.0},
                    "Hematocrit": {"name": "Hematocrit", "unit": "%", "normal_range": [38.8, 50.0], "critical_low": 20.0, "critical_high": 60.0},
                    "Platelets": {"name": "Platelets", "unit": "×10³/μL", "normal_range": [150, 400], "critical_low": 50, "critical_high": 1000},
                    "Neutrophils": {"name": "Neutrophils", "unit": "%", "normal_range": [40, 70]},
                    "Lymphocytes": {"name": "Lymphocytes", "unit": "%", "normal_range": [20, 40]},
                }
            },
            
            # ===== METABOLIC PANEL =====
            "CMP": {
                "name": "Comprehensive Metabolic Panel",
                "category": "Chemistry",
                "tests": {
                    "Glucose": {"name": "Glucose", "unit": "mg/dL", "normal_range": [70, 100], "critical_low": 40, "critical_high": 400},
                    "BUN": {"name": "Blood Urea Nitrogen", "unit": "mg/dL", "normal_range": [7, 20], "critical_high": 100},
                    "Creatinine": {"name": "Creatinine", "unit": "mg/dL", "normal_range": [0.7, 1.3], "critical_high": 10.0},
                    "eGFR": {"name": "eGFR", "unit": "mL/min/1.73m²", "normal_range": [90, 120], "critical_low": 15},
                    "Sodium": {"name": "Sodium", "unit": "mEq/L", "normal_range": [136, 145], "critical_low": 120, "critical_high": 160},
                    "Potassium": {"name": "Potassium", "unit": "mEq/L", "normal_range": [3.5, 5.0], "critical_low": 2.5, "critical_high": 6.5},
                    "Chloride": {"name": "Chloride", "unit": "mEq/L", "normal_range": [98, 107]},
                    "CO2": {"name": "CO2", "unit": "mEq/L", "normal_range": [23, 29]},
                    "Calcium": {"name": "Calcium", "unit": "mg/dL", "normal_range": [8.5, 10.5], "critical_low": 6.0, "critical_high": 13.0},
                    "Albumin": {"name": "Albumin", "unit": "g/dL", "normal_range": [3.5, 5.5]},
                    "Total_Protein": {"name": "Total Protein", "unit": "g/dL", "normal_range": [6.0, 8.3]},
                    "ALT": {"name": "ALT", "unit": "U/L", "normal_range": [7, 56], "critical_high": 1000},
                    "AST": {"name": "AST", "unit": "U/L", "normal_range": [10, 40], "critical_high": 1000},
                    "ALP": {"name": "Alkaline Phosphatase", "unit": "U/L", "normal_range": [44, 147]},
                    "Total_Bilirubin": {"name": "Total Bilirubin", "unit": "mg/dL", "normal_range": [0.1, 1.2], "critical_high": 15.0},
                }
            },
            
            # ===== LIPID PANEL =====
            "Lipid_Panel": {
                "name": "Lipid Panel",
                "category": "Chemistry",
                "tests": {
                    "Total_Cholesterol": {"name": "Total Cholesterol", "unit": "mg/dL", "normal_range": [0, 200]},
                    "LDL": {"name": "LDL Cholesterol", "unit": "mg/dL", "normal_range": [0, 100]},
                    "HDL": {"name": "HDL Cholesterol", "unit": "mg/dL", "normal_range": [40, 100]},
                    "Triglycerides": {"name": "Triglycerides", "unit": "mg/dL", "normal_range": [0, 150]},
                    "VLDL": {"name": "VLDL", "unit": "mg/dL", "normal_range": [2, 30]},
                }
            },
            
            # ===== THYROID PANEL =====
            "Thyroid_Panel": {
                "name": "Thyroid Function Tests",
                "category": "Endocrine",
                "tests": {
                    "TSH": {"name": "TSH", "unit": "μIU/mL", "normal_range": [0.4, 4.0], "critical_low": 0.01, "critical_high": 100},
                    "T4": {"name": "Free T4", "unit": "ng/dL", "normal_range": [0.8, 1.8]},
                    "T3": {"name": "Free T3", "unit": "pg/mL", "normal_range": [2.3, 4.2]},
                }
            },
            
            # ===== VITAMINS & MINERALS =====
            "Vitamin_Panel": {
                "name": "Vitamin & Nutrient Panel",
                "category": "Nutritional",
                "tests": {
                    "Vitamin_D": {"name": "Vitamin D, 25-OH", "unit": "ng/mL", "normal_range": [30, 100], "critical_low": 10},
                    "Vitamin_B12": {"name": "Vitamin B12", "unit": "pg/mL", "normal_range": [200, 900], "critical_low": 150},
                    "Folate": {"name": "Folate", "unit": "ng/mL", "normal_range": [2.7, 17.0]},
                    "Iron": {"name": "Iron", "unit": "μg/dL", "normal_range": [60, 170]},
                    "Ferritin": {"name": "Ferritin", "unit": "ng/mL", "normal_range": [24, 336]},
                    "TIBC": {"name": "Total Iron Binding Capacity", "unit": "μg/dL", "normal_range": [250, 450]},
                }
            },
            
            # ===== CARDIAC MARKERS =====
            "Cardiac_Markers": {
                "name": "Cardiac Markers",
                "category": "Cardiac",
                "tests": {
                    "Troponin_I": {"name": "Troponin I", "unit": "ng/mL", "normal_range": [0, 0.04], "critical_high": 0.1},
                    "Troponin_T": {"name": "Troponin T", "unit": "ng/mL", "normal_range": [0, 0.01], "critical_high": 0.1},
                    "CK_MB": {"name": "CK-MB", "unit": "ng/mL", "normal_range": [0, 5], "critical_high": 25},
                    "BNP": {"name": "BNP", "unit": "pg/mL", "normal_range": [0, 100], "critical_high": 400},
                }
            },
            
            # ===== DIABETES =====
            "Diabetes_Panel": {
                "name": "Diabetes Screening",
                "category": "Metabolic",
                "tests": {
                    "HbA1c": {"name": "Hemoglobin A1c", "unit": "%", "normal_range": [4.0, 5.6], "critical_high": 10.0},
                    "Fasting_Glucose": {"name": "Fasting Glucose", "unit": "mg/dL", "normal_range": [70, 100], "critical_low": 40, "critical_high": 400},
                }
            },
            
            # ===== COAGULATION =====
            "Coagulation_Panel": {
                "name": "Coagulation Studies",
                "category": "Hematology",
                "tests": {
                    "PT": {"name": "Prothrombin Time", "unit": "seconds", "normal_range": [11, 13.5]},
                    "INR": {"name": "INR", "unit": "", "normal_range": [0.8, 1.2], "critical_high": 5.0},
                    "aPTT": {"name": "aPTT", "unit": "seconds", "normal_range": [25, 35]},
                }
            },
            
            # ===== URINALYSIS =====
            "Urinalysis": {
                "name": "Urinalysis",
                "category": "Urine",
                "tests": {
                    "pH": {"name": "pH", "unit": "", "normal_range": [5.0, 8.0]},
                    "Specific_Gravity": {"name": "Specific Gravity", "unit": "", "normal_range": [1.005, 1.030]},
                    "Protein": {"name": "Protein", "unit": "mg/dL", "normal_range": [0, 10]},
                    "Glucose": {"name": "Glucose", "unit": "mg/dL", "normal_range": [0, 15]},
                    "Blood": {"name": "Blood", "unit": "", "normal_range": ["Negative", "Negative"]},
                    "WBC": {"name": "WBC", "unit": "/HPF", "normal_range": [0, 5]},
                    "RBC": {"name": "RBC", "unit": "/HPF", "normal_range": [0, 2]},
                }
            },
        }
        
        # Radiological Studies Database
        self.radiology_studies = {
            "Chest_XRay": {
                "name": "Chest X-Ray",
                "modality": "X-Ray",
                "common_findings": ["Pneumonia", "Cardiomegaly", "Pleural Effusion", "Pneumothorax", "Nodules", "Infiltrates"]
            },
            "CT_Head": {
                "name": "CT Head (Non-Contrast)",
                "modality": "CT",
                "common_findings": ["Hemorrhage", "Stroke", "Mass", "Hydrocephalus", "Fracture", "Midline Shift"]
            },
            "CT_Chest": {
                "name": "CT Chest",
                "modality": "CT",
                "common_findings": ["Pulmonary Embolism", "Pneumonia", "COPD", "Interstitial Lung Disease", "Masses"]
            },
            "CT_Abdomen_Pelvis": {
                "name": "CT Abdomen/Pelvis",
                "modality": "CT",
                "common_findings": ["Appendicitis", "Bowel Obstruction", "Renal Stones", "Abscess", "Free Fluid"]
            },
            "MRI_Brain": {
                "name": "MRI Brain",
                "modality": "MRI",
                "common_findings": ["Stroke", "Multiple Sclerosis", "Tumor", "Aneurysm", "Demyelination"]
            },
            "Ultrasound_Abdomen": {
                "name": "Ultrasound Abdomen",
                "modality": "Ultrasound",
                "common_findings": ["Gallstones", "Fatty Liver", "Hepatomegaly", "Splenomegaly", "Ascites"]
            },
            "Echo": {
                "name": "Echocardiogram",
                "modality": "Ultrasound",
                "common_findings": ["Ejection Fraction", "Valvular Disease", "Wall Motion Abnormality", "Pericardial Effusion"]
            },
        }
    
    def analyze_lab_report(self, report_text, patient_id=None):
        """
        Analyze uploaded lab report text and extract structured data
        Uses AI to parse unstructured reports
        """
        if not self.model:
            return self._fallback_lab_analysis(report_text)
        
        try:
            prompt = f"""
You are a clinical laboratory data extraction specialist.

TASK: Extract all lab values from the following lab report and structure them in JSON format.

LAB REPORT:
{report_text}

OUTPUT FORMAT (JSON only, no markdown):
{{
    "test_date": "YYYY-MM-DD or Unknown",
    "patient_name": "Name if found or Unknown",
    "tests": [
        {{
            "test_name": "Test Name",
            "value": numeric_value_or_string,
            "unit": "unit",
            "normal_range": "range if mentioned",
            "flag": "High/Low/Critical/Normal"
        }}
    ],
    "overall_summary": "Brief 1-2 sentence summary of key findings"
}}

Extract ALL numeric values. If a value is abnormal, mark it with the appropriate flag.
Return ONLY valid JSON with no additional text.
"""
            
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Clean up markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            # Add interpretation
            data['interpretation'] = self._interpret_lab_results(data.get('tests', []))
            data['analyzed_at'] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            print(f"AI Lab Analysis Error: {e}")
            return self._fallback_lab_analysis(report_text)
    
    def _interpret_lab_results(self, tests):
        """Provide clinical interpretation of lab results"""
        interpretations = []
        critical_findings = []
        abnormal_findings = []
        
        for test in tests:
            flag = test.get('flag', 'Normal')
            test_name = test.get('test_name', '')
            value = test.get('value', '')
            
            if flag == 'Critical':
                critical_findings.append(f"{test_name}: {value} (CRITICAL)")
            elif flag in ['High', 'Low']:
                abnormal_findings.append(f"{test_name}: {value} ({flag})")
        
        if critical_findings:
            interpretations.append(f"⚠️ CRITICAL: {', '.join(critical_findings)}")
        if abnormal_findings:
            interpretations.append(f"Abnormal: {', '.join(abnormal_findings)}")
        if not critical_findings and not abnormal_findings:
            interpretations.append("All values within normal limits")
        
        return " | ".join(interpretations)
    
    def _fallback_lab_analysis(self, report_text):
        """Fallback parser for when AI is unavailable"""
        return {
            "test_date": "Unknown",
            "patient_name": "Unknown",
            "tests": [],
            "overall_summary": "AI analysis unavailable. Manual review required.",
            "raw_text": report_text,
            "interpretation": "Please review manually",
            "analyzed_at": datetime.now().isoformat()
        }
    
    def analyze_radiology_report(self, report_text, study_type="Unknown", patient=None):
        """
        Analyze radiological report and extract comprehensive, clinically actionable findings
        with patient-contextualized interpretation
        """
        if not self.model:
            return self._fallback_radiology_analysis(report_text, study_type)
        
        # Build patient context if available
        patient_context = ""
        if patient:
            patient_context = f"""
PATIENT CONTEXT (Use this for clinical correlation):
- Age: {patient.get('age', 'Unknown')} years
- Sex: {patient.get('sex', 'Unknown')}
- Weight: {patient.get('weight', 'Unknown')} kg
- Known Conditions: {', '.join(patient.get('medical_conditions', [])) or 'None documented'}
- Current Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])]) or 'None documented'}
- Allergies: {', '.join(patient.get('allergies', [])) or 'None documented'}
- Recent Symptoms: {', '.join(patient.get('symptoms', [])) if patient.get('symptoms') else 'Not specified'}
"""
        
        try:
            prompt = f"""
You are a board-certified radiologist providing a COMPREHENSIVE, DETAILED analysis of an imaging study.

STUDY TYPE: {study_type}
{patient_context}
REPORT/IMAGE DESCRIPTION:
{report_text}

TASK: Provide an exhaustive, clinically actionable radiology interpretation.

OUTPUT FORMAT (JSON only):
{{
    "study_type": "{study_type}",
    "modality": "X-ray/CT/MRI/Ultrasound/etc",
    "study_date": "Date if found or Unknown",
    "indication": "Clinical indication/reason for study",
    "technique": "Detailed imaging technique/protocol used",
    
    "anatomical_coverage": "Body regions included in this study",
    
    "detailed_findings": [
        {{
            "anatomical_region": "Specific anatomical area",
            "finding": "Detailed description of finding",
            "location": "Precise anatomical location (e.g., 'right lower lobe, posterior basal segment')",
            "size_measurement": "Measurements if applicable (e.g., '2.3 x 1.8 cm')",
            "characteristics": "Density/signal/echogenicity/enhancement pattern",
            "severity": "Normal/Mild/Moderate/Severe",
            "clinical_significance": "High/Medium/Low/Incidental",
            "change_from_prior": "New/Stable/Improved/Worsened/No prior for comparison"
        }}
    ],
    
    "normal_structures": ["List of anatomical structures that appear normal"],
    
    "impression": "Radiologist's overall impression - numbered list of key conclusions",
    
    "critical_findings": [
        {{
            "finding": "Critical finding description",
            "urgency": "Immediate/Urgent/Soon",
            "recommended_action": "Specific action required"
        }}
    ],
    
    "differential_diagnosis": [
        {{
            "diagnosis": "Possible diagnosis",
            "likelihood": "High/Moderate/Low",
            "supporting_features": "Imaging features supporting this diagnosis",
            "against_features": "Features arguing against this diagnosis"
        }}
    ],
    
    "patient_specific_correlation": "How these findings correlate with the patient's known conditions, medications, and symptoms",
    
    "comparison_with_prior": "Comparison with any prior studies if mentioned",
    
    "recommendations": [
        {{
            "recommendation": "Specific recommendation",
            "priority": "Immediate/Routine/Optional",
            "rationale": "Why this is recommended"
        }}
    ],
    
    "follow_up_imaging": "Recommended follow-up imaging if any (type, timing)",
    
    "limitations": "Any limitations of this study (motion artifact, incomplete coverage, etc)",
    
    "quality_assessment": "Diagnostic quality: Excellent/Good/Adequate/Limited/Non-diagnostic",
    
    "clinical_summary": "2-3 sentence summary written for the referring clinician"
}}

Be thorough and specific. Include measurements, locations, and clinical correlations.
Return ONLY valid JSON.
"""
            
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Clean up markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            data['analyzed_at'] = datetime.now().isoformat()
            data['patient_context_applied'] = patient is not None
            
            return data
            
        except Exception as e:
            print(f"AI Radiology Analysis Error: {e}")
            return self._fallback_radiology_analysis(report_text, study_type)
    
    def _fallback_radiology_analysis(self, report_text, study_type):
        """Fallback for radiology analysis"""
        return {
            "study_type": study_type,
            "study_date": "Unknown",
            "indication": "Unknown",
            "technique": "Unknown",
            "findings": [],
            "impression": "AI analysis unavailable",
            "critical_findings": [],
            "recommendations": [],
            "raw_text": report_text,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def get_test_info(self, test_key):
        """Get information about a specific lab test"""
        for panel_key, panel in self.lab_tests.items():
            if test_key in panel['tests']:
                return panel['tests'][test_key]
        return None
    
    def check_value_status(self, test_key, value):
        """Check if a lab value is normal, abnormal, or critical"""
        test_info = self.get_test_info(test_key)
        if not test_info:
            return "Unknown"
        if "u/L" in unit or "U/L" in unit:
            return "U/L"nknown"
        
        try:
            numeric_value = float(value)
            normal_range = test_info.get('normal_range', [])
            
            if len(normal_range) == 2:
                if numeric_value < normal_range[0]:
                    if 'critical_low' in test_info and numeric_value <= test_info['critical_low']:
                        return "Critical Low"
                    return "Low"
                elif numeric_value > normal_range[1]:
                    if 'critical_high' in test_info and numeric_value >= test_info['critical_high']:
                        return "Critical High"
                    return "High"
                else:
                    return "Normal"
        except:
            return "Unknown"
        
        return "Normal"
    
    def analyze_lab_report_with_council(self, report_text, patient_id=None, patient=None):
        """
        Analyze lab report with MULTI-MODEL AI COUNCIL deliberation.
        
        Uses 4 different AI models to independently interpret lab results
        and debate clinical significance.
        
        Returns:
            Dict with lab analysis AND council deliberation
        """
        # Step 1: Run base lab analysis
        base_analysis = self.analyze_lab_report(report_text, patient_id)
        
        # Step 2: Get AI Council deliberation on findings
        try:
            from .ai_council import ai_council
            
            # Extract key abnormal values for council
            abnormal_values = []
            for test in base_analysis.get('tests', []):
                if test.get('flag') in ['High', 'Low', 'Critical']:
                    abnormal_values.append(f"{test.get('test_name')}: {test.get('value')} ({test.get('flag')})")
            
            case_data = {
                "report_type": "LAB_REPORT",
                "findings": base_analysis.get('overall_summary', ''),
                "abnormal_values": abnormal_values,
                "test_count": len(base_analysis.get('tests', [])),
                "interpretation": base_analysis.get('interpretation', '')
            }
            
            # Run council deliberation
            council_verdict = ai_council.deliberate(
                case_type="REPORT_ANALYSIS",
                case_data=case_data,
                patient=patient
            )
            
            base_analysis["council_deliberation"] = council_verdict.to_dict()
            base_analysis["council_verdict"] = council_verdict.final_verdict
            base_analysis["council_consensus"] = council_verdict.consensus_score
            
        except Exception as e:
            print(f"Council deliberation error in lab analysis: {e}")
            base_analysis["council_deliberation"] = None
        
        return base_analysis
    
    def analyze_radiology_report_with_council(self, report_text, study_type="Unknown", patient=None):
        """
        Analyze radiology report with MULTI-MODEL AI COUNCIL deliberation.
        
        Uses 4 different AI models to independently interpret imaging findings
        and debate clinical significance.
        
        Returns:
            Dict with radiology analysis AND council deliberation
        """
        # Step 1: Run base radiology analysis with patient context
        base_analysis = self.analyze_radiology_report(report_text, study_type, patient)
        
        # Step 2: Get AI Council deliberation on findings
        try:
            from .ai_council import ai_council
            
            case_data = {
                "report_type": f"RADIOLOGY_{study_type.upper()}",
                "findings": ", ".join(base_analysis.get('findings', [])),
                "impression": base_analysis.get('impression', ''),
                "critical_findings": base_analysis.get('critical_findings', []),
                "abnormal_values": base_analysis.get('critical_findings', [])
            }
            
            # Run council deliberation
            council_verdict = ai_council.deliberate(
                case_type="REPORT_ANALYSIS",
                case_data=case_data,
                patient=patient
            )
            
            base_analysis["council_deliberation"] = council_verdict.to_dict()
            base_analysis["council_verdict"] = council_verdict.final_verdict
            base_analysis["council_consensus"] = council_verdict.consensus_score
            
        except Exception as e:
            print(f"Council deliberation error in radiology analysis: {e}")
            base_analysis["council_deliberation"] = None
        
        return base_analysis

# Initialize the engine
lab_engine = LabAnalysisEngine()
