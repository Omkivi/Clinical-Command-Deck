"""
Clinical Note AI Assistant

Auto-generates clinical documentation:
1. Progress notes (SOAP format)
2. Discharge summaries
3. H&P (History & Physical)
4. Procedure notes
5. Billing code suggestions (ICD-10, CPT)
6. Quality metrics documentation
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class ClinicalNoteAssistant:
    """
    AI-powered clinical documentation assistant.
    """
    
    def __init__(self):
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
    
    def generate_progress_note(self, patient: Dict, encounter_data: Dict) -> Dict:
        """
        Generate SOAP note (Subjective, Objective, Assessment, Plan)
        
        Args:
            patient: Patient data
            encounter_data: {
                'chief_complaint': str,
                'hpi': str (history of present illness),
                'vitals': dict,
                'physical_exam': str,
                'labs_today': list,
                'imaging_today': list,
                'assessment': str,
                'plan': str
            }
        """
        if not self.model:
            return self._fallback_progress_note()
        
        try:
            prompt = f"""
Generate a professional progress note in SOAP format.

PATIENT: {patient.get('name', 'Unknown')}, {patient.get('age', 'Unknown')}yo {patient.get('sex', 'Unknown')}

ENCOUNTER DATA:
- Chief Complaint: {encounter_data.get('chief_complaint', 'Not provided')}
- HPI: {encounter_data.get('hpi', 'Patient reports ongoing symptoms')}
- Vitals: {encounter_data.get('vitals', {})}
- Physical Exam: {encounter_data.get('physical_exam', 'Deferred')}
- Labs: {encounter_data.get('labs_today', [])}
- Imaging: {encounter_data.get('imaging_today', [])}
- Medical History: {', '.join(patient.get('conditions', []))}
- Current Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])])}

Generate complete SOAP note with:

SUBJECTIVE:
- Chief complaint
- HPI
- Review of systems (pertinent positives/negatives)

OBJECTIVE:
- Vitals
- Physical exam findings
- Lab results
- Imaging results

ASSESSMENT:
- Problem list with ICD-10 codes
- Clinical reasoning

PLAN:
- For each problem
- Medications
- Follow-up

Include appropriate ICD-10 and CPT codes.

Return ONLY valid JSON:
{{
    "note_type": "Progress Note",
    "date": "{datetime.now().strftime('%Y-%m-%d')}",
    "subjective": "Full subjective section",
    "objective": "Full objective section",
    "assessment": "Assessment with problem list",
    "plan": "Detailed plan",
    "icd10_codes": ["Code1: Description", "Code2: Description"],
    "cpt_codes": ["99213: Office visit", "etc"],
    "billing_level": "99213|99214|99215",
    "full_note_text": "Complete formatted note"
}}
"""
            
            response = self.model.generate_content(prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            note = json.loads(text)
            note['generated_at'] = datetime.now().isoformat()
            note['provider'] = "AI Assistant (requires physician review)"
            
            return note
            
        except Exception as e:
            print(f"Progress note generation error: {e}")
            return self._fallback_progress_note()
    
    def generate_discharge_summary(self, patient: Dict, admission_data: Dict) -> Dict:
        """
        Generate comprehensive discharge summary
        
        Args:
            admission_data: {
                'admission_date': str,
                'discharge_date': str,
                'chief_complaint': str,
                'hospital_course': str,
                'procedures': list,
                'discharge_condition': str,
                'discharge_medications': list,
                'follow_up': str
            }
        """
        if not self.model:
            return self._fallback_discharge_summary()
        
        try:
            prompt = f"""
Generate a comprehensive discharge summary.

PATIENT: {patient.get('name')}, {patient.get('age')}yo {patient.get('sex')}

ADMISSION DETAILS:
- Admitted: {admission_data.get('admission_date', 'Unknown')}
- Discharged: {admission_data.get('discharge_date', datetime.now().strftime('%Y-%m-%d'))}
- Chief Complaint: {admission_data.get('chief_complaint', 'Unknown')}
- Past Medical History: {', '.join(patient.get('conditions', []))}
- Home Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])])}

HOSPITAL COURSE:
{admission_data.get('hospital_course', 'Patient admitted and treated')}

PROCEDURES:
{', '.join(admission_data.get('procedures', []))}

Generate complete discharge summary with:

1. ADMISSION DIAGNOSIS
2. DISCHARGE DIAGNOSIS
3. HOSPITAL COURSE (by problem)
4. PROCEDURES PERFORMED
5. DISCHARGE MEDICATIONS (with reconciliation)
6. DISCHARGE CONDITION
7. DISCHARGE INSTRUCTIONS
8. FOLLOW-UP APPOINTMENTS
9. ICD-10 CODES (all diagnoses)
10. DRG CODE (if applicable)

Return JSON:
{{
    "admission_diagnosis": [],
    "discharge_diagnosis": [],
    "hospital_course": "Narrative",
    "procedures": [],
    "discharge_medications": [],
    "discharge_condition": "Stable|Improved|etc",
    "patient_instructions": [],
    "follow_up": [],
    "icd10_codes": [],
    "drg_code": "if applicable",
    "full_summary_text": "Complete formatted summary"
}}
"""
            
            response = self.model.generate_content(prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            summary = json.loads(text)
            summary['generated_at'] = datetime.now().isoformat()
            
            return summary
            
        except Exception as e:
            print(f"Discharge summary error: {e}")
            return self._fallback_discharge_summary()
    
    def suggest_billing_codes(self, encounter_type: str, diagnoses: List[str], procedures: List[str] = None) -> Dict:
        """
        Suggest appropriate ICD-10 and CPT codes
        
        Args:
            encounter_type: 'office_visit', 'hospital', 'procedure', 'telehealth'
            diagnoses: List of diagnosis descriptions
            procedures: Optional list of procedures performed
        """
        if not self.model:
            return self._fallback_billing_codes()
        
        try:
            prompt = f"""
Medical coding expert needed.

Encounter Type: {encounter_type}
Diagnoses: {', '.join(diagnoses)}
Procedures: {', '.join(procedures) if procedures else 'None'}

Provide:
1. ICD-10-CM codes for all diagnoses (with descriptions)
2. CPT codes for procedures/visit level
3. Appropriate E/M code based on complexity
4. Modifiers if needed
5. Documentation tips for billing compliance

Return JSON:
{{
    "icd10_codes": [
        {{"code": "I10", "description": "Essential hypertension"}},
        ...
    ],
    "cpt_codes": [
        {{"code": "99214", "description": "Office visit level 4"}},
        ...
    ],
    "em_level": "99213|99214|99215|etc",
    "em_justification": "Why this level",
    "modifiers": [],
    "documentation_tips": []
}}
"""
            
            response = self.model.generate_content(prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            codes = json.loads(text)
            return codes
            
        except Exception as e:
            print(f"Billing code suggestion error: {e}")
            return self._fallback_billing_codes()
    
    def generate_history_and_physical(self, patient: Dict, chief_complaint: str, hpi: str, ros: Dict, physical_exam: Dict) -> Dict:
        """
        Generate admission H&P (History and Physical)
        """
        if not self.model:
            return {"error": "H&P generation unavailable"}
        
        try:
            prompt = f"""
Generate a complete admission History & Physical.

PATIENT: {patient.get('name')}, {patient.get('age')}yo {patient.get('sex')}

CC: {chief_complaint}
HPI: {hpi}
PMH: {', '.join(patient.get('conditions', []))}
Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])])}
Allergies: {', '.join(patient.get('allergies', []))}
Social History: {patient.get('social_history', 'Not documented')}
Family History: {patient.get('family_history', 'Non-contributory')}

ROS: {ros}
Physical Exam: {physical_exam}

Format as complete H&P with all standard sections.

Return formatted text.
"""
            
            response = self.model.generate_content(prompt)
            return {
                "h_and_p": response.text,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _fallback_progress_note(self) -> Dict:
        """Fallback progress note template"""
        return {
            "note_type": "Progress Note",
            "error": "AI generation unavailable",
            "template": """
SUBJECTIVE:
[Chief complaint]
[HPI]

OBJECTIVE:
Vitals: [VS]
Physical Exam: [PE]
Labs: [Labs]

ASSESSMENT:
1. [Problem 1] - ICD10: [code]
2. [Problem 2] - ICD10: [code]

PLAN:
1. [Plan for problem 1]
2. [Plan for problem 2]
"""
        }
    
    def _fallback_discharge_summary(self) -> Dict:
        """Fallback discharge summary"""
        return {
            "error": "AI generation unavailable",
            "template": "Standard discharge summary template"
        }
    
    def _fallback_billing_codes(self) -> Dict:
        """Fallback billing codes"""
        return {
            "error": "Billing code suggestions unavailable",
            "recommendation": "Consult coding specialist or ICD-10/CPT references"
        }


# Singleton instance
note_assistant = ClinicalNoteAssistant()
