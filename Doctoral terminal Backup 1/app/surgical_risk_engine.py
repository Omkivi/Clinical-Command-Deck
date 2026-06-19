"""
Surgical Risk Assessment Engine

Calculates pre-operative risk scores:
1. ASA Physical Status Classification
2. Revised Cardiac Risk Index (RCRI)
3. NSQIP Risk Calculator
4. Post-op complication prediction
5. ICU admission probability
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class SurgicalRiskEngine:
    """
    Comprehensive surgical risk assessment using validated scoring systems.
    """
    
    def __init__(self):
        # Initialize Gemini for complex risk assessments
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
    
    def calculate_asa_score(self, patient: Dict) -> Dict:
        """
        Calculate ASA Physical Status Classification (1-6)
        
        ASA 1: Healthy
        ASA 2: Mild systemic disease
        ASA 3: Severe systemic disease
        ASA 4: Severe disease, constant threat to life
        ASA 5: Moribund, not expected to survive without operation
        ASA 6: Brain-dead organ donor
        """
        age = patient.get('age', 0)
        conditions = patient.get('conditions', [])
        current_meds = patient.get('current_medications', [])
        
        # Simple rule-based ASA classification
        asa_score = 1
        asa_description = "Healthy patient"
        
        # Check for conditions
        mild_conditions = ['hypertension controlled', 'mild asthma', 'obesity bmi<35']
        severe_conditions = ['heart failure', 'copd severe', 'diabetes uncontrolled', 'renal failure']
        critical_conditions = ['unstable angina', 'sepsis', 'shock', 'acute MI']
        
        has_mild = any(cond.lower() in str(conditions).lower() for cond in mild_conditions)
        has_severe = any(cond.lower() in str(conditions).lower() for cond in severe_conditions)
        has_critical = any(cond.lower() in str(conditions).lower() for cond in critical_conditions)
        
        if has_critical:
            asa_score = 4
            asa_description = "Severe systemic disease that is a constant threat to life"
        elif has_severe or len(current_meds) > 5:
            asa_score = 3
            asa_description = "Severe systemic disease"
        elif has_mild or len(current_meds) > 0 or age > 65:
            asa_score = 2
            asa_description = "Mild systemic disease"
        
        return {
            "asa_score": asa_score,
            "asa_description": asa_description,
            "mortality_risk": self._asa_mortality_risk(asa_score),
            "anesthesia_considerations": self._asa_anesthesia_notes(asa_score)
        }
    
    def calculate_rcri(self, patient: Dict, surgery_type: str) -> Dict:
        """
        Revised Cardiac Risk Index for non-cardiac surgery
        
        Risk factors (1 point each):
        1. High-risk surgery
        2. History of ischemic heart disease
        3. History of congestive heart failure
        4. History of cerebrovascular disease
        5. Diabetes on insulin
        6. Creatinine > 2.0 mg/dL
        """
        points = 0
        risk_factors = []
        
        # High-risk surgery types
        high_risk_surgeries = ['vascular', 'major orthopedic', 'intraperitoneal', 'intrathoracic']
        if any(surg in surgery_type.lower() for surg in high_risk_surgeries):
            points += 1
            risk_factors.append("High-risk surgery type")
        
        # Check conditions
        conditions_str = str(patient.get('conditions', [])).lower()
        history = str(patient.get('history', [])).lower()
        
        if 'ischemic heart' in conditions_str or 'coronary artery disease' in conditions_str or 'mi' in history:
            points += 1
            risk_factors.append("History of ischemic heart disease")
        
        if 'heart failure' in conditions_str or 'chf' in conditions_str:
            points += 1
            risk_factors.append("History of congestive heart failure")
        
        if 'stroke' in history or 'tia' in history or 'cerebrovascular' in conditions_str:
            points += 1
            risk_factors.append("History of cerebrovascular disease")
        
        # Check medications for insulin
        meds = [m.get('name', '').lower() for m in patient.get('current_medications', [])]
        if any('insulin' in med for med in meds):
            points += 1
            risk_factors.append("Diabetes requiring insulin")
        
        # Check labs for creatinine
        lab_reports = patient.get('lab_reports', [])
        for report in lab_reports:
            findings = report.get('findings', {})
            creatinine = findings.get('creatinine', {}).get('value', 0)
            if isinstance(creatinine, (int, float)) and creatinine > 2.0:
                points += 1
                risk_factors.append("Creatinine > 2.0 mg/dL")
                break
        
        # Calculate risk
        if points == 0:
            cardiac_event_risk = 0.4
            risk_category = "Very Low"
        elif points == 1:
            cardiac_event_risk = 1.0
            risk_category = "Low"
        elif points == 2:
            cardiac_event_risk = 2.4
            risk_category = "Moderate"
        else:  # >= 3
            cardiac_event_risk = 5.4
            risk_category = "High"
        
        return {
            "rcri_score": points,
            "risk_factors": risk_factors,
            "cardiac_event_risk_percent": cardiac_event_risk,
            "risk_category": risk_category,
            "recommendations": self._rcri_recommendations(points, risk_category)
        }
    
    def calculate_nsqip_risk(self, patient: Dict, surgery_type: str, surgery_duration_hours: float = 2.0) -> Dict:
        """
        Simplified NSQIP (National Surgical Quality Improvement Program) Risk Calculator
        
        Predicts:
        - Any complication
        - Serious complication
        - Pneumonia
        - Cardiac complication
        - Surgical site infection
        - UTI
        - VTE
        - Renal failure
        - Readmission
        - Death
        """
        # This is a simplified version - actual NSQIP uses complex regression models
        # We'll use AI to estimate based on patient factors
        
        if not self.model:
            return self._fallback_nsqip(patient, surgery_type)
        
        try:
            patient_summary = f"""
Age: {patient.get('age', 'Unknown')}
ASA Score: {self.calculate_asa_score(patient)['asa_score']}
Conditions: {', '.join(patient.get('conditions', []))}
Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])])}
Functional Status: {patient.get('functional_status', 'Independent')}
Surgery: {surgery_type}
Duration: {surgery_duration_hours} hours
"""
            
            prompt = f"""
You are a surgical outcomes expert using NSQIP data.

Patient Profile:
{patient_summary}

Estimate the percentage risk for:
1. any_complication
2. serious_complication
3. pneumonia
4. cardiac_complication
5. surgical_site_infection
6. uti
7. vte (venous thromboembolism)
8. acute_kidney_injury
9. readmission_30day
10. death_30day

Return ONLY valid JSON:
{{
    "any_complication": <percent 0-100>,
    "serious_complication": <percent 0-100>,
    "pneumonia": <percent 0-100>,
    "cardiac_complication": <percent 0-100>,
    "surgical_site_infection": <percent 0-100>,
    "uti": <percent 0-100>,
    "vte": <percent 0-100>,
    "acute_kidney_injury": <percent 0-100>,
    "readmission_30day": <percent 0-100>,
    "death_30day": <percent 0-100>,
    "overall_risk_category": "Low|Moderate|High",
    "key_risk_factors": ["factor1", "factor2"],
    "recommendations": "Brief clinical recommendations"
}}
"""
            
            response = self.model.generate_content(prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            risks = json.loads(text)
            return risks
            
        except Exception as e:
            print(f"NSQIP calculation error: {e}")
            return self._fallback_nsqip(patient, surgery_type)
    
    def predict_icu_admission(self, patient: Dict, surgery_type: str) -> Dict:
        """
        Predict probability of ICU admission after surgery
        """
        asa_score = self.calculate_asa_score(patient)['asa_score']
        age = patient.get('age', 50)
        
        # Simple rule-based prediction
        icu_probability = 0
        
        # ASA factor
        if asa_score >= 4:
            icu_probability = 80
        elif asa_score == 3:
            icu_probability = 40
        elif asa_score == 2:
            icu_probability = 15
        else:
            icu_probability = 5
        
        # Age factor
        if age > 80:
            icu_probability += 20
        elif age > 70:
            icu_probability += 10
        
        # Surgery type factor
        high_risk_surgeries = [
            'cardiac', 'thoracic', 'vascular', 'neurosurgery', 
            'esophagectomy', 'whipple', 'liver resection'
        ]
        if any(surgery in surgery_type.lower() for surgery in high_risk_surgeries):
            icu_probability += 30
        
        # Cap at 100
        icu_probability = min(icu_probability, 95)
        
        if icu_probability >= 70:
            recommendation = "Plan for ICU admission"
            icu_level = "Likely"
        elif icu_probability >= 40:
            recommendation = "Reserve ICU bed, monitor closely"
            icu_level = "Possible"
        else:
            recommendation = "PACU observation, step-down unit as needed"
            icu_level = "Unlikely"
        
        return {
            "icu_admission_probability": icu_probability,
            "icu_level": icu_level,
            "recommendation": recommendation,
            "estimated_los_days": self._estimate_length_of_stay(asa_score, surgery_type)
        }
    
    def comprehensive_risk_assessment(self, patient: Dict, surgery_type: str, surgery_duration_hours: float = 2.0,
                                        urgency: str = 'elective', anesthesia_type: str = 'general',
                                        estimated_blood_loss: str = 'moderate', positioning: str = 'supine',
                                        is_custom: bool = False, custom_risk_level: str = None, 
                                        custom_description: str = None) -> Dict:
        """
        Complete pre-operative risk assessment combining all calculators
        Now includes: urgency, anesthesia type, blood loss estimate, positioning AND custom surgery overrides
        """
        asa = self.calculate_asa_score(patient)
        rcri = self.calculate_rcri(patient, surgery_type)
        icu = self.predict_icu_admission(patient, surgery_type)
        nsqip = self.calculate_nsqip_risk(patient, surgery_type, surgery_duration_hours)
        
        # Risk modifiers based on new parameters
        risk_modifiers = []
        modifier_score = 0
        
        # Custom Surgery Logic
        if is_custom and custom_risk_level:
            risk_modifiers.append(f"Custom Procedure ({custom_risk_level.upper()} risk)")
            
            # If custom surgery is high risk, ensure RCRI accounts for it
            if custom_risk_level == 'high':
                # Check if RCRI already caught it (unlikely if name is unique)
                if "High-risk surgery type" not in rcri['risk_factors']:
                    rcri['rcri_score'] += 1
                    rcri['risk_factors'].append("High-risk surgery type (Custom)")
                    # Re-evaluate RCRI category
                    if rcri['rcri_score'] >= 3:
                        rcri['risk_category'] = "High"
                        rcri['cardiac_event_risk_percent'] = 5.4
                    elif rcri['rcri_score'] == 2:
                        rcri['risk_category'] = "Moderate"
                        rcri['cardiac_event_risk_percent'] = 2.4
                    elif rcri['rcri_score'] == 1:
                        rcri['risk_category'] = "Low"
                        rcri['cardiac_event_risk_percent'] = 1.0

        # Emergency surgery is a MAJOR risk modifier
        if urgency == 'emergent':
            modifier_score += 2
            risk_modifiers.append("Emergency surgery (+2 risk points)")
            asa['asa_score'] = min(asa['asa_score'] + 1, 5)  # Emergency adds E suffix / increases ASA
            asa['asa_description'] += " (Emergency)"
        
        # High blood loss increases risk
        blood_loss_risk = {
            'minimal': 0, 'low': 0, 'moderate': 0,
            'significant': 1, 'major': 2, 'massive': 3
        }
        if estimated_blood_loss in blood_loss_risk:
            modifier_score += blood_loss_risk[estimated_blood_loss]
            if blood_loss_risk[estimated_blood_loss] > 0:
                risk_modifiers.append(f"High blood loss expected: {estimated_blood_loss} (+{blood_loss_risk[estimated_blood_loss]} points)")
        
        # Positioning risk
        positioning_risk = {
            'supine': 0, 'lithotomy': 0, 'trendelenburg': 1,
            'reverse_trendelenburg': 0, 'lateral': 1, 'prone': 2,
            'beach_chair': 2, 'knee_chest': 1
        }
        if positioning in positioning_risk:
            modifier_score += positioning_risk[positioning]
            if positioning_risk[positioning] > 0:
                risk_modifiers.append(f"High-risk position: {positioning} (+{positioning_risk[positioning]} points)")
        
        # Anesthesia considerations
        anesthesia_notes = []
        if anesthesia_type == 'general':
            anesthesia_notes.append("General anesthesia - standard airway management")
        elif anesthesia_type == 'combined':
            modifier_score += 1
            anesthesia_notes.append("Combined GA + Regional - complex anesthesia plan")
            risk_modifiers.append("Combined anesthesia (+1 point)")
        elif 'regional' in anesthesia_type:
            anesthesia_notes.append(f"Regional anesthesia - potential for spinal/epidural complications")
        elif anesthesia_type == 'mac':
            anesthesia_notes.append("MAC - monitor for over-sedation")
        
        # Recalculate overall risk with modifiers
        total_risk_score = asa['asa_score'] + rcri['rcri_score'] + modifier_score
        
        # Override or adjust based on custom risk
        if is_custom and custom_risk_level == 'high':
            # Force at least moderate/high
            if total_risk_score < 5: total_risk_score = 5 # Ensure at least moderate
        
        if total_risk_score >= 8 or asa['asa_score'] >= 4 or urgency == 'emergent':
            overall_risk = "High"
            clearance_needed = "Extensive pre-op workup, cardiac clearance, anesthesia consult, ICU bed reservation"
        elif total_risk_score >= 5 or asa['asa_score'] == 3 or rcri['rcri_score'] >= 2:
            overall_risk = "Moderate"
            clearance_needed = "Standard pre-op testing, consider cardiology consult, blood bank notification"
        else:
            overall_risk = "Low"
            clearance_needed = "Routine pre-op clearance"
        
        # Force high risk if custom says so and we aren't already
        if is_custom and custom_risk_level == 'high' and overall_risk == 'Low':
            overall_risk = "Moderate" # Upgrade to at least moderate
            
        # Special considerations
        special_considerations = self._generate_special_considerations(patient, asa, rcri)
        
        if is_custom and custom_description:
             special_considerations.insert(0, f"Custom Procedure Notes: {custom_description}")
        
        # Add procedure-specific considerations
        if urgency == 'emergent':
            special_considerations.insert(0, "EMERGENCY CASE - expedite all clearances")
        
        if estimated_blood_loss in ['major', 'massive']:
            special_considerations.append(f"Blood products: Type and screen required, consider T&C 2-4 units")
            if estimated_blood_loss == 'massive':
                special_considerations.append("Activate Massive Transfusion Protocol (MTP) standby")
        
        if positioning == 'prone':
            special_considerations.append("Prone positioning - eye protection, padding, airway management considerations")
        elif positioning == 'beach_chair':
            special_considerations.append("Beach chair - monitor for hypotension, air embolism risk")
        elif positioning == 'lateral':
            special_considerations.append("Lateral decubitus - padding, ventilation management")
        
        # Add anesthesia notes to considerations
        special_considerations.extend(anesthesia_notes)
        
        return {
            "patient_name": patient.get('name', 'Unknown'),
            "surgery_planned": surgery_type,
            "overall_risk_category": overall_risk,
            "total_risk_score": total_risk_score,
            "asa_classification": asa,
            "cardiac_risk": rcri,
            "nsqip_predictions": nsqip,
            "icu_prediction": icu,
            "pre_op_clearance": clearance_needed,
            "special_considerations": special_considerations,
            "risk_modifiers": risk_modifiers,
            "procedure_parameters": {
                "urgency": urgency,
                "anesthesia_type": anesthesia_type,
                "estimated_blood_loss": estimated_blood_loss,
                "positioning": positioning,
                "duration_hours": surgery_duration_hours
            },
            "assessment_date": "2025-12-18"
        }
    
    def _asa_mortality_risk(self, asa_score: int) -> str:
        """Mortality risk by ASA score"""
        risks = {
            1: "0.1-0.2%",
            2: "0.2-0.5%",
            3: "1.8-4.3%",
            4: "7.8-23%",
            5: "9.4-50%",
            6: "N/A (organ donor)"
        }
        return risks.get(asa_score, "Unknown")
    
    def _asa_anesthesia_notes(self, asa_score: int) -> str:
        """Anesthesia considerations by ASA score"""
        notes = {
            1: "No special considerations",
            2: "Standard monitoring, manage chronic conditions",
            3: "Enhanced monitoring, optimize medical conditions pre-op",
            4: "Intensive monitoring, ICU likely, optimize hemodynamics",
            5: "Extreme caution, ICU mandatory, high mortality risk",
            6: "N/A"
        }
        return notes.get(asa_score, "Consult anesthesia")
    
    def _rcri_recommendations(self, score: int, category: str) -> List[str]:
        """Clinical recommendations based on RCRI score"""
        if score == 0:
            return ["Proceed with surgery", "Standard monitoring"]
        elif score == 1:
            return ["Consider cardiology consult", "Optimize medications", "Standard monitoring"]
        elif score == 2:
            return [
                "Cardiology clearance recommended",
                "Optimize beta-blocker therapy",
                "Consider stress test",
                "Enhanced monitoring"
            ]
        else:  # >= 3
            return [
                "Mandatory cardiology consult",
                "Consider coronary angiography",
                "Optimize all cardiac medications",
                "Continuous telemetry",
                "ICU admission",
                "Consider postponing non-urgent surgery"
            ]
    
    def _estimate_length_of_stay(self, asa_score: int, surgery_type: str) -> str:
        """Estimate hospital length of stay"""
        base_days = 1
        
        if asa_score >= 4:
            base_days = 7
        elif asa_score == 3:
            base_days = 4
        elif asa_score == 2:
            base_days = 2
        
        # Adjust for surgery type
        if any(term in surgery_type.lower() for term in ['cardiac', 'thoracic', 'whipple']):
            base_days += 3
        elif any(term in surgery_type.lower() for term in ['major', 'vascular', 'orthopedic']):
            base_days += 1
        
        return f"{base_days}-{base_days + 2} days"
    
    def _generate_special_considerations(self, patient: Dict, asa: Dict, rcri: Dict) -> List[str]:
        """Generate special pre-op considerations"""
        considerations = []
        
        age = patient.get('age', 0)
        if age > 80:
            considerations.append("Elderly patient - increased anesthesia sensitivity, delirium risk")
        
        if asa['asa_score'] >= 3:
            considerations.append("Optimize all chronic conditions pre-operatively")
        
        if rcri['rcri_score'] >= 2:
            considerations.append("High cardiac risk - consider non-invasive stress testing")
        
        # Check for anticoagulation
        meds = [m.get('name', '').lower() for m in patient.get('current_medications', [])]
        if any(med in ['warfarin', 'apixaban', 'rivaroxaban', 'dabigatran'] for med in meds):
            considerations.append("Anticoagulation management - bridge or hold per protocol")
        
        if any('aspirin' in med or 'clopidogrel' in med for med in meds):
            considerations.append("Antiplatelet therapy - discuss with surgeon re: holding")
        
        # Check for diabetes
        if any('insulin' in med or 'metformin' in med for med in meds):
            considerations.append("Diabetes management - NPO insulin protocol, hold metformin")
        
        return considerations if considerations else ["Standard pre-operative preparation"]
    
    def _fallback_nsqip(self, patient: Dict, surgery_type: str) -> Dict:
        """Fallback NSQIP when AI unavailable"""
        asa_score = self.calculate_asa_score(patient)['asa_score']
        
        # Very simplified estimates based on ASA
        if asa_score >= 4:
            base_risk = 30
        elif asa_score == 3:
            base_risk = 15
        else:
            base_risk = 5
        
        return {
            "any_complication": base_risk,
            "serious_complication": base_risk * 0.4,
            "pneumonia": base_risk * 0.2,
            "cardiac_complication": base_risk * 0.3,
            "surgical_site_infection": base_risk * 0.25,
            "uti": base_risk * 0.15,
            "vte": base_risk * 0.1,
            "acute_kidney_injury": base_risk * 0.2,
            "readmission_30day": base_risk * 0.5,
            "death_30day": base_risk * 0.1,
            "overall_risk_category": "Moderate" if asa_score >= 3 else "Low",
            "key_risk_factors": ["ASA score", "Age", "Surgery type"],
            "recommendations": "Detailed NSQIP predictions unavailable. Use clinical judgment."
        }


# Singleton instance
surgical_risk_engine = SurgicalRiskEngine()
