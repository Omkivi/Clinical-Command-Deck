"""
Sepsis Early Detection & Monitoring Engine

Real-time sepsis screening and management:
1. qSOFA (Quick Sequential Organ Failure Assessment)
2. SIRS Criteria (Systemic Inflammatory Response Syndrome)
3. SOFA Score (Sequential Organ Failure Assessment)
4. Sepsis-3 Criteria
5. 1-Hour Bundle Compliance Tracking
6. Lactate Trend Analysis
7. Early Warning Score
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SepsisDetectionEngine:
    """
    Early detection and monitoring system for sepsis.
    
    Implements latest Sepsis-3 criteria and sepsis bundles.
    """
    
    def __init__(self):
        pass
    
    def screen_for_sepsis(self, patient: Dict, vitals: Dict, labs: Dict = None) -> Dict:
        """
        Comprehensive sepsis screening
        
        Args:
            patient: Patient data
            vitals: Current vitals {
                'temp_c': float,
                'hr': int (heart rate),
                'rr': int (respiratory rate),
                'sbp': int (systolic BP),
                'map': int (mean arterial pressure),
                'spo2': int (oxygen saturation),
                'mental_status': 'alert'|'confused'|'unresponsive'
            }
            labs: Latest labs (if available)
        
        Returns:
            Sepsis risk assessment with alerts
        """
        # Calculate all scores
        qsofa = self.calculate_qsofa(vitals)
        sirs = self.calculate_sirs(vitals, labs)
        sofa = self.calculate_sofa(vitals, labs, patient)
        
        # Sepsis-3 criteria: Infection + SOFA >= 2
        has_infection_suspected = self._check_infection_suspected(patient, vitals, labs)
        
        # Determine sepsis probability
        sepsis_risk = "No"
        alert_level = "ROUTINE"
        recommendations = []
        
        if qsofa['score'] >= 2:
            sepsis_risk = "High Risk - qSOFA Positive"
            alert_level = "CRITICAL"
            recommendations.extend([
                "🚨 IMMEDIATE SEPSIS PROTOCOL ACTIVATION",
                "Obtain blood cultures x2 before antibiotics",
                "Administer broad-spectrum antibiotics within 1 hour",
                "IV fluid resuscitation 30ml/kg crystalloid",
                "Measure lactate",
                "Notify ICU/rapid response team"
            ])
        
        elif sofa['total_score'] >= 2 and has_infection_suspected:
            sepsis_risk = "Sepsis Likely (Sepsis-3 Criteria)"
            alert_level = "URGENT"
            recommendations.extend([
                "⚠️ Start sepsis bundle NOW",
                "Blood cultures before antibiotics",
                "Broad-spectrum antibiotics within 3 hours",
                "Lactate measurement",
                "30ml/kg IV fluids if hypotensive or lactate >4",
                "Frequent reassessment"
            ])
        
        elif sirs['criteria_met'] >= 2 and has_infection_suspected:
            sepsis_risk = "Possible Sepsis (SIRS + Infection)"
            alert_level = "URGENT"
            recommendations.extend([
                "Monitor closely for deterioration",
                "Consider sepsis workup",
                "Serial lactates",
                "Trend vital signs q1-2h"
            ])
        
        elif qsofa['score'] >= 1 or sirs['criteria_met'] >= 1:
            sepsis_risk = "At Risk - Monitor Closely"
            alert_level = "CAUTION"
            recommendations.extend([
                "Serial vital signs",
                "Consider infection source",
                "Trending labs if available"
            ])
        
        return {
            "sepsis_risk": sepsis_risk,
            "alert_level": alert_level,
            "qsofa_score": qsofa,
            "sirs_score": sirs,
            "sofa_score": sofa,
            "infection_suspected": has_infection_suspected,
            "recommendations": recommendations,
            "bundle_checklist": self._generate_bundle_checklist() if alert_level in ["CRITICAL", "URGENT"] else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_qsofa(self, vitals: Dict) -> Dict:
        """
        Quick SOFA Score (0-3)
        
        1 point each for:
        - Respiratory rate >= 22/min
        - Altered mentation
        - Systolic BP <= 100 mmHg
        
        Score >= 2 = High risk of poor outcome
        """
        score = 0
        criteria = []
        
        rr = vitals.get('rr', 0)
        if rr >= 22:
            score += 1
            criteria.append(f"Tachypnea (RR {rr})")
        
        mental_status = vitals.get('mental_status', 'alert').lower()
        if mental_status in ['confused', 'lethargic', 'obtunded', 'unresponsive']:
            score += 1
            criteria.append(f"Altered mentation ({mental_status})")
        
        sbp = vitals.get('sbp', 120)
        if sbp <= 100:
            score += 1
            criteria.append(f"Hypotension (SBP {sbp})")
        
        return {
            "score": score,
            "criteria_met": criteria,
            "interpretation": "HIGH RISK" if score >= 2 else "Lower risk" if score == 1 else "Low risk",
            "recommendation": "Activate sepsis protocol" if score >= 2 else "Monitor closely" if score == 1 else "Continue routine care"
        }
    
    def calculate_sirs(self, vitals: Dict, labs: Dict = None) -> Dict:
        """
        SIRS Criteria (0-4)
        
        1 point each for:
        - Temp >38°C (100.4°F) or <36°C (96.8°F)
        - Heart rate >90/min
        - Respiratory rate >20/min
        - WBC >12,000 or <4,000 or >10% bands
        
        >= 2 criteria = SIRS positive
        """
        score = 0
        criteria = []
        
        temp_c = vitals.get('temp_c', 37)
        if temp_c > 38 or temp_c < 36:
            score += 1
            criteria.append(f"Temperature {temp_c}°C")
        
        hr = vitals.get('hr', 0)
        if hr > 90:
            score += 1
            criteria.append(f"Tachycardia (HR {hr})")
        
        rr = vitals.get('rr', 0)
        if rr > 20:
            score += 1
            criteria.append(f"Tachypnea (RR {rr})")
        
        if labs:
            wbc = labs.get('wbc', {}).get('value', 7)
            bands = labs.get('bands_percent', 0)
            if wbc > 12 or wbc < 4 or bands > 10:
                score += 1
                criteria.append(f"WBC abnormal ({wbc}K)")
        
        return {
            "criteria_met": score,
            "criteria": criteria,
            "sirs_positive": score >= 2,
            "interpretation": "SIRS Positive" if score >= 2 else "SIRS Negative"
        }
    
    def calculate_sofa(self, vitals: Dict, labs: Dict = None, patient: Dict = None) -> Dict:
        """
        SOFA Score (Sequential Organ Failure Assessment) 0-24
        
        Assesses 6 organ systems (0-4 points each):
        1. Respiration (PaO2/FiO2 ratio)
        2. Coagulation (Platelets)
        3. Liver (Bilirubin)
        4. Cardiovascular (MAP or vasopressors)
        5. CNS (Glasgow Coma Scale)
        6. Renal (Creatinine or urine output)
        """
        total_score = 0
        breakdown = {}
        
        # 1. Respiration
        if labs:
            pao2 = labs.get('pao2', {}).get('value', 400)
            fio2 = labs.get('fio2', 0.21)
            pf_ratio = pao2 / fio2 if fio2 > 0 else 400
            
            if pf_ratio < 100:
                resp_score = 4
            elif pf_ratio < 200:
                resp_score = 3
            elif pf_ratio < 300:
                resp_score = 2
            elif pf_ratio < 400:
                resp_score = 1
            else:
                resp_score = 0
            
            breakdown['respiration'] = resp_score
            total_score += resp_score
        
        # 2. Coagulation
        if labs:
            platelets = labs.get('platelets', {}).get('value', 200)
            if platelets < 20:
                coag_score = 4
            elif platelets < 50:
                coag_score = 3
            elif platelets < 100:
                coag_score = 2
            elif platelets < 150:
                coag_score = 1
            else:
                coag_score = 0
            
            breakdown['coagulation'] = coag_score
            total_score += coag_score
        
        # 3. Liver
        if labs:
            bilirubin = labs.get('bilirubin', {}).get('value', 1.0)
            if bilirubin >= 12:
                liver_score = 4
            elif bilirubin >= 6:
                liver_score = 3
            elif bilirubin >= 2:
                liver_score = 2
            elif bilirubin >= 1.2:
                liver_score = 1
            else:
                liver_score = 0
            
            breakdown['liver'] = liver_score
            total_score += liver_score
        
        # 4. Cardiovascular
        map_val = vitals.get('map', 0)
        on_pressors = patient.get('on_vasopressors', False) if patient else False
        
        if on_pressors:
            cardio_score = 3  # Simplified - actual score depends on dose
        elif map_val < 70:
            cardio_score = 1
        else:
            cardio_score = 0
        
        breakdown['cardiovascular'] = cardio_score
        total_score += cardio_score
        
        # 5. CNS (Glasgow Coma Scale)
        gcs = vitals.get('gcs', 15)
        if gcs < 6:
            cns_score = 4
        elif gcs < 10:
            cns_score = 3
        elif gcs < 13:
            cns_score = 2
        elif gcs < 15:
            cns_score = 1
        else:
            cns_score = 0
        
        breakdown['cns'] = cns_score
        total_score += cns_score
        
        # 6. Renal
        if labs:
            creatinine = labs.get('creatinine', {}).get('value', 1.0)
            if creatinine >= 5:
                renal_score = 4
            elif creatinine >= 3.5:
                renal_score = 3
            elif creatinine >= 2:
                renal_score = 2
            elif creatinine >= 1.2:
                renal_score = 1
            else:
                renal_score = 0
            
            breakdown['renal'] = renal_score
            total_score += renal_score
        
        return {
            "total_score": total_score,
            "breakdown": breakdown,
            "interpretation": self._interpret_sofa(total_score),
            "mortality_risk": self._sofa_mortality_estimate(total_score)
        }
    
    def track_lactate_trend(self, lactate_values: List[Dict]) -> Dict:
        """
        Track lactate trends for sepsis monitoring
        
        Args:
            lactate_values: [{'value': 4.5, 'timestamp': '2025-12-16T10:00:00'}, ...]
        
        Returns:
            Trend analysis
        """
        if not lactate_values or len(lactate_values) == 0:
            return {"status": "No lactate data"}
        
        # Sort by timestamp
        sorted_values = sorted(lactate_values, key=lambda x: x['timestamp'])
        
        latest = sorted_values[-1]['value']
        initial = sorted_values[0]['value']
        
        # Trend
        if len(sorted_values) >= 2:
            if latest < initial:
                trend = "IMPROVING ✓"
                clearance = ((initial - latest) / initial) * 100
            elif latest > initial:
                trend = "WORSENING ✗"
                clearance = -((latest - initial) / initial) * 100
            else:
                trend = "STABLE"
                clearance = 0
        else:
            trend = "SINGLE VALUE"
            clearance = 0
        
        # Interpretation
        if latest >= 4:
            interpretation = "CRITICAL - Severe hyperlactatemia, high mortality risk"
            action = "Aggressive resuscitation, consider ICU transfer"
        elif latest >= 2:
            interpretation = "ELEVATED - Tissue hypoperfusion likely"
            action = "IV fluids, investigate cause, recheck in 2-6 hours"
        elif latest >= 1:
            interpretation = "MILDLY ELEVATED - Monitor"
            action = "Continue current management, trend q6-12h"
        else:
            interpretation = "NORMAL"
            action = "Routine monitoring"
        
        return {
            "current_lactate": latest,
            "initial_lactate": initial,
            "trend": trend,
            "clearance_percent": round(clearance, 1),
            "interpretation": interpretation,
            "action": action,
            "goal": "Lactate clearance >10% in 6 hours",
            "values_tracked": len(sorted_values)
        }
    
    def generate_sepsis_bundle_checklist(self, activation_time: datetime = None) -> Dict:
        """
        Generate 1-hour sepsis bundle checklist
        
        Surviving Sepsis Campaign 1-Hour Bundle:
        1. Measure lactate
        2. Obtain blood cultures before antibiotics
        3. Administer broad-spectrum antibiotics
        4. Administer 30ml/kg crystalloid for hypotension or lactate >=4
        5. Apply vasopressors if hypotensive during/after fluid resuscitation
        """
        if not activation_time:
            activation_time = datetime.now()
        
        deadline_1hr = activation_time + timedelta(hours=1)
        deadline_3hr = activation_time + timedelta(hours=3)
        
        bundle = {
            "bundle_activated": activation_time.isoformat(),
            "1_hour_deadline": deadline_1hr.isoformat(),
            "3_hour_deadline": deadline_3hr.isoformat(),
            "tasks": [
                {
                    "task": "Measure lactate",
                    "deadline": "1 hour",
                    "completed": False,
                    "timestamp": None
                },
                {
                    "task": "Obtain blood cultures x2 (before antibiotics)",
                    "deadline": "1 hour",
                    "completed": False,
                    "timestamp": None
                },
                {
                    "task": "Administer broad-spectrum antibiotics",
                    "deadline": "1 hour",
                    "completed": False,
                    "timestamp": None,
                    "notes": "Empiric: Vancomycin + Piperacillin-Tazobactam (adjust per guidelines)"
                },
                {
                    "task": "IV fluid bolus 30ml/kg (if hypotensive or lactate >=4)",
                    "deadline": "3 hours",
                    "completed": False,
                    "timestamp": None
                },
                {
                    "task": "Apply vasopressors if still hypotensive",
                    "deadline": "ongoing",
                    "completed": False,
                    "timestamp": None,
                    "notes": "Target MAP >=65 mmHg"
                },
                {
                    "task": "Repeat lactate if initial >2",
                    "deadline": "2-4 hours",
                    "completed": False,
                    "timestamp": None
                }
            ],
            "compliance_status": "PENDING"
        }
        
        return bundle
    
    def _check_infection_suspected(self, patient: Dict, vitals: Dict, labs: Dict = None) -> bool:
        """Check if infection is suspected"""
        # Simplified - in reality would check clinical notes, orders, etc.
        
        # Check for fever/hypothermia
        temp_c = vitals.get('temp_c', 37)
        if temp_c > 38 or temp_c < 36:
            return True
        
        # Check for elevated WBC
        if labs:
            wbc = labs.get('wbc', {}).get('value', 7)
            if wbc > 12 or wbc < 4:
                return True
        
        # Check if patient has known infection in conditions
        conditions = patient.get('conditions', [])
        infection_keywords = ['pneumonia', 'uti', 'cellulitis', 'sepsis', 'bacteremia', 'infection']
        if any(kw in str(conditions).lower() for kw in infection_keywords):
            return True
        
        return False
    
    def _interpret_sofa(self, score: int) -> str:
        """Interpret SOFA score"""
        if score >= 15:
            return "Severe organ dysfunction, very high mortality risk"
        elif score >= 10:
            return "Significant organ dysfunction, high mortality risk"
        elif score >= 5:
            return "Moderate organ dysfunction"
        elif score >= 2:
            return "Mild organ dysfunction"
        else:
            return "Minimal or no organ dysfunction"
    
    def _sofa_mortality_estimate(self, score: int) -> str:
        """Hospital mortality estimate based on SOFA"""
        if score >= 15:
            return ">90%"
        elif score >= 12:
            return "50-90%"
        elif score >= 8:
            return "30-50%"
        elif score >= 5:
            return "15-30%"
        elif score >= 2:
            return "<15%"
        else:
            return "<10%"
    
    def _generate_bundle_checklist(self) -> Dict:
        """Generate bundle checklist"""
        return self.generate_sepsis_bundle_checklist()


# Singleton instance
sepsis_engine = SepsisDetectionEngine()
