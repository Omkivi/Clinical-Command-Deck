import random
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
from .contraindication_oracle import contraindication_oracle
from .pharmacogenomics_engine import pharmacogenomics_engine
from .api_key_manager import api_key_manager

# Load environment variables
load_dotenv()

# Configure Gemini (still needed for some operations)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class AIEngine:
    def __init__(self):
        # Use API Key Manager for automatic rotation on quota errors
        self.key_manager = api_key_manager
        self.model = self.key_manager.get_gemini_model()
        self.oracle = contraindication_oracle
        self.pgx_engine = pharmacogenomics_engine  # Pharmacogenomics integration
    
    def _check_herb_interactions(self, patient, drug_name):
        """Check for interactions between prescribed drug and patient supplements"""
        interactions = []
        supplements = patient.get("supplements", [])
        
        if not supplements:
            return interactions
            
        # Simplified interaction logic based on keywords
        # In a real system, this would use a comprehensive database
        
        drug_name_lower = drug_name.lower()
        
        # Define some common drug descriptors for matching
        is_anticoagulant = any(x in drug_name_lower for x in ["warfarin", "heparin", "eliquis", "apixaban", "rivaroxaban", "dabigatran"])
        is_antiplatelet = any(x in drug_name_lower for x in ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"])
        is_cns_depressant = any(x in drug_name_lower for x in ["alprazolam", "diazepam", "lorazepam", "zolpidem", "codeine", "oxycodone"])
        is_statin = any(x in drug_name_lower for x in ["statin"])
        is_antidiabetic = any(x in drug_name_lower for x in ["metformin", "glipizide", "insulin"])
        
        for supp in supplements:
            cyp_impact = supp.get("cyp_impact", "").lower()
            supp_name = supp.get("name", "").lower()
            risk_level = "MODERATE"
            effect = ""
            
            # 1. Bleeding Risk (The most common dangerous interaction)
            if (is_anticoagulant or is_antiplatelet) and any(x in cyp_impact for x in ["antiplatelet", "bleeding"]):
                interactions.append({
                    "herb": supp["name"],
                    "drug": drug_name,
                    "level": "CRITICAL",
                    "mechanism": "Additive antiplatelet/anticoagulant effect",
                    "effect": "Significantly increased risk of spontaneous bleeding"
                })
                continue
                
            # 2. CYP3A4 Inducers (Reduces drug efficacy)
            # Many cardiac meds, statins, HIV meds are CYP3A4 substrates
            if "inducer" in cyp_impact and (is_statin or is_anticoagulant):
                interactions.append({
                    "herb": supp["name"],
                    "drug": drug_name,
                    "level": "HIGH",
                    "mechanism": "CYP induction (accelerated metabolism)",
                    "effect": f"May reduce {drug_name} levels, leading to therapeutic failure"
                })
                continue

            # 3. CNS Depression
            if is_cns_depressant and any(x in supp_name for x in ["valerian", "ashwagandha", "kava", "chamomile"]):
                 interactions.append({
                    "herb": supp["name"],
                    "drug": drug_name,
                    "level": "MODERATE",
                    "mechanism": "Additive CNS depression",
                    "effect": "Increased sedation, dizziness, respiratory depression risk"
                })
                 continue
            
            # 4. Specific known killers
            if "st. john's wort" in supp_name and (is_anticoagulant or "digoxin" in drug_name_lower):
                 interactions.append({
                    "herb": supp["name"],
                    "drug": drug_name,
                    "level": "CRITICAL",
                    "mechanism": "Strong CYP3A4/P-gp Induction",
                    "effect": "Dangerous reduction in drug levels; potential for stroke/clots"
                })

        return interactions

    def simulate_scenario(self, patient, drug_name, dosage, duration="4 weeks"):
        """
        Advanced drug simulation with comprehensive safety analysis.
        
        Analyzes:
        - All patient data (age, allergies, medical history, current meds, vitals, labs)
        - Drug-drug interactions
        - Drug-disease contraindications
        - Drug-food interactions
        - Organ function considerations
        - Age-specific concerns (Beers Criteria for elderly)
        - Pharmacogenomics (CYP450 enzyme status)
        - Weight-based dosing adjustments
        
        Args:
            patient: Patient data dictionary
            drug_name: Name of the drug to simulate
            dosage: Dosage and frequency
            duration: Intervention period (e.g., "4 weeks", "12 weeks", "indefinite")
        
        Returns detailed simulation with safety warnings and intervention recommendations.
        """
        # Step 0: Pharmacogenomics Analysis
        pgx_alert = None
        weight_based_dose = None
        
        if patient:
            # Check for drug-gene interactions
            pgx_alert = self.pgx_engine.check_drug_pgx_interaction(patient, drug_name)
            
            # Calculate weight-based dosing if applicable
            weight_based_dose = self.pgx_engine.calculate_weight_based_dose(patient, drug_name)

        # Step 0.5: Herb/Supplement Interaction Check (New Safety Shield)
        herb_interactions = []
        if patient:
            herb_interactions = self._check_herb_interactions(patient, drug_name)
        
        # Step 1: Comprehensive Safety Analysis using Oracle
        safety_report = self.oracle.analyze_patient_safety(patient, drug_name, dosage)

        # Add Herb Interactions to safety report
        if herb_interactions:
            if "warnings" not in safety_report:
                safety_report["warnings"] = []
                
            for interaction in herb_interactions:
                # Add to warnings list
                safety_report["warnings"].append({
                    "type": "INTERACTION_HERB",
                    "level": interaction["level"],
                    "drug": interaction["drug"],
                    "source": interaction["herb"],
                    "effect": interaction["effect"],
                    "mechanism": interaction["mechanism"]
                })
                
                # If critical, force a contraindication
                if interaction["level"] == "CRITICAL":
                    safety_report["can_prescribe"] = False
                    safety_report["primary_concern"] = f"CRITICAL INTERACTION with {interaction['herb']}: {interaction['effect']}"
        
        # Add pharmacogenomics to safety report
        if pgx_alert:
            if pgx_alert.get("alert_level") == "CRITICAL":
                safety_report["can_prescribe"] = False
                safety_report["primary_concern"] = f"Pharmacogenomic contraindication: {pgx_alert.get('effect')}"
            
            if "warnings" not in safety_report:
                safety_report["warnings"] = []
            safety_report["warnings"].append({
                "type": "PHARMACOGENOMICS",
                "level": pgx_alert.get("alert_level"),
                "enzyme": pgx_alert.get("enzyme") or pgx_alert.get("marker"),
                "status": pgx_alert.get("patient_status"),
                "effect": pgx_alert.get("effect"),
                "recommendation": pgx_alert.get("recommendation"),
                "special_note": pgx_alert.get("special_note")
            })
        
        # Add weight-based dosing recommendation
        if weight_based_dose:
            safety_report["weight_based_dosing"] = weight_based_dose
        
        # Step 2: Determine if simulation should proceed
        if not safety_report["can_prescribe"]:
            # Critical contraindication - return immediate warning
            result = self._generate_contraindication_response(patient, drug_name, dosage, safety_report)
            result["pharmacogenomics_alert"] = pgx_alert
            result["weight_based_dosing"] = weight_based_dose
            return result
        
        # Step 3: Run enhanced simulation with safety context
        simulation_result = self._run_advanced_simulation(patient, drug_name, dosage, safety_report, duration)
        
        # Add pharmacogenomics data to result
        simulation_result["pharmacogenomics_alert"] = pgx_alert
        simulation_result["weight_based_dosing"] = weight_based_dose
        
        # Step 4: Add AI-powered interpretation if available
        if self.model and GOOGLE_API_KEY:
            try:
                ai_interpretation = self._get_ai_interpretation(patient, drug_name, dosage, safety_report, simulation_result)
                simulation_result["ai_interpretation"] = ai_interpretation
            except Exception as e:
                print(f"   [FAIL] AI interpretation failed: {e}")
                simulation_result["ai_interpretation"] = None
        
        return simulation_result
    
    def simulate_with_council(self, patient, drug_name, dosage, duration="4 weeks"):
        """
        Advanced drug simulation with MULTI-MODEL AI COUNCIL deliberation.
        
        Uses 4 different AI models (Gemini, GPT-4, Mistral, Llama) to:
        1. Independently analyze the drug safety
        2. Debate each other's perspectives (2 rounds)
        3. Reach a weighted consensus
        
        This eliminates single-model bias for maximum patient safety.
        
        Returns:
            Dict with simulation results AND council deliberation
        """
        # Step 1: Run base simulation (deterministic + Oracle)
        base_result = self.simulate_scenario(patient, drug_name, dosage, duration)
        
        # Step 2: Get AI Council deliberation
        try:
            from .ai_council import ai_council
            
            # Prepare case data for council
            case_data = {
                "drug": drug_name,
                "dosage": dosage,
                "duration": duration,
                "safety_report": base_result.get("safety_analysis", {}),
                "efficacy_score": base_result.get("efficacy_score", 0),
                "outcome": base_result.get("outcome", ""),
                "side_effects": base_result.get("side_effects", []),
                "interventions": base_result.get("interventions_required", [])
            }
            
            # Run full council deliberation with debate
            council_verdict = ai_council.deliberate(
                case_type="SIMULATION",
                case_data=case_data,
                patient=patient
            )
            
            # Add council verdict to results
            base_result["council_deliberation"] = council_verdict.to_dict()
            base_result["council_verdict"] = council_verdict.final_verdict
            base_result["council_consensus"] = council_verdict.consensus_score
            base_result["council_dissent"] = council_verdict.dissenting_models
            
        except Exception as e:
            print(f"Council deliberation error: {e}")
            import traceback
            traceback.print_exc()
            base_result["council_deliberation"] = None
            base_result["council_error"] = str(e)
        
        return base_result
    
    def _generate_contraindication_response(self, patient, drug_name, dosage, safety_report):
        """Generate response for contraindicated drug"""
        
        # Collect all blocking reasons
        blocking_reasons = []
        for alert in safety_report["critical_alerts"]:
            blocking_reasons.append(alert.get("message", str(alert)))
        
        for reason in safety_report["blocked_reasons"]:
            blocking_reasons.append(str(reason))
        
        # Format alternatives from critical alerts
        alternatives = []
        for alert in safety_report["critical_alerts"]:
            if isinstance(alert, dict) and "alternatives" in alert:
                alts = alert["alternatives"]
                if isinstance(alts, list):
                    alternatives.extend(alts)
                elif isinstance(alts, str):
                    alternatives.append(alts)
        
        return {
            "outcome": "CONTRAINDICATED",
            "details": f"❌ {drug_name} ({dosage}) CANNOT be prescribed for this patient due to critical contraindications.",
            "efficacy_score": 0,
            "safety_score": safety_report["safety_score"],
            "overall_risk": safety_report["overall_risk"],
            "can_simulate": False,
            "blocking_reasons": blocking_reasons,
            "alternative_drugs": list(set(alternatives)) if alternatives else ["Consult pharmacist for alternative therapy"],
            "safety_analysis": safety_report,
            "predicted_outcome": {
                "status": "BLOCKED",
                "reason": "Critical contraindication prevents safe use"
            },
            "clinical_recommendation": "⛔ DO NOT PRESCRIBE. See blocking reasons and alternatives above."
        }
    
    def _run_advanced_simulation(self, patient, drug_name, dosage, safety_report, duration="4 weeks"):
        """Run advanced simulation incorporating safety context"""
        
        # Simulate network latency
        time.sleep(1.5)
        
        # Base efficacy calculation
        base_efficacy = random.randint(70, 95)
        
        # Adjust efficacy based on interactions and warnings
        efficacy_penalty = 0
        
        # Critical interactions reduce efficacy
        if safety_report["critical_alerts"]:
            efficacy_penalty += 40
        
        # Major warnings
        if safety_report["major_warnings"]:
            efficacy_penalty += len(safety_report["major_warnings"]) * 15
        
        # Moderate warnings
        if safety_report["moderate_warnings"]:
            efficacy_penalty += len(safety_report["moderate_warnings"]) * 8
        
        # Organ dysfunction - Use actual lab values from reports
        from .patient_data_aggregator import patient_data_aggregator
        
        lab_values = patient_data_aggregator.get_latest_lab_values(patient)
        renal_function = "Normal"
        hepatic_function = "Normal"
        
        # Determine renal function from lab values
        if 'creatinine' in lab_values or 'egfr' in lab_values:
            creatinine = lab_values.get('creatinine', {}).get('value', 0)
            egfr = lab_values.get('egfr', {}).get('value', 100)
            
            if egfr > 0:  # Use eGFR if available
                if egfr < 15:
                    renal_function = "Severe Renal Impairment (eGFR <15)"
                    efficacy_penalty += 25
                elif egfr < 30:
                    renal_function = "Moderate-Severe Renal Impairment (eGFR 15-29)"
                    efficacy_penalty += 20
                elif egfr < 60:
                    renal_function = "Mild-Moderate Renal Impairment (eGFR 30-59)"
                    efficacy_penalty += 10
            elif creatinine > 0:  # Fallback to creatinine
                if creatinine > 3.0:
                    renal_function = "Severe Renal Impairment (Cr >3.0)"
                    efficacy_penalty += 25
                elif creatinine > 2.0:
                    renal_function = "Moderate Renal Impairment (Cr 2.0-3.0)"
                    efficacy_penalty += 20
                elif creatinine > 1.5:
                    renal_function = "Mild Renal Impairment (Cr 1.5-2.0)"
                    efficacy_penalty += 10
        else:
            # Fallback to vitals dict if no lab values
            vitals = patient.get("vitals", {})
            renal_function = vitals.get("renal_function", patient.get("renal_function", "Normal"))
            
            if "Severe" in renal_function:
                efficacy_penalty += 20
            elif "Moderate" in renal_function or "Mild" in renal_function:
                efficacy_penalty += 10
        
        # Determine hepatic function from lab values
        if 'ast' in lab_values or 'alt' in lab_values:
            ast = lab_values.get('ast', {}).get('value', 0)
            alt = lab_values.get('alt', {}).get('value', 0)
            bilirubin = lab_values.get('bilirubin', {}).get('value', 0)
            
            # Check for hepatic impairment
            if (ast > 120 or alt > 120) or bilirubin > 3.0:
                hepatic_function = "Significant Hepatic Impairment"
                efficacy_penalty += 15
            elif (ast > 80 or alt > 80) or bilirubin > 2.0:
                hepatic_function = "Mild-Moderate Hepatic Impairment"
                efficacy_penalty += 10
        else:
            # Fallback to vitals dict
            vitals = patient.get("vitals", {})
            hepatic_function = vitals.get("hepatic_function", patient.get("hepatic_function", "Normal"))
            
            if "Impairment" in hepatic_function:
                efficacy_penalty += 15
        
        # Duration-based adjustments
        duration_weeks = self._parse_duration_weeks(duration)
        
        # Longer durations may have cumulative effects
        long_term_bonus = 0
        cumulative_risk = 0
        if duration_weeks >= 12:
            # Long-term therapy often has better outcomes for chronic conditions
            long_term_bonus = random.randint(5, 10)
            # But also higher cumulative risk for side effects
            cumulative_risk = random.randint(5, 15)
        elif duration_weeks >= 8:
            long_term_bonus = random.randint(2, 5)
            cumulative_risk = random.randint(2, 8)
        
        # Calculate final efficacy
        final_efficacy = max(20, base_efficacy - efficacy_penalty + long_term_bonus)
        
        # Adjust safety score based on duration
        adjusted_safety_score = safety_report["safety_score"] - cumulative_risk
        adjusted_safety_score = max(0, min(100, adjusted_safety_score))
        
        # Determine outcome based on safety score and efficacy
        if adjusted_safety_score < 40:
            outcome = "HIGH RISK - CAUTION REQUIRED"
            outcome_class = "ADVERSE"
        elif adjusted_safety_score < 70:
            outcome = "MODERATE RISK - MONITORING REQUIRED"
            outcome_class = "SUBOPTIMAL"
        elif final_efficacy > 75:
            outcome = "POSITIVE RESPONSE EXPECTED"
            outcome_class = "POSITIVE"
        elif final_efficacy > 50:
            outcome = "MODEST BENEFIT EXPECTED"
            outcome_class = "MODEST"
        else:
            outcome = "SUBOPTIMAL RESPONSE"
            outcome_class = "SUBOPTIMAL"
        
        # Generate detailed side effects based on interactions
        side_effects = self._generate_side_effects(drug_name, safety_report, patient)
        
        # Add duration-specific warnings
        if duration_weeks >= 12:
            side_effects.append("⏱️ Long-term use: Monitor for cumulative effects")
        
        # Generate predicted vitals changes
        vitals_change = self._predict_vitals_change(drug_name, outcome_class, patient)
        
        # Generate monitoring plan (adjusted for duration)
        monitoring_plan = self._generate_monitoring_plan(safety_report, drug_name, patient)
        
        # Generate intervention recommendations
        interventions = self._generate_interventions(safety_report, patient, drug_name)
        
        # Generate detailed explanation
        details = self._generate_detailed_explanation(patient, drug_name, dosage, safety_report, outcome_class)
        
        return {
            "outcome": outcome,
            "outcome_class": outcome_class,
            "details": details,
            "efficacy_score": final_efficacy,
            "safety_score": adjusted_safety_score,
            "overall_risk": safety_report["overall_risk"],
            "can_simulate": True,
            "intervention_duration": duration,
            "duration_weeks": duration_weeks,
            "predicted_vitals_change": vitals_change,
            "side_effects": side_effects,
            "safety_analysis": safety_report,
            "monitoring_plan": monitoring_plan,
            "interventions_required": interventions,
            "patient_factors_considered": self._get_patient_factors_summary(patient),
            "clinical_recommendation": self._generate_clinical_recommendation(safety_report, outcome_class)
        }
    
    def _parse_duration_weeks(self, duration):
        """Parse duration string to number of weeks"""
        if not duration:
            return 4
        
        duration_lower = duration.lower()
        
        if "indefinite" in duration_lower or "long-term" in duration_lower:
            return 52  # 1 year
        
        # Extract number from string
        import re
        number_match = re.search(r'(\d+)', duration_lower)
        if number_match:
            num = int(number_match.group(1))
            
            if "month" in duration_lower:
                return num * 4
            elif "week" in duration_lower:
                return num
            elif "day" in duration_lower:
                return max(1, num // 7)
            else:
                # Assume weeks if no unit specified
                return num
        
        return 4  # Default to 4 weeks
    
    def _generate_side_effects(self, drug_name, safety_report, patient):
        """Generate realistic side effects based on drug and patient factors"""
        side_effects = []
        
        # Add interaction-based side effects
        for alert in safety_report["critical_alerts"]:
            if isinstance(alert, dict) and "clinical_effect" in alert:
                side_effects.append(f"⚠️ {alert['clinical_effect']}")
        
        for warning in safety_report["major_warnings"]:
            if isinstance(warning, dict) and "clinical_effect" in warning:
                side_effects.append(f"⚠️ {warning['clinical_effect']}")
        
        # Drug-specific common side effects
        drug_side_effects = {
            "Metformin": ["GI upset", "Nausea", "Diarrhea"],
            "Lisinopril": ["Dry cough", "Dizziness", "Hypotension"],
            "Atorvastatin": ["Muscle pain", "Elevated liver enzymes"],
            "Warfarin": ["Bleeding risk", "Bruising"],
            "Aspirin": ["GI irritation", "Bleeding risk"],
            "Ibuprofen": ["GI upset", "Kidney effects"],
        }
        
        # Add common side effects for the drug
        for drug_key, effects in drug_side_effects.items():
            if drug_key.lower() in drug_name.lower():
                side_effects.extend(effects)
                break
        
        # Add generic side effects if none found
        if not side_effects:
            side_effects = ["Mild nausea", "Headache", "Fatigue"]
        
        # Add age-related side effects
        if patient.get("age", 0) >= 65:
            side_effects.append("👴 Elderly: Increased sensitivity to medication")
        
        return list(set(side_effects[:6]))  # Limit to 6 unique effects
    
    def _predict_vitals_change(self, drug_name, outcome_class, patient):
        """Predict changes in vital signs"""
        changes = {}
        
        # Blood pressure changes
        if any(term in drug_name.lower() for term in ["lisinopril", "amlodipine", "losartan", "atenolol", "metoprolol"]):
            if outcome_class == "POSITIVE":
                changes["bp"] = f"-{random.randint(8, 15)} mmHg systolic"
            else:
                changes["bp"] = f"-{random.randint(3, 7)} mmHg systolic"
        elif "ibuprofen" in drug_name.lower():
            changes["bp"] = f"+{random.randint(3, 8)} mmHg (NSAIDs can raise BP)"
        else:
            changes["bp"] = f"{random.randint(-3, 3)} mmHg (minimal change)"
        
        # Heart rate
        if "metoprolol" in drug_name.lower() or "atenolol" in drug_name.lower():
            changes["hr"] = f"-{random.randint(5, 12)} bpm (beta-blocker effect)"
        else:
            changes["hr"] = f"{random.randint(-5, 5)} bpm"
        
        # Glucose (for diabetes meds)
        if any(term in drug_name.lower() for term in ["metformin", "insulin", "glipizide"]):
            if outcome_class == "POSITIVE":
                changes["glucose"] = f"-{random.randint(30, 60)} mg/dL (fasting)"
            else:
                changes["glucose"] = f"-{random.randint(15, 30)} mg/dL (modest reduction)"
        
        # Cholesterol (for statins)
        if "statin" in drug_name.lower() or any(term in drug_name.lower() for term in ["atorvastatin", "simvastatin", "rosuvastatin"]):
            if outcome_class == "POSITIVE":
                changes["ldl_cholesterol"] = f"-{random.randint(35, 50)}% LDL reduction"
            else:
                changes["ldl_cholesterol"] = f"-{random.randint(20, 35)}% LDL reduction"
        
        return changes
    
    def _generate_monitoring_plan(self, safety_report, drug_name, patient):
        """Generate comprehensive monitoring plan"""
        monitoring = []
        
        # Add monitoring from safety report
        for item in safety_report["monitoring_required"]:
            monitoring.append(item)
        
        # Add drug-specific monitoring
        drug_monitoring = {
            "Metformin": {"test": "Renal function (Creatinine, eGFR)", "frequency": "Every 6 months"},
            "Warfarin": {"test": "INR", "frequency": "Weekly initially, then monthly when stable"},
            "Statins": {"test": "Lipid panel, Liver enzymes (AST/ALT), CK", "frequency": "Every 3-6 months"},
            "Lisinopril": {"test": "Blood pressure, Potassium, Renal function", "frequency": "Every 3 months"},
            "Digoxin": {"test": "Digoxin level, Potassium, Renal function", "frequency": "Every 3-6 months"},
        }
        
        for drug_key, mon in drug_monitoring.items():
            if drug_key.lower() in drug_name.lower():
                monitoring.append({
                    "test": mon["test"],
                    "reason": f"Standard monitoring for {drug_name}",
                    "frequency": mon["frequency"]
                })
        
        # Add baseline monitoring recommendation
        if not monitoring:
            monitoring.append({
                "test": "Clinical follow-up",
                "reason": "Monitor therapeutic response and side effects",
                "frequency": "2-4 weeks after initiation"
            })
        
        return monitoring
    
    def _generate_interventions(self, safety_report, patient, drug_name):
        """Generate intervention recommendations"""
        interventions = []
        
        # Critical interactions require dose adjustment or drug change
        if safety_report["critical_alerts"]:
            for alert in safety_report["critical_alerts"]:
                if isinstance(alert, dict):
                    intervention = {
                        "priority": "HIGH",
                        "action": alert.get("management", "Consult physician"),
                        "rationale": alert.get("clinical_effect", "Safety concern"),
                        "alternatives": alert.get("alternatives", [])
                    }
                    interventions.append(intervention)
        
        # Major warnings may need dose reduction
        if safety_report["major_warnings"]:
            interventions.append({
                "priority": "MODERATE",
                "action": "Start with lowest effective dose and titrate slowly",
                "rationale": "Major drug interactions or contraindications detected",
                "monitoring": "Close monitoring required during titration"
            })
        
        # Organ dysfunction
        vitals = patient.get("vitals", {})
        renal_function = vitals.get("renal_function", patient.get("renal_function", "Normal"))
        
        if "Severe" in renal_function or "Moderate" in renal_function:
            interventions.append({
                "priority": "HIGH",
                "action": f"Adjust dose for {renal_function}",
                "rationale": "Impaired drug clearance",
                "recommendation": "Reduce dose by 25-50% or increase dosing interval"
            })
        
        # Elderly patients
        if patient.get("age", 0) >= 75:
            interventions.append({
                "priority": "MODERATE",
                "action": "Start low, go slow - Use 50% of standard starting dose",
                "rationale": "Elderly patient with increased drug sensitivity",
                "note": "Titrate slowly based on response"
            })
        
        return interventions
    
    def _get_patient_factors_summary(self, patient):
        """Summarize all patient factors considered"""
        factors = []
        
        # Import aggregator
        from .patient_data_aggregator import patient_data_aggregator
        
        # Age
        age = patient.get("age", "Unknown")
        if age != "Unknown":
            age_cat = "Elderly (>65)" if age >= 65 else "Adult"
            factors.append(f"Age: {age} years ({age_cat})")
        
        # Allergies
        allergies = patient.get("allergies", [])
        if allergies:
            factors.append(f"Allergies: {', '.join(allergies)}")
        else:
            factors.append("Allergies: None reported")
        
        # Current medications
        current_meds = patient.get("current_medications", [])
        if current_meds:
            med_names = [m.get("name", str(m)) if isinstance(m, dict) else str(m) for m in current_meds]
            factors.append(f"Current Medications: {', '.join(med_names[:5])}" + 
                          (f" and {len(med_names) - 5} more" if len(med_names) > 5 else ""))
        else:
            factors.append("Current Medications: None")
        
        # Medical history
        history = patient.get("medical_history", [])
        if history:
            conditions = [h.get("condition", str(h)) if isinstance(h, dict) else str(h) for h in history]
            factors.append(f"Medical History: {', '.join(conditions[:3])}" +
                          (f" and {len(conditions) - 3} more" if len(conditions) > 3 else ""))
        
        # Lab values from reports
        lab_values = patient_data_aggregator.get_latest_lab_values(patient)
        if lab_values:
            key_labs = []
            if 'creatinine' in lab_values:
                cr = lab_values['creatinine']
                key_labs.append(f"Creatinine: {cr['value']} ({cr['date']})")
            if 'egfr' in lab_values:
                egfr = lab_values['egfr']
                key_labs.append(f"eGFR: {egfr['value']} ({egfr['date']})")
            if 'glucose' in lab_values:
                gluc = lab_values['glucose']
                key_labs.append(f"Glucose: {gluc['value']} ({gluc['date']})")
            
            if key_labs:
                factors.append(f"Recent Labs: {'; '.join(key_labs[:3])}")
        
        # Radiology findings
        radiology = patient_data_aggregator.get_radiology_findings(patient)
        if radiology:
            latest = radiology[0] if radiology else None
            if latest:
                finding_preview = latest['finding'][:60] + "..." if len(latest['finding']) > 60 else latest['finding']
                factors.append(f"Imaging: {finding_preview} ({latest['date']})")
        
        # Organ function (derived from labs or vitals)
        vitals = patient.get("vitals", {})
        renal = vitals.get("renal_function", patient.get("renal_function", "Normal"))
        hepatic = vitals.get("hepatic_function", patient.get("hepatic_function", "Normal"))
        
        # Override with calculated values if we have labs
        if 'egfr' in lab_values:
            egfr_val = lab_values['egfr'].get('value', 100)
            if egfr_val < 60:
                renal = f"Impaired (eGFR: {egfr_val})"
        
        factors.append(f"Renal Function: {renal}")
        factors.append(f"Hepatic Function: {hepatic}")
        
        # Lab reports and timeline summary
        lab_reports = patient.get("lab_reports", [])
        timeline_events = patient_data_aggregator.get_timeline_medical_events(patient)
        if lab_reports or timeline_events:
            factors.append(f"Data Sources: {len(lab_reports)} lab reports, {len(timeline_events)} timeline events")
        
        return factors
    
    def _generate_detailed_explanation(self, patient, drug_name, dosage, safety_report, outcome_class):
        """Generate detailed explanation of simulation results"""
        
        explanation_parts = []
        
        # Opening statement
        if outcome_class == "POSITIVE":
            explanation_parts.append(f"✅ {drug_name} ({dosage}) is expected to be effective for this patient.")
        elif outcome_class == "SUBOPTIMAL" or outcome_class == "MODEST":
            explanation_parts.append(f"⚠️ {drug_name} ({dosage}) may have reduced effectiveness due to patient-specific factors.")
        else:
            explanation_parts.append(f"⚠️ {drug_name} ({dosage}) presents significant safety concerns.")
        
        # Safety considerations
        if safety_report["critical_alerts"]:
            explanation_parts.append(f"\n⛔ CRITICAL SAFETY CONCERNS: {len(safety_report['critical_alerts'])} critical alerts detected.")
        elif safety_report["major_warnings"]:
            explanation_parts.append(f"\n🟠 {len(safety_report['major_warnings'])} major safety warnings identified.")
        elif safety_report["moderate_warnings"]:
            explanation_parts.append(f"\n🟡 {len(safety_report['moderate_warnings'])} moderate warnings noted.")
        else:
            explanation_parts.append("\n✅ No significant safety concerns detected.")
        
        # Patient factors
        age = patient.get("age", 0)
        if age >= 65:
            explanation_parts.append(f"\n👴 Elderly patient (age {age}): Increased monitoring required.")
        
        current_meds = patient.get("current_medications", [])
        if current_meds:
            explanation_parts.append(f"\n💊 Patient on {len(current_meds)} concurrent medications: Interaction analysis completed.")
        
        # Organ function
        vitals = patient.get("vitals", {})
        renal = vitals.get("renal_function", patient.get("renal_function", "Normal"))
        if renal != "Normal":
            explanation_parts.append(f"\n🔬 {renal}: Dose adjustment may be needed.")
        
        return " ".join(explanation_parts)
    
    def _generate_clinical_recommendation(self, safety_report, outcome_class):
        """Generate final clinical recommendation"""
        
        if safety_report["safety_score"] < 40:
            return "⛔ HIGH RISK: Consider alternative therapy. If no alternatives, proceed with extreme caution and intensive monitoring."
        elif safety_report["safety_score"] < 70:
            return "⚠️ MODERATE RISK: Acceptable with careful monitoring. Implement recommended interventions and monitoring plan."
        elif outcome_class == "POSITIVE":
            return "✅ RECOMMENDED: Expected to be safe and effective. Proceed with standard monitoring."
        elif outcome_class == "MODEST":
            return "✅ ACCEPTABLE: Modest benefit expected. Monitor response and consider dose optimization."
        else:
            return "⚠️ PROCEED WITH CAUTION: Benefits may not outweigh risks. Consider alternatives."
    
    def _get_ai_interpretation(self, patient, drug_name, dosage, safety_report, simulation_result):
        """Get AI-powered clinical interpretation using Gemini"""
        
        prompt = f"""
        You are a clinical pharmacologist reviewing a drug safety analysis.
        
        PATIENT: Age {patient.get('age', 'Unknown')}, Allergies: {patient.get('allergies', 'None')}
        CURRENT MEDICATIONS: {[m.get('name', str(m)) if isinstance(m, dict) else str(m) for m in patient.get('current_medications', [])]}
        MEDICAL HISTORY: {[h.get('condition', str(h)) if isinstance(h, dict) else str(h) for h in patient.get('medical_history', [])]}
        RENAL FUNCTION: {patient.get('vitals', {}).get('renal_function', 'Normal')}
        
        PROPOSED DRUG: {drug_name} ({dosage})
        
        SAFETY ANALYSIS:
        - Safety Score: {safety_report['safety_score']}/100
        - Risk Level: {safety_report['overall_risk']}
        - Critical Alerts: {len(safety_report['critical_alerts'])}
        - Major Warnings: {len(safety_report['major_warnings'])}
        
        SIMULATION RESULT:
        - Predicted Efficacy: {simulation_result['efficacy_score']}%
        - Outcome: {simulation_result['outcome']}
        
        Provide a concise (3-4 sentences) clinical interpretation explaining:
        1. Whether this drug is appropriate for this specific patient
        2. The most important risk factor to monitor
        3. One actionable recommendation
        4. PREDICTED NEUROLOGICAL IMPACT: Briefly mention sedation, cognitive effects, or seizure threshold risks.
        
        Be direct and practical.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return None
    
    def analyze_timeline_report(self, file_path, event_context=None):
        """
        Analyze uploaded medical report/image using Gemini Vision API.
        Extracts key medical information, findings, and recommendations.
        
        Args:
            file_path: Path to the uploaded file (image or PDF)
            event_context: Optional dict with event details (date, event_type, notes)
        
        Returns:
            dict with analysis results including summary, findings, and recommendations
        """
        if not self.model or not GOOGLE_API_KEY:
            return {
                "error": "AI analysis not available - API key not configured",
                "summary": "File uploaded successfully but AI analysis unavailable"
            }
        
        try:
            # Upload the file to Gemini
            uploaded_file = genai.upload_file(file_path)
            
            # Create analysis prompt with context
            context_info = ""
            if event_context:
                context_info = f"""
                CONTEXT FROM TIMELINE EVENT:
                - Event Type: {event_context.get('event', 'Medical Event')}
                - Date: {event_context.get('date', 'Not specified')}
                - Notes: {event_context.get('notes', 'None')}
                """
            
            prompt = f"""You are an expert medical AI assistant specializing in analyzing patient medical documents, including radiological imaging studies (CT scans, X-rays, MRIs).
            
            {context_info}
            
            Carefully analyze this medical report/image and extract detailed information:
            
            1. **Document Type**: What kind of medical document is this? 
               - If it's an imaging study, specify: CT scan, X-ray, MRI, Ultrasound, etc.
               - Note the body region/anatomy being examined
            
            2. **Key Findings**: List ALL important medical findings. For imaging studies:
               - Describe location (anatomical region, laterality)
               - Size and measurements where visible
               - Density/intensity characteristics
               - Any masses, lesions, nodules, abnormalities
               - Organ abnormalities
               - Bone/skeletal findings
               - Vascular findings
            
            3. **Abnormal Values / Pathological Findings**: Highlight concerning findings that need clinical attention:
               - Classify severity: CRITICAL, SIGNIFICANT, MILD, INCIDENTAL
               - Describe what makes them abnormal
               - If this is a CT scan showing disease, identify the pathology
            
            4. **Clinical Summary**: Provide a concise but comprehensive 3-4 sentence clinical interpretation.
            
            5. **Recommendations**: Clinical follow-up actions based on findings.
            
            6. **Comparison**: Note if comparison with prior studies is needed/mentioned.
            
            Format your response as JSON with these exact keys:
            {{
                "document_type": "CT Scan - [Body Region]" or appropriate type,
                "imaging_modality": "CT/MRI/X-ray/Ultrasound/Other/N/A",
                "body_region": "anatomical region examined",
                "key_findings": ["detailed finding 1", "detailed finding 2", ...],
                "abnormal_values": ["SEVERITY: description of abnormality 1", ...],
                "pathological_diagnosis": "primary pathological finding if evident",
                "summary": "comprehensive clinical interpretation",
                "recommendations": ["recommendation 1", "recommendation 2", ...],
                "urgency_level": "ROUTINE/URGENT/EMERGENT",
                "comparison_needed": true/false
            }}
            
            Be thorough and clinically precise. For CT scans, pay special attention to lesions, masses, calcifications, fluid collections, and organ abnormalities.
            """
            
            # Generate analysis using key manager (auto-rotates on quota errors)
            response = self.key_manager.generate_content([uploaded_file, prompt])
            
            # Parse JSON response
            import json
            import re
            
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: return raw text
                    return {
                        "document_type": "Medical Document",
                        "summary": response_text,
                        "key_findings": [],
                        "abnormal_values": [],
                        "recommendations": [],
                        "raw_analysis": response_text
                    }
            
            try:
                analysis = json.loads(json_str)
                
                # Ensure all required fields exist
                analysis.setdefault("document_type", "Medical Document")
                analysis.setdefault("key_findings", [])
                analysis.setdefault("abnormal_values", [])
                analysis.setdefault("summary", "Analysis completed")
                analysis.setdefault("recommendations", [])
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "document_type": "Medical Document",
                    "summary": response_text[:500],
                    "key_findings": [],
                    "abnormal_values": [],
                    "recommendations": [],
                    "raw_analysis": response_text
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "summary": f"Analysis failed: {str(e)}",
                "document_type": "Unknown",
                "key_findings": [],
                "abnormal_values": [],
                "recommendations": []
            }


ai_engine = AIEngine()
