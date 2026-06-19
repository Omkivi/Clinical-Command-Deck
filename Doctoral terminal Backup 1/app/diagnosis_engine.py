import math
import os

class DiagnosisEngine:
    def __init__(self, knowledge_path: str = None, unknown_symptom_penalty: float = 0.1, use_ai_council: bool = True):
        """Initialize the DiagnosisEngine.

        Parameters
        ----------
        knowledge_path: str, optional
            Path to the JSON knowledge base. If ``None`` the engine will look for
            ``data/medical_knowledge.json`` relative to the project root.
        unknown_symptom_penalty: float, default 0.2
            Probability multiplier applied when an observed symptom is not present
            in a disease's symptom list. Replaces the previous hard-coded ``0.05`` penalty.
        """
        import os
        from .knowledge_loader import load_knowledge

        if knowledge_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            knowledge_path = os.path.join(project_root, "data", "medical_knowledge.json")

        self.diseases = load_knowledge(knowledge_path)
        self.unknown_symptom_penalty = unknown_symptom_penalty
        self.use_ai_council = use_ai_council
        
        # Initialize AI Council if enabled
        self.council = None
        if use_ai_council:
            try:
                from .ai_council import ai_council
                self.council = ai_council
            except Exception as e:
                print(f"AI Council unavailable: {e}")
                self.use_ai_council = False

    def diagnose(self, observed_symptoms, patient=None):
        print(f"DEBUG: diagnose called with symptoms={observed_symptoms}, patient={patient is not None}")
        """
        Advanced Bayesian Diagnostic Engine with AI Council Enhancement
        
        FLOW (Option 1: Bayesian First, AI Validation):
        1. Run Bayesian inference for precise probabilities (0.1%, 0.4%, etc.)
        2. Enhance with AI Council clinical insights (optional)
        3. Return combined results
        
        Returns structured output:
        - Primary Diagnosis: Highest probability with strong match
        - Differential Diagnoses: Top 2-4 alternatives
        - Other Considerations: Lower probability conditions worth noting
        """
        if not observed_symptoms:
            return {
                "primary": None,
                "differential": [],
                "other_considerations": [],
                "total_evaluated": 0
            }
        
        # STEP 1: BAYESIAN ENGINE FIRST (for precise probabilities)
        print("[DIAGNOSIS] Running Bayesian inference for precise probabilities...")
        bayesian_results = self._run_bayesian_inference(observed_symptoms)
        
        # If Bayesian found no matches, fall back to pure AI
        if bayesian_results.get("primary") is None:
            print("[DIAGNOSIS] Bayesian found no matches, falling back to AI...")
            if self.use_ai_council and self.council:
                try:
                    return self._diagnose_with_council(observed_symptoms, patient=patient)
                except Exception as e:
                    print(f"AI Council also failed: {e}")
            
            # Ultimate fallback
            ai_suggestions = self._llm_fallback(observed_symptoms)
            return {
                "primary": None,
                "differential": [],
                "other_considerations": [],
                "total_evaluated": bayesian_results.get("total_evaluated", 0),
                "ai_fallback": ai_suggestions
            }
        
        # STEP 2: ENHANCE WITH AI COUNCIL (for clinical insights)
        if self.use_ai_council and self.council:
            try:
                print("[DIAGNOSIS] Enhancing with AI Council clinical insights...")
                bayesian_results = self._enhance_with_council(bayesian_results, observed_symptoms, patient)
            except Exception as e:
                print(f"AI Council enhancement failed: {e}. Using pure Bayesian results.")
                bayesian_results["council_error"] = str(e)
        
        bayesian_results["method"] = "Bayesian + AI Council" if self.use_ai_council else "Bayesian"
        return bayesian_results
    
    def _run_bayesian_inference(self, observed_symptoms):
        """Run pure Bayesian inference for precise probabilities."""
        
        # 1. Calculate scores for each disease
        total_probability_space = 0
        disease_scores = []

        for disease_name, data in self.diseases.items():
            prior = data['base_rate']
            likelihood = 1.0
            
            matched_symptoms = []
            unmatched_symptoms = []
            disease_specific_symptoms = list(data['symptoms'].keys())
            
            # Calculate Likelihood: P(Symptoms | Disease)
            for symptom in observed_symptoms:
                if symptom in data['symptoms']:
                    p_symptom_given_disease = data['symptoms'][symptom]
                    likelihood *= p_symptom_given_disease
                    matched_symptoms.append(symptom)
                else:
                    # Symptom not associated with this disease - penalize heavily
                    likelihood *= self.unknown_symptom_penalty
                    unmatched_symptoms.append(symptom)
            
            # Calculate "match quality" - how well do symptoms align?
            match_ratio = len(matched_symptoms) / len(observed_symptoms) if observed_symptoms else 0
            
            # Calculate unnormalized posterior
            unnormalized_posterior = likelihood * prior
            
            # Only consider diseases with decent match quality
            # OR critical conditions with any match
            is_critical = data.get("is_critical", False)
            min_match_threshold = 0.25 if not is_critical else 0.15
            
            if match_ratio >= min_match_threshold or (is_critical and matched_symptoms):
                disease_scores.append({
                    'name': disease_name,
                    'score': unnormalized_posterior,
                    'matched_symptoms': matched_symptoms,
                    'unmatched_symptoms': unmatched_symptoms,
                    'match_ratio': match_ratio,
                    'is_critical': is_critical,
                    'key_symptoms': disease_specific_symptoms[:5]  # Top 5 characteristic symptoms
                })
                total_probability_space += unnormalized_posterior

        # 2. Normalize probabilities
        if total_probability_space == 0:
            return {
                "primary": None,
                "differential": [],
                "other_considerations": [],
                "total_evaluated": len(disease_scores)
            }

        for disease in disease_scores:
            disease['probability'] = (disease['score'] / total_probability_space) * 100

        # 3. Sort by probability × match quality (highest first)
        disease_scores.sort(key=lambda x: x['probability'] * x['match_ratio'], reverse=True)

        # 4. Structure the output for clinical usefulness
        results = {
            "primary": None,
            "differential": [],
            "other_considerations": [],
            "total_evaluated": len(disease_scores),
            "warning": None
        }

        # Primary diagnosis - highest probability
        if disease_scores:
            top = disease_scores[0]
            results["primary"] = {
                "name": top['name'],
                "probability": round(top['probability'], 2),
                "confidence": self._calculate_confidence(top['match_ratio'], top['probability']),
                "is_critical": top['is_critical'],
                "matched_symptoms": top['matched_symptoms'],
                "key_diagnostic_symptoms": top['key_symptoms'],
                "clinical_note": self._get_clinical_note(top['name'], top['matched_symptoms'], top['unmatched_symptoms'])
            }

        # Differential diagnoses - next 2-4 strong alternatives
        differential_candidates = disease_scores[1:5]
        for disease in differential_candidates:
            # Show more granular probabilities - include if > 0.5% OR critical with any probability
            if disease['probability'] > 0.5 or disease['is_critical']:
                results["differential"].append({
                    "name": disease['name'],
                    "probability": round(disease['probability'], 2),
                    "is_critical": disease['is_critical'],
                    "matched_symptoms": disease['matched_symptoms'],
                    "why_consider": self._get_differential_reason(disease)
                })

        # Other considerations - lower probability but worth mentioning
        other_candidates = disease_scores[5:10]
        for disease in other_candidates:
            # Include all with > 0.1% probability for transparency
            if disease['is_critical'] or disease['probability'] > 0.1:
                results["other_considerations"].append({
                    "name": disease['name'],
                    "probability": round(disease['probability'], 2),
                    "is_critical": disease['is_critical'],
                    "note": f"Consider if {', '.join(disease['key_symptoms'][:3])} develop"
                })

        # Add warning if critical condition in differential
        critical_in_differential = [d for d in results["differential"] if d['is_critical']]
        if critical_in_differential:
            results["warning"] = f"⚠️ {len(critical_in_differential)} life-threatening condition(s) in differential - rule out immediately"

        return results
    
    def _enhance_with_council(self, bayesian_results, observed_symptoms, patient=None):
        """Enhance Bayesian results with AI Council clinical insights.
        
        The AI Council provides:
        - Clinical validation of the diagnosis
        - Additional insights and recommendations
        - Consensus from multiple AI models
        
        But probabilities remain from Bayesian calculation.
        """
        # Prepare case data for council
        case_data = {
            "symptoms": observed_symptoms,
            "symptom_count": len(observed_symptoms),
            "primary_diagnosis": bayesian_results.get("primary", {}).get("name"),
            "primary_probability": bayesian_results.get("primary", {}).get("probability"),
            "differentials": [d.get("name") for d in bayesian_results.get("differential", [])]
        }
        
        # Run council deliberation
        verdict = self.council.deliberate(
            case_type="DIAGNOSIS",
            case_data=case_data,
            patient=patient
        )
        
        # Enhance primary diagnosis with council insights
        if bayesian_results.get("primary"):
            primary = bayesian_results["primary"]
            
            # Add council consensus info
            primary["council_verdict"] = verdict.final_verdict
            primary["council_consensus"] = round(verdict.consensus_score * 100, 1)
            
            # Add unanimous recommendations if available
            if verdict.unanimous_recommendations:
                primary["ai_recommendations"] = verdict.unanimous_recommendations[:3]
            
            # Add unanimous concerns if available
            if verdict.unanimous_concerns:
                primary["ai_concerns"] = verdict.unanimous_concerns[:3]
        
        # Enhance differentials with council insights
        for diff in bayesian_results.get("differential", []):
            # Check if any model specifically mentioned this diagnosis
            for opinion in verdict.model_opinions:
                if diff["name"].lower() in opinion.reasoning.lower():
                    diff["ai_supported"] = True
                    break
        
        # Add full council deliberation for transparency
        bayesian_results["council_deliberation"] = verdict.to_dict()
        bayesian_results["models_consulted"] = [op.model_name for op in verdict.model_opinions]
        
        # Add AI-specific warnings
        if verdict.consensus_score < 0.6:
            if bayesian_results.get("warning"):
                bayesian_results["warning"] += " | ⚠️ Low AI consensus - recommend additional review"
            else:
                bayesian_results["warning"] = "⚠️ Low AI consensus - recommend additional review"
        
        return bayesian_results

    def _calculate_confidence(self, match_ratio, probability):
        """Calculate clinical confidence level"""
        if match_ratio > 0.8 and probability > 60:
            return "High"
        elif match_ratio > 0.6 and probability > 40:
            return "Moderate"
        elif match_ratio > 0.4 or probability > 20:
            return "Low"
        else:
            return "Very Low"

    def _get_clinical_note(self, disease_name, matched, unmatched):
        """Generate clinical note for the diagnosis"""
        note_parts = []
        
        if matched:
            note_parts.append(f"Presentation consistent with {len(matched)} characteristic symptoms")
        
        if unmatched:
            note_parts.append(f"Note: {len(unmatched)} atypical symptoms present")
        
        # Disease-specific clinical pearls
        clinical_pearls = {
            "Bacterial Meningitis": "Urgent: Lumbar puncture and IV antibiotics required. Do not delay treatment.",
            "Myocardial Infarction": "STEMI protocol: ECG, troponins, immediate cardiology consult",
            "Stroke (Ischemic)": "Time-critical: CT scan, consider thrombolysis within 4.5 hours",
            "Pulmonary Embolism": "Wells score, D-dimer, CT angiography if high suspicion",
            "Appendicitis": "Surgical evaluation required. NPO, IV fluids, imaging",
            "Sepsis": "qSOFA score, blood cultures, early antibiotics (within 1 hour)",
            "Diabetic Ketoacidosis": "Check glucose, ketones, VBG. Insulin and fluid resuscitation",
            "Anaphylaxis": "IM epinephrine 0.3mg immediately. Monitor for biphasic reaction",
            "Subarachnoid Hemorrhage": "CT head non-contrast. If negative with high suspicion: lumbar puncture"
        }
        
        if disease_name in clinical_pearls:
            note_parts.append(clinical_pearls[disease_name])
        
        return " | ".join(note_parts)

    def _get_differential_reason(self, disease):
        """Explain why this is in differential diagnosis"""
        if disease['is_critical']:
            return f"Life-threatening condition - must rule out despite {disease['probability']:.1f}% probability"
        elif disease['match_ratio'] > 0.7:
            return f"Strong symptom overlap ({len(disease['matched_symptoms'])} matching)"
        elif disease['probability'] > 15:
            return f"High probability ({disease['probability']:.1f}%) warrants consideration"
        else:
            return f"Possible alternative diagnosis"

    def get_all_symptoms(self):
        """Returns a unique list of all known symptoms for the UI autocomplete"""
        symptoms = set()
        for data in self.diseases.values():
            symptoms.update(data['symptoms'].keys())
        return sorted(list(symptoms))

    # -------------------------------------------------------------------------
    # AI Council Diagnosis – Multi-model deliberation for maximum accuracy
    # -------------------------------------------------------------------------
    def _diagnose_with_council(self, observed_symptoms, patient=None):
        """Use AI Council (4-5 models) to deliberate and reach consensus diagnosis.
        
        Parameters
        ----------
        observed_symptoms : list of str
            Symptoms reported by the patient
        patient : dict, optional
            Patient context (age, medical history, etc.)
            
        Returns
        -------
        dict
            Diagnosis result with council deliberation details
        """
        # Prepare case data for council
        case_data = {
            "symptoms": observed_symptoms,
            "symptom_count": len(observed_symptoms),
            "primary_diagnosis": None,
            "differentials": []
        }
        
        # Run full council deliberation
        print(f"[DIAGNOSIS] Consulting AI Council for {len(observed_symptoms)} symptoms...")
        verdict = self.council.deliberate(
            case_type="DIAGNOSIS",
            case_data=case_data,
            patient=patient
        )
        
        # Map council verdict to diagnosis format
        return self._map_council_to_diagnosis(verdict, observed_symptoms)
    
    def _map_council_to_diagnosis(self, verdict, observed_symptoms):
        """Convert Council verdict to diagnosis format.
        
        Now uses structured ranked_diagnoses from each model instead of 
        keyword matching, which provides proper probability differentiation.
        """
        
        # 1. Aggregate ranked diagnoses from all models
        # Each model provides explicit rankings with probabilities
        diagnosis_scores = {}
        diagnosis_rationales = {}
        model_count = len(verdict.model_opinions)
        
        for opinion in verdict.model_opinions:
            model_weight = opinion.confidence  # Use model confidence as weight
            
            # Parse ranked_diagnoses from this model
            ranked_dx = opinion.ranked_diagnoses
            if not ranked_dx:
                # Fallback to keyword matching if model didn't provide ranked diagnoses
                for disease_name in self.diseases.keys():
                    if disease_name.lower() in opinion.reasoning.lower():
                        current_score = diagnosis_scores.get(disease_name, 0.0)
                        # Give a flat score weighted by confidence
                        diagnosis_scores[disease_name] = current_score + (10 * model_weight)
                continue
            
            # Process explicit rankings with their probabilities
            for dx in ranked_dx:
                name = dx.get('name', '')
                probability = float(dx.get('probability', 0))
                rationale = dx.get('rationale', '')
                
                # Normalize diagnosis name - try to match to known diseases
                matched_name = self._match_disease_name(name)
                if matched_name:
                    # Weight the probability by model confidence
                    weighted_prob = probability * model_weight
                    current_score = diagnosis_scores.get(matched_name, 0.0)
                    diagnosis_scores[matched_name] = current_score + weighted_prob
                    
                    # Store best rationale for each diagnosis
                    if matched_name not in diagnosis_rationales or probability > 30:
                        diagnosis_rationales[matched_name] = rationale
        
        # 2. Sort by aggregated score (higher = more likely)
        ranked_diagnoses = sorted(diagnosis_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not ranked_diagnoses:
            return {
                "primary": None, "differential": [], "other_considerations": [], 
                "total_evaluated": len(verdict.model_opinions), "council_deliberation": verdict.to_dict(), 
                "method": "AI Council", "warning": "No diagnosis identified"
            }

        # 3. Calculate probabilities from aggregated scores
        # Normalize to 100%
        total_score = sum(score for _, score in ranked_diagnoses)
        
        def get_prob(score):
            return round((score / total_score) * 100, 1) if total_score > 0 else 0

        # Primary Diagnosis
        top_name, top_score = ranked_diagnoses[0]
        primary_prob = get_prob(top_score)
        
        primary_diagnosis = {
            "name": top_name,
            "probability": primary_prob,
            "confidence": "High" if primary_prob > 40 else ("Moderate" if primary_prob > 25 else "Low"),
            "is_critical": self.diseases.get(top_name, {}).get("is_critical", False),
            "matched_symptoms": observed_symptoms,
            "council_verdict": verdict.final_verdict,
            "consensus_score": round(verdict.consensus_score * 100, 1),
            "clinical_note": diagnosis_rationales.get(top_name, " | ".join(verdict.unanimous_recommendations[:2]) if verdict.unanimous_recommendations else "See deliberation")
        }
        
        # Differentials (next 3-4)
        differential = []
        for name, score in ranked_diagnoses[1:5]:
            prob = get_prob(score)
            if prob < 2:  # Skip very low probability conditions
                continue
            differential.append({
                "name": name,
                "probability": prob,
                "is_critical": self.diseases.get(name, {}).get("is_critical", False),
                "matched_symptoms": observed_symptoms,
                "why_consider": diagnosis_rationales.get(name, f"Supported by {model_count} model(s)")
            })

        # Other considerations (5th onwards with >1% probability)
        other_considerations = []
        for name, score in ranked_diagnoses[5:10]:
            prob = get_prob(score)
            if prob < 1:
                continue
            other_considerations.append({
                "name": name,
                "probability": prob,
                "is_critical": self.diseases.get(name, {}).get("is_critical", False),
                "note": diagnosis_rationales.get(name, "Lower probability consideration")
            })

        # Check for critical conditions in differential
        warning = None
        critical_conditions = [d for d in differential if d.get('is_critical')]
        if critical_conditions:
            warning = f"⚠️ {len(critical_conditions)} life-threatening condition(s) in differential - rule out immediately"
        elif verdict.consensus_score < 0.7:
            warning = "⚠️ AI-generated diagnosis - verify with clinical judgment"

        return {
            "primary": primary_diagnosis,
            "differential": differential,
            "other_considerations": other_considerations,
            "total_evaluated": len(verdict.model_opinions),
            "council_deliberation": verdict.to_dict(),
            "method": "AI Council",
            "warning": warning
        }
    
    def _match_disease_name(self, name):
        """Match a diagnosis name to known diseases in our knowledge base.
        
        Returns the matched disease name or None if no match.
        """
        if not name:
            return None
            
        name_lower = name.lower().strip()
        
        # Exact match first
        for disease in self.diseases.keys():
            if disease.lower() == name_lower:
                return disease
        
        # Partial match (disease name contained in the provided name)
        for disease in self.diseases.keys():
            if disease.lower() in name_lower or name_lower in disease.lower():
                return disease
        
        # Fuzzy matching for common variations
        name_variations = {
            "mi": "Myocardial Infarction",
            "heart attack": "Myocardial Infarction",
            "acute coronary syndrome": "Myocardial Infarction",
            "acs": "Myocardial Infarction",
            "pe": "Pulmonary Embolism",
            "aortic dissection": "Aortic Dissection",
            "dissection": "Aortic Dissection",
            "ua": "Unstable Angina",
            "angina": "Unstable Angina",
            "stroke": "Stroke (Ischemic)",
            "cva": "Stroke (Ischemic)",
            "ischemic stroke": "Stroke (Ischemic)",
            "hemorrhagic stroke": "Hemorrhagic Stroke",
            "sah": "Subarachnoid Hemorrhage",
            "meningitis": "Bacterial Meningitis",
            "sepsis": "Sepsis",
            "pneumonia": "Pneumonia",
            "copd": "COPD Exacerbation",
            "dka": "Diabetic Ketoacidosis",
            "dvt": "Deep Vein Thrombosis",
            "uti": "Urinary Tract Infection",
            "pyelonephritis": "Pyelonephritis",
            "appendicitis": "Appendicitis",
            "pancreatitis": "Pancreatitis",
            "cholecystitis": "Cholecystitis",
        }
        
        for variation, disease in name_variations.items():
            if variation in name_lower:
                if disease in self.diseases:
                    return disease
        
        # If no match found, return the original name (for display purposes)
        return name


    # -------------------------------------------------------------------------
    # AI Fallback – called when Bayesian engine yields no match or low confidence
    # -------------------------------------------------------------------------
    def _llm_fallback(self, observed_symptoms, patient_context: dict = None):
        """Query Gemini for differential suggestions when the local model is uncertain.

        Parameters
        ----------
        observed_symptoms : list of str
            Symptoms entered by the user.
        patient_context : dict, optional
            Additional patient info (age, sex, history).

        Returns
        -------
        dict
            {
                "suggestions": [{"name": str, "rationale": str}, ...],
                "source": "gemini"
            }
        """
        import os
        import json
        try:
            import google.generativeai as genai
        except ImportError:
            return {"suggestions": [], "source": "gemini", "error": "google-generativeai not installed"}

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {"suggestions": [], "source": "gemini", "error": "GOOGLE_API_KEY not set"}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        context_str = ""
        if patient_context:
            context_str = f"Patient context: {json.dumps(patient_context)}. "

        prompt = f"""You are a clinical decision support system.
{context_str}A patient presents with the following symptoms: {', '.join(observed_symptoms)}.

Provide up to 5 differential diagnoses ranked by likelihood. For each diagnosis include a short one‑sentence rationale.

Respond ONLY with valid JSON in this exact format (no markdown):
{{
  "suggestions": [
    {{"name": "<Diagnosis>", "rationale": "<Why this diagnosis>"}},
    ...
  ]
}}"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            return {"suggestions": data.get("suggestions", []), "source": "gemini"}
        except Exception as e:
            return {"suggestions": [], "source": "gemini", "error": str(e)}
    
    def recommend_tests(self, symptoms, diagnosis_results=None, patient=None):
        """
        Recommend specific lab tests and imaging studies based on symptoms and diagnosis
        
        Args:
            symptoms: List of observed symptoms
            diagnosis_results: Optional diagnosis results from diagnose()
            patient: Optional patient data (age, sex, medical history)
        
        Returns:
            {
                "essential_tests": [...],
                "recommended_tests": [...],
                "imaging_studies": [...],
                "rationale": "Why these tests are needed"
            }
        """
        essential_tests = []
        recommended_tests = []
        imaging_studies = []
        
        # Disease-to-test mapping
        disease_test_map = {
            # NEUROLOGICAL
            "Bacterial Meningitis": {
                "essential": ["CSF Analysis (Lumbar Puncture)", "Blood Culture", "CBC"],
                "recommended": ["CRP", "Procalcitonin"],
                "imaging": ["CT Head (before LP if focal signs)"]
            },
            "Stroke (Ischemic)": {
                "essential": ["CT Head Non-Contrast", "CBC", "CMP", "Coagulation Panel"],
                "recommended": ["Lipid Panel", "HbA1c", "ECG"],
                "imaging": ["CT Head", "MRI Brain with DWI", "CT Angiography"]
            },
            "Subarachnoid Hemorrhage": {
                "essential": ["CT Head Non-Contrast", "CSF Analysis if CT negative"],
                "recommended": [],
                "imaging": ["CT Head", "CT Angiography"]
            },
            "Migraine": {
                "essential": [],
                "recommended": ["CBC", "Vitamin D", "Magnesium"],
                "imaging": ["MRI Brain if atypical features"]
            },
            
            # CARDIOVASCULAR
            "Myocardial Infarction": {
                "essential": ["Troponin I/T (Serial)", "ECG", "CK-MB"],
                "recommended": ["CBC", "CMP", "Lipid Panel", "BNP"],
                "imaging": ["Chest X-Ray", "Echocardiogram"]
            },
            "Pulmonary Embolism": {
                "essential": ["D-Dimer", "CT Pulmonary Angiography"],
                "recommended": ["CBC", "Coagulation Panel", "BNP"],
                "imaging": ["CT Chest with PE Protocol", "Chest X-Ray"]
            },
            "Heart Failure": {
                "essential": ["BNP or NT-proBNP", "Echocardiogram", "ECG"],
                "recommended": ["CBC", "CMP", "TSH", "Lipid Panel"],
                "imaging": ["Chest X-Ray", "Echocardiogram"]
            },
            
            # RESPIRATORY
            "Pneumonia": {
                "essential": ["Chest X-Ray", "CBC", "Blood Culture if severe"],
                "recommended": ["CRP", "Procalcitonin", "Sputum Culture"],
                "imaging": ["Chest X-Ray"]
            },
            "COVID-19": {
                "essential": ["COVID-19 PCR or Antigen Test"],
                "recommended": ["CBC", "CMP", "D-Dimer", "Ferritin"],
                "imaging": ["Chest X-Ray if moderate/severe"]
            },
            "Tuberculosis": {
                "essential": ["Chest X-Ray", "Sputum AFB Smear & Culture", "Tuberculin Skin Test or IGRA"],
                "recommended": ["CBC", "HIV Test"],
                "imaging": ["Chest X-Ray", "CT Chest"]
            },
            
            # GASTROINTESTINAL
            "Appendicitis": {
                "essential": ["CBC", "Ultrasound or CT Abdomen/Pelvis"],
                "recommended": ["CRP", "Urinalysis"],
                "imaging": ["Ultrasound Abdomen", "CT Abdomen/Pelvis"]
            },
            "Pancreatitis": {
                "essential": ["Lipase", "Amylase", "CMP"],
                "recommended": ["CBC", "Triglycerides", "Calcium"],
                "imaging": ["CT Abdomen/Pelvis", "Ultrasound Abdomen"]
            },
            "Cholecystitis": {
                "essential": ["Ultrasound Abdomen", "CBC", "CMP"],
                "recommended": ["Lipase"],
                "imaging": ["Ultrasound Abdomen", "HIDA Scan"]
            },
            
            # ENDOCRINE
            "Hypothyroidism": {
                "essential": ["TSH", "Free T4"],
                "recommended": ["Thyroid Antibodies", "CBC", "Lipid Panel"],
                "imaging": []
            },
            "Hyperthyroidism": {
                "essential": ["TSH", "Free T4", "Free T3"],
                "recommended": ["Thyroid Antibodies", "Radioactive Iodine Uptake"],
                "imaging": ["Thyroid Ultrasound"]
            },
            "Diabetic Ketoacidosis": {
                "essential": ["Glucose", "Ketones (Blood or Urine)", "VBG", "CMP"],
                "recommended": ["CBC", "Phosphate", "Magnesium"],
                "imaging": []
            },
            "Hypoglycemia": {
                "essential": ["Glucose (Fingerstick and serum)"],
                "recommended": ["Insulin Level", "C-Peptide", "HbA1c"],
                "imaging": []
            },
            
            # INFECTIOUS
            "Sepsis": {
                "essential": ["Blood Culture", "CBC", "CMP", "Lactate"],
                "recommended": ["Procalcitonin", "Coagulation Panel", "Urinalysis"],
                "imaging": ["Chest X-Ray", "Source-dependent imaging"]
            },
            "Urinary Tract Infection": {
                "essential": ["Urinalysis", "Urine Culture"],
                "recommended": ["CBC if systemic symptoms"],
                "imaging": []
            },
            "Pyelonephritis": {
                "essential": ["Urinalysis", "Urine Culture", "CBC", "CMP"],
                "recommended": ["Blood Culture"],
                "imaging": ["Ultrasound Kidneys", "CT Abdomen/Pelvis if complicated"]
            },
            
            # NUTRITIONAL DEFICIENCIES (NEW!)
            "Vitamin D Deficiency": {
                "essential": ["Vitamin D, 25-OH"],
                "recommended": ["PTH", "Calcium", "Phosphate"],
                "imaging": []
            },
            "Vitamin B12 Deficiency": {
                "essential": ["Vitamin B12", "CBC"],
                "recommended": ["Folate", "Methylmalonic Acid", "Homocysteine"],
                "imaging": []
            },
            "Iron Deficiency Anemia": {
                "essential": ["CBC", "Iron Panel (Iron, TIBC, Ferritin)"],
                "recommended": ["Stool Occult Blood", "Colonoscopy/Endoscopy if indicated"],
                "imaging": []
            },
        }
        
        # Symptom-to-test mapping (when diagnosis is uncertain)
        symptom_test_map = {
            "Chest Pain": {
                "essential": ["ECG", "Troponin"],
                "recommended": ["Chest X-Ray", "D-Dimer"],
            },
            "Headache": {
                "essential": ["CBC", "CMP"],
                "recommended": ["CT Head if red flags", "Vitamin D"],
            },
            "Fatigue": {
                "essential": ["CBC", "TSH", "CMP"],
                "recommended": ["Vitamin D", "Vitamin B12", "Iron Panel", "HbA1c"],
            },
            "Fever": {
                "essential": ["CBC", "Blood Culture if high fever"],
                "recommended": ["CRP", "Urinalysis"],
            },
            "Abdominal Pain": {
                "essential": ["CBC", "CMP", "Lipase"],
                "recommended": ["Urinalysis", "Pregnancy Test (females)"],
                "imaging": ["Ultrasound or CT Abdomen based on location"],
            },
            "Shortness of Breath": {
                "essential": ["Chest X-Ray", "Pulse Oximetry/ABG"],
                "recommended": ["BNP", "D-Dimer", "ECG"],
            },
            "Weight Loss": {
                "essential": ["CBC", "CMP", "TSH", "HbA1c"],
                "recommended": ["Malignancy workup if unexplained"],
            },
        }
        
        # Get tests based on primary and differential diagnoses
        if diagnosis_results:
            primary = diagnosis_results.get('primary')
            differential = diagnosis_results.get('differential', [])
            
            # Add tests for primary diagnosis
            if primary and primary['name'] in disease_test_map:
                tests = disease_test_map[primary['name']]
                essential_tests.extend(tests.get('essential', []))
                recommended_tests.extend(tests.get('recommended', []))
                imaging_studies.extend(tests.get('imaging', []))
            
            # Add tests for differential diagnoses
            for diff_dx in differential[:2]:  # Top 2 differentials
                if diff_dx['name'] in disease_test_map:
                    tests = disease_test_map[diff_dx['name']]
                    recommended_tests.extend(tests.get('essential', []))
                    recommended_tests.extend(tests.get('recommended', []))
                    imaging_studies.extend(tests.get('imaging', []))
        
        # Get tests based on symptoms
        for symptom in symptoms:
            if symptom in symptom_test_map:
                tests = symptom_test_map[symptom]
                # Add if not already present
                for test in tests.get('essential', []):
                    if test not in essential_tests:
                        essential_tests.append(test)
                for test in tests.get('recommended', []):
                    if test not in recommended_tests:
                        recommended_tests.append(test)
                for test in tests.get('imaging', []):
                    if test not in imaging_studies:
                        imaging_studies.append(test)
        
        # Remove duplicates
        essential_tests = list(set(essential_tests))
        recommended_tests = list(set(recommended_tests))
        imaging_studies = list(set(imaging_studies))
        
        # CHECK FOR EXISTING TESTS - Skip tests already performed within 30 days
        if patient:
            from .patient_data_aggregator import patient_data_aggregator
            
            filtered_essential = []
            filtered_recommended = []
            filtered_imaging = []
            existing_tests_info = []
            
            # Check essential tests
            for test in essential_tests:
                existing = patient_data_aggregator.has_existing_test(patient, test)
                if existing.get('exists'):
                    existing_tests_info.append(f"{test} (done {existing['age_days']} days ago)")
                else:
                    filtered_essential.append(test)
            
            # Check recommended tests
            for test in recommended_tests:
                existing = patient_data_aggregator.has_existing_test(patient, test)
                if existing.get('exists'):
                    existing_tests_info.append(f"{test} (done {existing['age_days']} days ago)")
                else:
                    filtered_recommended.append(test)
            
            # Check imaging
            for test in imaging_studies:
                existing = patient_data_aggregator.has_existing_test(patient, test)
                if existing.get('exists'):
                    existing_tests_info.append(f"{test} (done {existing['age_days']} days ago)")
                else:
                    filtered_imaging.append(test)
            
            essential_tests = filtered_essential
            recommended_tests = filtered_recommended
            imaging_studies = filtered_imaging
        
        # Build rationale
        rationale_parts = []
        if diagnosis_results and diagnosis_results.get('primary'):
            rationale_parts.append(f"Testing to confirm {diagnosis_results['primary']['name']}")
        if diagnosis_results and diagnosis_results.get('differential'):
            rationale_parts.append("and rule out differential diagnoses")
        
        # Add context from patient data
        if patient:
            from .patient_data_aggregator import patient_data_aggregator
            abnormal_labs = patient_data_aggregator.get_abnormal_labs(patient)
            if abnormal_labs:
                rationale_parts.append(f"Following up on {len(abnormal_labs)} abnormal lab values")
        
        rationale = ". ".join(rationale_parts) if rationale_parts else "Comprehensive diagnostic workup"
        
        result = {
            "essential_tests": essential_tests,
            "recommended_tests": recommended_tests,
            "imaging_studies": imaging_studies,
            "rationale": rationale
        }
        
        # Add existing tests info if any were skipped
        if patient and existing_tests_info:
            result["existing_tests"] = existing_tests_info
            result["note"] = f"Skipped {len(existing_tests_info)} tests already performed within 30 days"
        
        return result
    
    # ============ LAB-TO-DISEASE MAPPING ============
    # Maps lab abnormalities to diseases they support/contradict
    LAB_DISEASE_ADJUSTMENTS = {
        # Hematology
        'wbc': {
            'high': {'Sepsis': 1.5, 'Pneumonia': 1.4, 'Appendicitis': 1.4, 'Cholecystitis': 1.3, 
                     'Pyelonephritis': 1.4, 'Bacterial Meningitis': 1.5, 'Urinary Tract Infection': 1.3},
            'low': {'Sepsis': 1.3, 'Viral Meningitis': 1.2, 'HIV/AIDS': 1.5}
        },
        'hemoglobin': {
            'low': {'Dehydration': 0.7, 'Heart Failure': 1.3, 'Hypothyroidism': 1.2}
        },
        'platelets': {
            'low': {'Sepsis': 1.4, 'Dengue Fever': 1.8, 'Malaria': 1.5},
            'high': {'Sepsis': 1.2}
        },
        # Cardiac
        'troponin': {
            'high': {'Myocardial Infarction': 2.5, 'Pulmonary Embolism': 1.4, 'Heart Failure': 1.3, 'Sepsis': 1.2}
        },
        'bnp': {
            'high': {'Heart Failure': 2.0, 'Pulmonary Embolism': 1.3}
        },
        'nt-probnp': {
            'high': {'Heart Failure': 2.0, 'Pulmonary Embolism': 1.3}
        },
        # Renal
        'creatinine': {
            'high': {'Pyelonephritis': 1.3, 'Dehydration': 1.4, 'Heart Failure': 1.2, 'Sepsis': 1.3}
        },
        'egfr': {
            'low': {'Pyelonephritis': 1.3, 'Dehydration': 1.3}
        },
        'bun': {
            'high': {'Dehydration': 1.5, 'Pyelonephritis': 1.2}
        },
        # Metabolic
        'glucose': {
            'high': {'Diabetic Ketoacidosis': 2.5, 'Sepsis': 1.2},
            'low': {'Hypoglycemia': 3.0, 'Sepsis': 1.2}
        },
        'hba1c': {
            'high': {'Diabetic Ketoacidosis': 1.5}
        },
        # Liver
        'alt': {
            'high': {'Cholecystitis': 1.3, 'Hepatitis': 1.8}
        },
        'ast': {
            'high': {'Cholecystitis': 1.3, 'Hepatitis': 1.6, 'Myocardial Infarction': 1.2}
        },
        'bilirubin': {
            'high': {'Cholecystitis': 1.5, 'Pancreatitis': 1.3}
        },
        # Pancreatic
        'lipase': {
            'high': {'Pancreatitis': 3.0, 'Cholecystitis': 1.3}
        },
        'amylase': {
            'high': {'Pancreatitis': 2.5}
        },
        # Inflammatory
        'crp': {
            'high': {'Sepsis': 1.4, 'Pneumonia': 1.3, 'Appendicitis': 1.4, 'Pancreatitis': 1.3}
        },
        'procalcitonin': {
            'high': {'Sepsis': 1.8, 'Bacterial Meningitis': 1.5, 'Pneumonia': 1.4}
        },
        'esr': {
            'high': {'Sepsis': 1.2, 'Pneumonia': 1.2}
        },
        # Coagulation
        'd-dimer': {
            'high': {'Pulmonary Embolism': 1.8, 'Deep Vein Thrombosis': 1.8, 'Sepsis': 1.3}
        },
        # Thyroid
        'tsh': {
            'high': {'Hypothyroidism': 2.5},
            'low': {'Hyperthyroidism': 2.5}
        },
        't4': {
            'high': {'Hyperthyroidism': 2.0},
            'low': {'Hypothyroidism': 2.0}
        },
        't3': {
            'high': {'Hyperthyroidism': 1.8},
            'low': {'Hypothyroidism': 1.5}
        },
        # Electrolytes
        'sodium': {
            'low': {'Dehydration': 1.2, 'Heart Failure': 1.3},
            'high': {'Dehydration': 1.5}
        },
        'potassium': {
            'high': {'Diabetic Ketoacidosis': 1.3},
            'low': {'Dehydration': 1.3}
        },
        # Lactate
        'lactate': {
            'high': {'Sepsis': 2.0, 'Bowel Obstruction': 1.5}
        }
    }
    
    # Radiology finding to disease mapping
    RADIOLOGY_DISEASE_CLUES = {
        'consolidation': ['Pneumonia', 'COVID-19', 'Tuberculosis'],
        'infiltrate': ['Pneumonia', 'COVID-19'],
        'opacity': ['Pneumonia', 'Pulmonary Embolism'],
        'effusion': ['Heart Failure', 'Pneumonia', 'Pulmonary Embolism'],
        'cardiomegaly': ['Heart Failure'],
        'mass': ['Tuberculosis'],
        'nodule': ['Tuberculosis'],
        'pneumothorax': ['Asthma Attack', 'COPD Exacerbation'],
        'edema': ['Heart Failure'],
        'appendicitis': ['Appendicitis'],
        'cholecystitis': ['Cholecystitis'],
        'pancreatitis': ['Pancreatitis'],
        'obstruction': ['Bowel Obstruction'],
        'thrombus': ['Deep Vein Thrombosis', 'Pulmonary Embolism'],
        'embolism': ['Pulmonary Embolism'],
        'stroke': ['Stroke (Ischemic)'],
        'hemorrhage': ['Subarachnoid Hemorrhage', 'Stroke (Ischemic)'],
        'fracture': [],
        'abscess': ['Appendicitis', 'Cholecystitis', 'Pyelonephritis']
    }
    
    def _get_lab_adjustments(self, patient):
        """
        Get disease probability adjustments based on lab values from PatientDataAggregator
        
        Returns:
            dict: {disease_name: multiplier} for probability adjustments
        """
        from .patient_data_aggregator import patient_data_aggregator
        
        adjustments = {}
        lab_values = patient_data_aggregator.get_latest_lab_values(patient)
        abnormal_labs = patient_data_aggregator.get_abnormal_labs(patient)
        
        for lab in abnormal_labs:
            test_name = lab.get('test', '').lower()
            # Normalize common test name variations
            normalized_name = test_name.replace(' ', '').replace('-', '').replace('_', '')
            
            # Find matching lab in our mapping
            for lab_key, directions in self.LAB_DISEASE_ADJUSTMENTS.items():
                if lab_key in normalized_name or normalized_name in lab_key:
                    # Determine if high or low based on status or value context
                    status = lab.get('status', 'abnormal').lower()
                    direction = 'high' if 'high' in status or 'elevated' in status else 'low'
                    
                    if direction in directions:
                        for disease, multiplier in directions[direction].items():
                            if disease not in adjustments:
                                adjustments[disease] = multiplier
                            else:
                                adjustments[disease] *= multiplier
                    break
        
        return adjustments
    
    def _get_radiology_clues(self, patient):
        """
        Extract diagnostic clues from radiology findings
        
        Returns:
            tuple: (list of clue strings, dict of diseases supported by imaging)
        """
        from .patient_data_aggregator import patient_data_aggregator
        
        clues = []
        supported_diseases = {}
        
        radiology_findings = patient_data_aggregator.get_radiology_findings(patient)
        
        for finding in radiology_findings:
            finding_text = finding.get('finding', '').lower()
            modality = finding.get('modality', '')
            date = finding.get('date', '')
            
            # Search for known patterns
            for pattern, diseases in self.RADIOLOGY_DISEASE_CLUES.items():
                if pattern in finding_text:
                    clue_text = f"{modality}: {pattern} noted" + (f" ({date})" if date else "")
                    if clue_text not in clues:
                        clues.append(clue_text)
                    
                    for disease in diseases:
                        if disease not in supported_diseases:
                            supported_diseases[disease] = 1.5  # Imaging support multiplier
                        else:
                            supported_diseases[disease] *= 1.2  # Additional imaging evidence
        
        return clues, supported_diseases
    
    def _get_timeline_insights(self, patient):
        """
        Extract diagnostic insights from timeline medical events
        
        Returns:
            tuple: (list of insight strings, dict of diseases to consider)
        """
        from .patient_data_aggregator import patient_data_aggregator
        
        insights = []
        considerations = {}
        
        timeline_events = patient_data_aggregator.get_timeline_medical_events(patient)
        
        for event in timeline_events:
            event_text = event.get('event', '').lower()
            category = event.get('category', '')
            date = event.get('date', '')
            
            # Recent surgery - consider post-op complications
            if category == 'surgical':
                insights.append(f"Recent surgery: {event.get('event', '')} ({date})")
                considerations['Sepsis'] = considerations.get('Sepsis', 1.0) * 1.3
                considerations['Deep Vein Thrombosis'] = considerations.get('Deep Vein Thrombosis', 1.0) * 1.4
                considerations['Pulmonary Embolism'] = considerations.get('Pulmonary Embolism', 1.0) * 1.3
            
            # Recent infection - consider recurrence
            elif category == 'infectious':
                insights.append(f"Prior infection: {event.get('event', '')} ({date})")
                # Check for specific infections
                if 'uti' in event_text or 'urinary' in event_text:
                    considerations['Urinary Tract Infection'] = considerations.get('Urinary Tract Infection', 1.0) * 1.4
                    considerations['Pyelonephritis'] = considerations.get('Pyelonephritis', 1.0) * 1.3
                elif 'pneumonia' in event_text:
                    considerations['Pneumonia'] = considerations.get('Pneumonia', 1.0) * 1.3
            
            # Hospitalization - flag severity
            elif category == 'hospitalization':
                insights.append(f"Prior hospitalization: {event.get('event', '')} ({date})")
            
            # Prior diagnoses - consider chronic conditions
            elif category == 'diagnosis':
                insights.append(f"Known diagnosis: {event.get('event', '')}")
                # Check for relevant conditions
                if 'diabetes' in event_text:
                    considerations['Diabetic Ketoacidosis'] = considerations.get('Diabetic Ketoacidosis', 1.0) * 1.5
                    considerations['Hypoglycemia'] = considerations.get('Hypoglycemia', 1.0) * 1.4
                elif 'heart' in event_text or 'cardiac' in event_text:
                    considerations['Heart Failure'] = considerations.get('Heart Failure', 1.0) * 1.4
                    considerations['Myocardial Infarction'] = considerations.get('Myocardial Infarction', 1.0) * 1.3
                elif 'copd' in event_text or 'asthma' in event_text:
                    considerations['COPD Exacerbation'] = considerations.get('COPD Exacerbation', 1.0) * 1.5
                    considerations['Asthma Attack'] = considerations.get('Asthma Attack', 1.0) * 1.4
                elif 'thyroid' in event_text:
                    considerations['Hypothyroidism'] = considerations.get('Hypothyroidism', 1.0) * 1.3
                    considerations['Hyperthyroidism'] = considerations.get('Hyperthyroidism', 1.0) * 1.3
        
        return insights, considerations
    
    def _apply_probability_adjustments(self, results, adjustments):
        """
        Apply probability adjustments to diagnosis results
        
        Args:
            results: Diagnosis results from diagnose()
            adjustments: Dict of {disease_name: multiplier}
        
        Returns:
            Modified results with adjusted probabilities
        """
        if not adjustments:
            return results
        
        # Adjust primary diagnosis
        if results.get('primary'):
            disease_name = results['primary']['name']
            if disease_name in adjustments:
                original = results['primary']['probability']
                results['primary']['probability'] = min(99.0, original * adjustments[disease_name])
                results['primary']['lab_support'] = True
        
        # Adjust differential diagnoses
        for diff in results.get('differential', []):
            disease_name = diff['name']
            if disease_name in adjustments:
                original = diff['probability']
                diff['probability'] = min(99.0, original * adjustments[disease_name])
                diff['lab_support'] = True
        
        # Check if any adjusted disease should move up in ranking
        # (We don't re-sort here to preserve original clinical reasoning)
        
        return results
    
    def diagnose_with_patient_context(self, symptoms, patient=None):
        """
        Enhanced diagnosis that considers FULL patient context including:
        - Analyzed lab reports from PatientDataAggregator
        - Radiology/scan/pathology findings
        - Abnormal lab values affecting differential diagnoses
        - Timeline medical events
        
        Args:
            symptoms: List of symptoms
            patient: Patient data including all clinical information
        
        Returns:
            Enhanced diagnosis results with comprehensive patient-specific considerations
        """
        # Run base diagnosis
        results = self.diagnose(symptoms)
        
        # Add patient-specific context
        if patient:
            patient_considerations = []
            lab_insights = []
            radiology_insights = []
            timeline_insights = []
            all_adjustments = {}
            
            # ========== BASIC PATIENT INFO ==========
            # Age considerations
            age = patient.get('age', 0)
            if age > 65:
                patient_considerations.append("Elderly patient - consider atypical presentations")
            elif age < 18:
                patient_considerations.append("Pediatric patient - age-specific differential")
            
            # Medication interactions
            current_meds = patient.get('current_medications', [])
            if current_meds:
                patient_considerations.append(f"On {len(current_meds)} medications - check for drug-induced symptoms")
            
            # Chronic conditions from patient record
            medical_history = patient.get('medical_history', [])
            if medical_history:
                patient_considerations.append(f"Medical history: {', '.join(medical_history[:3])}")
            
            # Vitals
            vitals = patient.get('vitals', {})
            if vitals.get('temperature', 0) > 100.4:
                patient_considerations.append("Febrile - infectious etiology more likely")
            
            # ========== INTEGRATED DATA FROM AGGREGATOR ==========
            try:
                from .patient_data_aggregator import patient_data_aggregator
                
                # 1. ANALYZED LAB REPORTS
                lab_values = patient_data_aggregator.get_latest_lab_values(patient)
                abnormal_labs = patient_data_aggregator.get_abnormal_labs(patient)
                
                if lab_values:
                    lab_insights.append(f"📊 {len(lab_values)} lab values available from reports")
                
                if abnormal_labs:
                    lab_insight_texts = []
                    for lab in abnormal_labs[:5]:  # Top 5 abnormal values
                        lab_insight_texts.append(f"{lab['test']}: {lab['value']}")
                    lab_insights.append(f"⚠️ Abnormal: {', '.join(lab_insight_texts)}")
                    
                    # Get probability adjustments from labs
                    lab_adjustments = self._get_lab_adjustments(patient)
                    all_adjustments.update(lab_adjustments)
                
                # 2. RADIOLOGY/SCAN/PATHOLOGY FINDINGS
                radiology_clues, radiology_adjustments = self._get_radiology_clues(patient)
                if radiology_clues:
                    radiology_insights.extend(radiology_clues[:5])  # Top 5 clues
                    # Merge adjustments
                    for disease, mult in radiology_adjustments.items():
                        if disease in all_adjustments:
                            all_adjustments[disease] *= mult
                        else:
                            all_adjustments[disease] = mult
                
                # 3. TIMELINE MEDICAL EVENTS
                timeline_clues, timeline_adjustments = self._get_timeline_insights(patient)
                if timeline_clues:
                    timeline_insights.extend(timeline_clues[:5])  # Top 5 events
                    # Merge adjustments
                    for disease, mult in timeline_adjustments.items():
                        if disease in all_adjustments:
                            all_adjustments[disease] *= mult
                        else:
                            all_adjustments[disease] = mult
                
                # 4. APPLY ALL PROBABILITY ADJUSTMENTS
                if all_adjustments:
                    results = self._apply_probability_adjustments(results, all_adjustments)
                    patient_considerations.append(f"🔬 Diagnosis refined using {len(all_adjustments)} data-driven adjustments")
            
            except Exception as e:
                # Fallback if aggregator fails
                patient_considerations.append(f"Note: Advanced data integration unavailable")
                print(f"PatientDataAggregator error: {e}")
            
            # Compile all considerations
            results['patient_considerations'] = patient_considerations
            results['lab_insights'] = lab_insights if lab_insights else None
            results['radiology_insights'] = radiology_insights if radiology_insights else None
            results['timeline_insights'] = timeline_insights if timeline_insights else None
            
            # Summary of data sources used
            data_sources = []
            if lab_insights:
                data_sources.append("Lab Reports")
            if radiology_insights:
                data_sources.append("Radiology/Imaging")
            if timeline_insights:
                data_sources.append("Medical Timeline")
            if medical_history:
                data_sources.append("Medical History")
            
            results['data_sources_used'] = data_sources if data_sources else ["Basic Patient Info"]
        
        # Add test recommendations (already uses aggregator for existing tests)
        results['recommended_tests'] = self.recommend_tests(symptoms, results, patient)
        
        return results
    
    def diagnose_with_council(self, symptoms, patient=None):
        """
        Diagnosis with MULTI-MODEL AI COUNCIL deliberation.
        
        Uses 4 different AI models (Gemini, GPT-4, Mistral, Llama) to:
        1. Independently evaluate the diagnosis
        2. Debate each other's perspectives (2 rounds)
        3. Reach weighted consensus on primary and differential diagnoses
        
        Returns:
            Dict with diagnosis results AND council deliberation
        """
        # Step 1: Run patient-aware diagnosis
        base_results = self.diagnose_with_patient_context(symptoms, patient)
        
        # Step 2: Get AI Council deliberation
        try:
            from .ai_council import ai_council
            
            # Prepare case data for council
            case_data = {
                "symptoms": symptoms,
                "primary_diagnosis": base_results.get("primary", {}).get("name", "") if base_results.get("primary") else "",
                "primary_probability": base_results.get("primary", {}).get("probability", 0) if base_results.get("primary") else 0,
                "differentials": base_results.get("differential", []),
                "confidence": base_results.get("primary", {}).get("confidence", "") if base_results.get("primary") else "",
                "is_critical": base_results.get("primary", {}).get("is_critical", False) if base_results.get("primary") else False,
                "lab_insights": base_results.get("lab_insights", []),
                "radiology_insights": base_results.get("radiology_insights", [])
            }
            
            # Run council deliberation
            council_verdict = ai_council.deliberate(
                case_type="DIAGNOSIS",
                case_data=case_data,
                patient=patient
            )
            
            # Add council results
            base_results["council_deliberation"] = council_verdict.to_dict()
            base_results["council_verdict"] = council_verdict.final_verdict
            base_results["council_consensus"] = council_verdict.consensus_score
            base_results["council_dissent"] = council_verdict.dissenting_models
            
        except Exception as e:
            print(f"Council deliberation error in diagnosis: {e}")
            import traceback
            traceback.print_exc()
            base_results["council_deliberation"] = None
            base_results["council_error"] = str(e)
        
        return base_results

diagnosis_engine = DiagnosisEngine()

