"""
Advanced Contraindication Oracle
==================================
Comprehensive drug safety engine that analyzes patient data for:
- Drug-drug interactions
- Drug-disease contraindications
- Drug-food interactions
- Drug class allergies
- Age-specific concerns (Beers Criteria)
- Safety scoring and risk assessment
"""

class ContraindicationOracle:
    """
    World-class drug safety analysis engine.
    Provides comprehensive contraindication checking and interaction analysis.
    """
    
    def __init__(self):
        self._initialize_databases()
    
    def _initialize_databases(self):
        """Initialize all safety databases"""
        
        # ==================== DRUG-DRUG INTERACTIONS ====================
        # Format: (Drug1, Drug2): {severity, mechanism, clinical_effect, management, alternatives}
        self.drug_interactions = {
            # CRITICAL INTERACTIONS (Life-threatening)
            ("Warfarin", "Aspirin"): {
                "severity": "CRITICAL",
                "mechanism": "Additive antiplatelet + anticoagulation effects",
                "clinical_effect": "4-fold increased bleeding risk, potential for life-threatening hemorrhage",
                "incidence": "15-20% experience major bleeding",
                "management": "Avoid combination. If unavoidable, reduce Warfarin dose by 20-30% and monitor INR weekly",
                "monitoring": "INR, CBC, signs of bleeding (bruising, nosebleeds, GI bleeding)",
                "alternatives": ["Use Clopidogrel alone", "Consider Rivaroxaban instead of Warfarin"],
                "onset": "Within days"
            },
            
            ("Warfarin", "NSAIDs"): {
                "severity": "CRITICAL",
                "mechanism": "NSAIDs inhibit platelet function + increase GI bleeding risk",
                "clinical_effect": "GI hemorrhage, increased bleeding risk",
                "incidence": "3-5x increased bleeding risk",
                "management": "Avoid NSAIDs. Use Acetaminophen for pain instead",
                "monitoring": "Signs of GI bleeding, CBC",
                "alternatives": ["Acetaminophen", "Tramadol"],
                "onset": "Within 1-2 weeks"
            },
            
            ("MAO Inhibitors", "SSRIs"): {
                "severity": "CRITICAL",
                "mechanism": "Serotonin syndrome - excessive serotonergic activity",
                "clinical_effect": "Hyperthermia, seizures, altered mental status, death",
                "incidence": "High risk if combined",
                "management": "NEVER combine. Wait 14 days after stopping MAO inhibitor before starting SSRI",
                "monitoring": "Temperature, mental status, muscle rigidity",
                "alternatives": ["Use only one class at a time"],
                "onset": "Hours to days"
            },
            
            ("Metformin", "Contrast Dye"): {
                "severity": "CRITICAL",
                "mechanism": "Contrast-induced nephrotoxicity + Metformin = Lactic acidosis",
                "clinical_effect": "Severe lactic acidosis, metabolic crisis",
                "incidence": "Rare but life-threatening",
                "management": "Hold Metformin 48 hours before and 48 hours after contrast procedures",
                "monitoring": "Renal function, lactate levels, pH",
                "alternatives": ["Temporarily switch to insulin for glycemic control"],
                "onset": "24-48 hours post-contrast"
            },
            
            ("Digoxin", "Amiodarone"): {
                "severity": "CRITICAL",
                "mechanism": "Amiodarone increases Digoxin levels by 70-100%",
                "clinical_effect": "Digoxin toxicity: arrhythmias, nausea, visual disturbances",
                "incidence": "Very common if doses not adjusted",
                "management": "Reduce Digoxin dose by 50% when starting Amiodarone",
                "monitoring": "Digoxin levels, ECG, potassium",
                "alternatives": ["Consider different antiarrhythmic"],
                "onset": "Days to weeks"
            },
            
            # MAJOR INTERACTIONS (Serious but manageable)
            ("Lisinopril", "Spironolactone"): {
                "severity": "MAJOR",
                "mechanism": "Both increase potassium levels (ACE inhibitor + K-sparing diuretic)",
                "clinical_effect": "Hyperkalemia (K+ >5.5 mEq/L) → cardiac arrhythmias",
                "incidence": "15-20% of patients",
                "management": "Monitor potassium levels every 1-2 weeks initially, then monthly",
                "monitoring": "Serum potassium, renal function, ECG if K+ elevated",
                "alternatives": ["Use Hydrochlorothiazide instead of Spironolactone"],
                "onset": "1-4 weeks"
            },
            
            ("Simvastatin", "Amlodipine"): {
                "severity": "MAJOR",
                "mechanism": "Amlodipine inhibits CYP3A4 → increased Simvastatin levels",
                "clinical_effect": "Myopathy, rhabdomyolysis risk",
                "incidence": "5-10% myopathy risk at high Simvastatin doses",
                "management": "Limit Simvastatin to 20mg daily when combined with Amlodipine",
                "monitoring": "CK levels, muscle pain/weakness symptoms",
                "alternatives": ["Switch to Atorvastatin or Rosuvastatin (not CYP3A4 dependent)"],
                "onset": "Weeks to months"
            },
            
            ("Metformin", "Alcohol"): {
                "severity": "MAJOR",
                "mechanism": "Both can cause lactic acidosis; alcohol affects gluconeogenesis",
                "clinical_effect": "Increased lactic acidosis risk, hypoglycemia",
                "incidence": "Rare but serious",
                "management": "Limit alcohol to 1-2 drinks occasionally, avoid binge drinking",
                "monitoring": "Lactate levels if symptoms occur, blood glucose",
                "alternatives": ["Other diabetes medications if heavy alcohol use"],
                "onset": "Hours"
            },
            
            ("Levothyroxine", "Calcium"): {
                "severity": "MAJOR",
                "mechanism": "Calcium binds to Levothyroxine in GI tract, reducing absorption",
                "clinical_effect": "Reduced thyroid hormone levels, hypothyroid symptoms",
                "incidence": "Up to 40% reduction in absorption",
                "management": "Separate doses by at least 4 hours",
                "monitoring": "TSH levels every 6-8 weeks",
                "alternatives": ["Take Levothyroxine on empty stomach in morning, Calcium at night"],
                "onset": "Weeks"
            },
            
            ("Warfarin", "Antibiotics"): {
                "severity": "MAJOR",
                "mechanism": "Antibiotics kill gut flora that produce Vitamin K",
                "clinical_effect": "Increased anticoagulation → bleeding risk",
                "incidence": "Common with broad-spectrum antibiotics",
                "management": "Monitor INR more frequently (every 3-5 days) during antibiotic course",
                "monitoring": "INR, signs of bleeding",
                "alternatives": ["Consider narrow-spectrum antibiotic when possible"],
                "onset": "3-7 days"
            },
            
            # MODERATE INTERACTIONS (Require monitoring)
            ("Metformin", "Hydrochlorothiazide"): {
                "severity": "MODERATE",
                "mechanism": "Thiazides can increase blood glucose",
                "clinical_effect": "Reduced glycemic control",
                "incidence": "10-15% experience hyperglycemia",
                "management": "Monitor blood glucose, may need Metformin dose increase",
                "monitoring": "Fasting glucose, HbA1c every 3 months",
                "alternatives": ["Use ACE inhibitor or ARB for BP instead"],
                "onset": "Weeks"
            },
            
            ("Lisinopril", "Ibuprofen"): {
                "severity": "MODERATE",
                "mechanism": "NSAIDs reduce antihypertensive effect of ACE inhibitors",
                "clinical_effect": "Elevated blood pressure, reduced ACE inhibitor efficacy",
                "incidence": "Common - 10-15 mmHg BP increase possible",
                "management": "Monitor BP, use Acetaminophen for pain instead",
                "monitoring": "Blood pressure",
                "alternatives": ["Acetaminophen for pain/fever"],
                "onset": "Days to weeks"
            },
            
            ("Omeprazole", "Clopidogrel"): {
                "severity": "MODERATE",
                "mechanism": "Omeprazole inhibits CYP2C19, reducing Clopidogrel activation",
                "clinical_effect": "Reduced antiplatelet effect, increased cardiovascular events",
                "incidence": "Controversial - possible 25% reduction in Clopidogrel efficacy",
                "management": "Use Pantoprazole instead of Omeprazole if PPI needed",
                "monitoring": "Cardiovascular symptoms",
                "alternatives": ["Pantoprazole", "Famotidine"],
                "onset": "Immediate"
            },
            
            ("Metformin", "Prednisone"): {
                "severity": "MODERATE",
                "mechanism": "Corticosteroids increase insulin resistance and blood glucose",
                "clinical_effect": "Hyperglycemia, reduced diabetes control",
                "incidence": "Very common - nearly all patients affected",
                "management": "Increase Metformin dose or add additional diabetes medication",
                "monitoring": "Blood glucose 2-3x daily during steroid course",
                "alternatives": ["Use lowest effective steroid dose, short course"],
                "onset": "Days"
            },
            
            # Additional critical combinations
            ("Sildenafil", "Nitroglycerin"): {
                "severity": "CRITICAL",
                "mechanism": "Both cause vasodilation; synergistic hypotensive effect",
                "clinical_effect": "Severe hypotension, cardiovascular collapse, death",
                "incidence": "High risk - potentially fatal",
                "management": "Absolute contraindication. Never combine.",
                "monitoring": "Blood pressure",
                "alternatives": ["Avoid all PDE5 inhibitors if on nitrates"],
                "onset": "Minutes to hours"
            },
            
            ("Amiodarone", "Warfarin"): {
                "severity": "CRITICAL",
                "mechanism": "Amiodarone inhibits multiple CYP enzymes, dramatically increasing Warfarin levels",
                "clinical_effect": "Severe over-anticoagulation, bleeding",
                "incidence": "Nearly 100% will have elevated INR if not adjusted",
                "management": "Reduce Warfarin dose by 30-50% when starting Amiodarone. Check INR every 3-5 days",
                "monitoring": "INR at least weekly for 1 month",
                "alternatives": ["Consider DOACs instead of Warfarin"],
                "onset": "Days to weeks"
            },
            
            ("Lithium", "Hydrochlorothiazide"): {
                "severity": "CRITICAL",
                "mechanism": "Thiazides reduce lithium clearance",
                "clinical_effect": "Lithium toxicity: tremor, confusion, seizures, renal failure",
                "incidence": "High risk",
                "management": "Reduce lithium dose by 25-50%, check levels weekly initially",
                "monitoring": "Lithium levels, renal function, neurological symptoms",
                "alternatives": ["Use non-thiazide diuretic or different mood stabilizer"],
                "onset": "Days to weeks"
            },
        }
        
        # ==================== DRUG-DISEASE CONTRAINDICATIONS ====================
        self.drug_disease_contraindications = {
            ("Beta-blockers", "Asthma"): {
                "severity": "ABSOLUTE",
                "risk": "Bronchoconstriction, severe asthma exacerbation, respiratory failure",
                "rationale": "Beta-blockers block β2 receptors in airways causing bronchospasm",
                "alternatives": ["Calcium channel blockers (Diltiazem/Verapamil)", "ACE inhibitors", "ARBs"],
                "note": "Even selective β1-blockers (Metoprolol) can trigger asthma"
            },
            
            ("Beta-blockers", "COPD"): {
                "severity": "RELATIVE",
                "risk": "Bronchoconstriction, worsened respiratory function",
                "rationale": "Can worsen airflow obstruction",
                "alternatives": ["Use cardioselective β1-blockers with caution", "Prefer ACE inhibitors/ARBs"],
                "note": "Cardioselective beta-blockers (Metoprolol) safer than non-selective (Propranolol)"
            },
            
            ("NSAIDs", "Heart Failure"): {
                "severity": "ABSOLUTE",
                "risk": "Fluid retention, worsened heart failure, hospitalization",
                "rationale": "NSAIDs cause sodium/water retention and reduce diuretic efficacy",
                "alternatives": ["Acetaminophen", "Tramadol", "Topical analgesics"],
                "note": "Can precipitate acute decompensation"
            },
            
            ("NSAIDs", "Chronic Kidney Disease"): {
                "severity": "RELATIVE",
                "risk": "Acute kidney injury, worsened renal function",
                "rationale": "NSAIDs reduce renal blood flow via prostaglandin inhibition",
                "alternatives": ["Acetaminophen", "Opioids for severe pain"],
                "note": "Especially dangerous if eGFR <30"
            },
            
            ("Metformin", "Renal Impairment Severe"): {
                "severity": "ABSOLUTE",
                "risk": "Lactic acidosis - life-threatening metabolic emergency",
                "rationale": "Metformin accumulates with impaired renal clearance",
                "alternatives": ["Insulin", "DPP-4 inhibitors", "SGLT2 inhibitors (if eGFR allows)"],
                "note": "Contraindicated if eGFR <30 mL/min/1.73m²"
            },
            
            ("Sulfonylureas", "Severe Liver Disease"): {
                "severity": "ABSOLUTE",
                "risk": "Severe hypoglycemia, prolonged duration due to impaired metabolism",
                "rationale": "Hepatic metabolism impaired, drug accumulation occurs",
                "alternatives": ["Insulin", "DPP-4 inhibitors"],
                "note": "Hypoglycemia can be life-threatening"
            },
            
            ("ACE Inhibitors", "Bilateral Renal Artery Stenosis"): {
                "severity": "ABSOLUTE",
                "risk": "Acute renal failure",
                "rationale": "Blocks efferent arteriole constriction needed to maintain GFR",
                "alternatives": ["Calcium channel blockers", "Alpha blockers"],
                "note": "Can cause rapid deterioration in renal function"
            },
            
            ("Amiodarone", "Thyroid Disease"): {
                "severity": "RELATIVE",
                "risk": "Hypo or hyperthyroidism exacerbation",
                "rationale": "Amiodarone contains high iodine content, affects thyroid function",
                "alternatives": ["Other antiarrhythmics: Sotalol, Dofetilide"],
                "note": "Requires thyroid function monitoring every 6 months"
            },
            
            ("Anticholinergics", "Dementia"): {
                "severity": "RELATIVE",
                "risk": "Worsened cognitive function, confusion, falls",
                "rationale": "Cholinergic blockade worsens cognition",
                "alternatives": ["Non-anticholinergic alternatives when available"],
                "note": "Beers Criteria - avoid in elderly with dementia"
            },
            
            ("Benzodiazepines", "Sleep Apnea"): {
                "severity": "RELATIVE",
                "risk": "Respiratory depression, worsened sleep apnea",
                "rationale": "CNS depression decreases respiratory drive",
                "alternatives": ["Non-benzodiazepine sleep aids", "CBT for insomnia", "CPAP optimization"],
                "note": "Especially dangerous in untreated sleep apnea"
            },
        }
        
        # ==================== DRUG-FOOD INTERACTIONS ====================
        self.drug_food_interactions = {
            ("Warfarin", "Vitamin K Foods"): {
                "severity": "MAJOR",
                "foods": "Kale, Spinach, Broccoli, Brussels sprouts, Green leafy vegetables",
                "effect": "Reduced anticoagulation effect, decreased INR",
                "mechanism": "Vitamin K antagonizes Warfarin",
                "management": "Maintain consistent Vitamin K intake, don't avoid but be consistent",
                "monitoring": "INR"
            },
            
            ("Simvastatin", "Grapefruit Juice"): {
                "severity": "CRITICAL",
                "foods": "Grapefruit, grapefruit juice, Seville oranges",
                "effect": "10-15x increase in statin blood levels → rhabdomyolysis risk",
                "mechanism": "Grapefruit inhibits intestinal CYP3A4 enzyme",
                "management": "Avoid grapefruit entirely. Switch to statin not affected by grapefruit (Pravastatin, Rosuvastatin)",
                "monitoring": "CK levels, muscle symptoms"
            },
            
            ("Atorvastatin", "Grapefruit Juice"): {
                "severity": "MAJOR",
                "foods": "Grapefruit, grapefruit juice",
                "effect": "3-4x increase in statin levels → myopathy risk",
                "mechanism": "CYP3A4 inhibition",
                "management": "Avoid grapefruit or switch to Pravastatin/Rosuvastatin",
                "monitoring": "CK levels"
            },
            
            ("Levothyroxine", "Soy Products"): {
                "severity": "MODERATE",
                "foods": "Soy milk, tofu, edamame, soy protein",
                "effect": "Reduced thyroid hormone absorption",
                "mechanism": "Soy binds thyroid hormone in GI tract",
                "management": "Take Levothyroxine 4+ hours away from soy products",
                "monitoring": "TSH levels"
            },
            
            ("MAO Inhibitors", "Tyramine Foods"): {
                "severity": "CRITICAL",
                "foods": "Aged cheese, cured meats, soy sauce, draft beer, fermented foods",
                "effect": "Hypertensive crisis - severe BP elevation, stroke risk",
                "mechanism": "Tyramine accumulation causes catecholamine release",
                "management": "Strict dietary restriction of tyramine-rich foods",
                "monitoring": "Blood pressure, headache symptoms"
            },
            
            ("Antibiotics", "Dairy Products"): {
                "severity": "MODERATE",
                "foods": "Milk, yogurt, cheese, calcium-fortified foods",
                "effect": "Reduced antibiotic absorption (Tetracyclines, Fluoroquinolones)",
                "mechanism": "Calcium chelation in GI tract",
                "management": "Take antibiotic 2 hours before or 6 hours after dairy",
                "monitoring": "Clinical response to infection"
            },
            
            ("Metformin", "Alcohol"): {
                "severity": "MAJOR",
                "foods": "All alcoholic beverages",
                "effect": "Lactic acidosis risk, hypoglycemia",
                "mechanism": "Alcohol affects lactate metabolism and gluconeogenesis",
                "management": "Limit to 1-2 drinks occasionally, never binge drink",
                "monitoring": "Blood glucose, lactate if symptomatic"
            },
        }
        
        # ==================== DRUG CLASS FAMILIES (for allergy detection) ====================
        self.drug_class_families = {
            "Penicillins": [
                "Penicillin", "Amoxicillin", "Ampicillin", "Penicillin VK",
                "Piperacillin", "Ticarcillin", "Nafcillin", "Oxacillin",
                "Amoxicillin-Clavulanate", "Ampicillin-Sulbactam"
            ],
            
            "Cephalosporins": [
                "Cephalexin", "Cefazolin", "Cefuroxime", "Ceftriaxone",
                "Cefdinir", "Cefepime", "Ceftazidime", "Cefpodoxime"
            ],
            
            "Sulfonamides": [
                "Sulfamethoxazole", "Trimethoprim-Sulfamethoxazole", "Bactrim",
                "Sulfasalazine", "Sulfadiazine", "Celecoxib", "Furosemide",
                "Hydrochlorothiazide", "Sulfonylureas"
            ],
            
            "NSAIDs": [
                "Ibuprofen", "Naproxen", "Aspirin", "Diclofenac",
                "Indomethacin", "Ketorolac", "Meloxicam", "Celecoxib",
                "Piroxicam", "Etodolac"
            ],
            
            "Beta-Blockers": [
                "Metoprolol", "Atenolol", "Propranolol", "Carvedilol",
                "Bisoprolol", "Labetalol", "Nebivolol", "Timolol"
            ],
            
            "ACE Inhibitors": [
                "Lisinopril", "Enalapril", "Ramipril", "Benazepril",
                "Captopril", "Fosinopril", "Quinapril", "Perindopril"
            ],
            
            "ARBs": [
                "Losartan", "Valsartan", "Irbesartan", "Olmesartan",
                "Candesartan", "Telmisartan", "Azilsartan"
            ],
            
            "Statins": [
                "Atorvastatin", "Simvastatin", "Rosuvastatin", "Pravastatin",
                "Lovastatin", "Fluvastatin", "Pitavastatin"
            ],
            
            "Opioids": [
                "Morphine", "Hydrocodone", "Oxycodone", "Codeine",
                "Fentanyl", "Tramadol", "Hydromorphone", "Oxymorphone"
            ],
        }
        
        # ==================== BEERS CRITERIA (Elderly 65+) ====================
        self.beers_criteria = {
            "Benzodiazepines": {
                "drugs": ["Diazepam", "Lorazepam", "Alprazolam", "Clonazepam", "Temazepam"],
                "concern": "Increased sensitivity to benzodiazepines in elderly",
                "risks": ["Falls & fractures (2-3x risk)", "Cognitive impairment", "Delirium", "Motor vehicle accidents"],
                "recommendation": "Avoid if possible. Use non-pharmacologic approaches for anxiety/insomnia",
                "alternatives": ["CBT", "Melatonin", "Trazodone (low dose)"]
            },
            
            "Anticholinergics": {
                "drugs": ["Diphenhydramine", "Hydroxyzine", "Promethazine", "Cyclobenzaprine", "Oxybutynin"],
                "concern": "Anticholinergic burden in elderly",
                "risks": ["Confusion", "Dry mouth", "Constipation", "Urinary retention", "Blurred vision", "Dementia risk"],
                "recommendation": "Avoid in elderly, especially those with dementia",
                "alternatives": ["Non-anticholinergic antihistamines (Loratadine)", "Physical therapy for muscle spasms"]
            },
            
            "NSAIDs": {
                "drugs": ["Ibuprofen", "Naproxen", "Indomethacin", "Ketorolac"],
                "concern": "Increased GI bleeding and cardiovascular risks in elderly",
                "risks": ["GI ulcers/bleeding", "Acute kidney injury", "Heart failure exacerbation", "Hypertension"],
                "recommendation": "Avoid chronic use. Use lowest dose for shortest duration if needed",
                "alternatives": ["Acetaminophen", "Topical NSAIDs", "Physical therapy"]
            },
            
            "First-generation Antihistamines": {
                "drugs": ["Diphenhydramine", "Chlorpheniramine", "Hydroxyzine"],
                "concern": "Strong anticholinergic effects",
                "risks": ["Sedation", "Falls", "Confusion", "Cognitive decline"],
                "recommendation": "Avoid. Use 2nd or 3rd generation antihistamines",
                "alternatives": ["Loratadine", "Cetirizine", "Fexofenadine"]
            },
            
            "Sulfonylureas (long-acting)": {
                "drugs": ["Glyburide", "Chlorpropamide"],
                "concern": "Prolonged hypoglycemia risk in elderly",
                "risks": ["Severe, prolonged hypoglycemia", "Falls", "Cognitive impairment"],
                "recommendation": "Avoid. Use shorter-acting alternatives",
                "alternatives": ["Glipizide (short-acting)", "DPP-4 inhibitors", "Metformin if renal function adequate"]
            },
            
            "High-dose Aspirin": {
                "drugs": ["Aspirin >325mg"],
                "concern": "No additional cardiovascular benefit but increased bleeding",
                "risks": ["GI bleeding", "Intracranial hemorrhage"],
                "recommendation": "Use 81mg for cardiovascular protection",
                "alternatives": ["Low-dose aspirin (81mg)", "Clopidogrel"]
            },
        }
    
    # ==================== MAIN ANALYSIS METHODS ====================
    
    def analyze_patient_safety(self, patient, proposed_drug, dosage="Standard"):
        """
        Comprehensive safety analysis for a patient taking a proposed drug.
        
        Args:
            patient (dict): Full patient data including allergies, current_medications, 
                           medical_history, age, vitals, renal_function, hepatic_function, etc.
            proposed_drug (str): Drug name to analyze
            dosage (str): Proposed dosage
        
        Returns:
            dict: Comprehensive safety report with all warnings, contraindications, interactions
        """
        report = {
            "overall_risk": "LOW",  # LOW, MODERATE, HIGH, CRITICAL
            "safety_score": 100,
            "can_prescribe": True,
            "critical_alerts": [],
            "major_warnings": [],
            "moderate_warnings": [],
            "minor_notes": [],
            "blocked_reasons": [],
            "monitoring_required": [],
            "patient_considerations": []
        }
        
        # 1. Check for drug class allergies
        allergy_check = self._check_drug_allergies(patient, proposed_drug)
        if allergy_check:
            report["critical_alerts"].extend(allergy_check["critical"])
            report["major_warnings"].extend(allergy_check["warnings"])
            if allergy_check["blocked"]:
                report["can_prescribe"] = False
                report["overall_risk"] = "CRITICAL"
                report["blocked_reasons"].append(f"Patient allergic to {proposed_drug} or drug class")
        
        # 2. Check drug-drug interactions with current medications
        if patient.get("current_medications"):
            interaction_check = self._check_drug_interactions(patient["current_medications"], proposed_drug)
            report["critical_alerts"].extend(interaction_check["critical"])
            report["major_warnings"].extend(interaction_check["major"])
            report["moderate_warnings"].extend(interaction_check["moderate"])
            report["monitoring_required"].extend(interaction_check["monitoring"])
            
            if interaction_check["critical"]:
                report["overall_risk"] = "CRITICAL"
        
        # 3. Check drug-disease contraindications
        if patient.get("medical_history"):
            disease_check = self._check_disease_contraindications(patient["medical_history"], proposed_drug)
            if disease_check["absolute"]:
                report["critical_alerts"].extend(disease_check["absolute"])
                report["can_prescribe"] = False
                report["overall_risk"] = "CRITICAL"
            report["major_warnings"].extend(disease_check["relative"])
        
        # 4. Check organ function (renal/hepatic)
        organ_check = self._check_organ_function(patient, proposed_drug)
        if organ_check:
            report["major_warnings"].extend(organ_check["warnings"])
            report["patient_considerations"].extend(organ_check["considerations"])
            report["monitoring_required"].extend(organ_check["monitoring"])
        
        # 5. Check age-specific concerns (Beers Criteria for elderly)
        if patient.get("age", 0) >= 65:
            beers_check = self._check_beers_criteria(proposed_drug)
            if beers_check:
                report["major_warnings"].extend(beers_check["warnings"])
                report["patient_considerations"].extend(beers_check["considerations"])
        
        # 6. Check for drug-food interactions
        food_check = self._check_food_interactions(proposed_drug)
        if food_check:
            report["moderate_warnings"].extend(food_check)
        
        # 7. Calculate overall safety score
        report["safety_score"] = self._calculate_safety_score(report)
        
        # 8. Determine overall risk level
        if not report["overall_risk"] == "CRITICAL":
            report["overall_risk"] = self._determine_risk_level(report["safety_score"])
        
        return report
    
    def _check_drug_allergies(self, patient, proposed_drug):
        """Check if patient is allergic to proposed drug or its drug class"""
        allergies = patient.get("allergies", [])
        if not allergies:
            return None
        
        result = {"critical": [], "warnings": [], "blocked": False}
        
        # Direct allergy match
        for allergy in allergies:
            if allergy.lower() in proposed_drug.lower() or proposed_drug.lower() in allergy.lower():
                result["critical"].append({
                    "type": "DRUG ALLERGY",
                    "message": f"⛔ CRITICAL: Patient is allergic to {proposed_drug}",
                    "severity": "CRITICAL",
                    "action": "DO NOT PRESCRIBE - Absolute contraindication"
                })
                result["blocked"] = True
                return result
        
        # Check drug class allergies
        for drug_class, drugs in self.drug_class_families.items():
            # Check if allergy is to drug class
            for allergy in allergies:
                if allergy.lower() in drug_class.lower():
                    # Check if proposed drug is in that class
                    if any(drug.lower() in proposed_drug.lower() or proposed_drug.lower() in drug.lower() 
                           for drug in drugs):
                        result["critical"].append({
                            "type": "DRUG CLASS ALLERGY",
                            "message": f"⛔ CRITICAL: Patient allergic to {drug_class} class. {proposed_drug} is in this class.",
                            "severity": "CRITICAL",
                            "action": "DO NOT PRESCRIBE - Cross-sensitivity risk"
                        })
                        result["blocked"] = True
                        return result
            
            # Check if proposed drug is in a class the patient is allergic to
            if any(drug.lower() in proposed_drug.lower() or proposed_drug.lower() in drug.lower() 
                   for drug in drugs):
                for allergy in allergies:
                    if any(drug.lower() in allergy.lower() for drug in drugs):
                        result["warnings"].append({
                            "type": "CROSS-SENSITIVITY WARNING",
                            "message": f"⚠️ WARNING: Patient allergic to {allergy}. {proposed_drug} is in same class ({drug_class}). Cross-sensitivity possible.",
                            "severity": "MAJOR",
                            "action": "Use with extreme caution or choose alternative"
                        })
        
        return result if result["critical"] or result["warnings"] else None
    
    def _check_drug_interactions(self, current_medications, proposed_drug):
        """Check for interactions between proposed drug and current medications"""
        result = {
            "critical": [],
            "major": [],
            "moderate": [],
            "monitoring": []
        }
        
        for med in current_medications:
            current_drug = med.get("name", "") if isinstance(med, dict) else str(med)
            
            # Check both directions (Drug A + Drug B and Drug B + Drug A)
            interaction = None
            if (current_drug, proposed_drug) in self.drug_interactions:
                interaction = self.drug_interactions[(current_drug, proposed_drug)]
            elif (proposed_drug, current_drug) in self.drug_interactions:
                interaction = self.drug_interactions[(proposed_drug, current_drug)]
            
            if interaction:
                alert = {
                    "drugs": f"{current_drug} + {proposed_drug}",
                    "severity": interaction["severity"],
                    "mechanism": interaction["mechanism"],
                    "clinical_effect": interaction["clinical_effect"],
                    "management": interaction["management"],
                    "alternatives": interaction.get("alternatives", []),
                    "onset": interaction.get("onset", "Variable")
                }
                
                if interaction["severity"] == "CRITICAL":
                    alert["message"] = f"🔴 CRITICAL INTERACTION: {current_drug} + {proposed_drug}"
                    result["critical"].append(alert)
                elif interaction["severity"] == "MAJOR":
                    alert["message"] = f"🟠 MAJOR INTERACTION: {current_drug} + {proposed_drug}"
                    result["major"].append(alert)
                elif interaction["severity"] == "MODERATE":
                    alert["message"] = f"🟡 MODERATE INTERACTION: {current_drug} + {proposed_drug}"
                    result["moderate"].append(alert)
                
                # Add monitoring requirements
                if interaction.get("monitoring"):
                    result["monitoring"].append({
                        "test": interaction["monitoring"],
                        "reason": f"Monitor for {current_drug} + {proposed_drug} interaction",
                        "frequency": "Per interaction guidelines"
                    })
        
        return result
    
    def _check_disease_contraindications(self, medical_history, proposed_drug):
        """Check if proposed drug is contraindicated based on patient's medical history"""
        result = {"absolute": [], "relative": []}
        
        for condition in medical_history:
            condition_name = condition.get("condition", "") if isinstance(condition, dict) else str(condition)
            
            # Check if proposed drug class is contraindicated for this condition
            for (drug_class, disease), contraindication in self.drug_disease_contraindications.items():
                # Match disease
                if disease.lower() in condition_name.lower() or condition_name.lower() in disease.lower():
                    # Check if proposed drug is in this drug class
                    if drug_class.lower() in proposed_drug.lower() or proposed_drug.lower() in drug_class.lower():
                        alert = {
                            "condition": condition_name,
                            "drug_class": drug_class,
                            "severity": contraindication["severity"],
                            "risk": contraindication["risk"],
                            "rationale": contraindication["rationale"],
                            "alternatives": contraindication.get("alternatives", [])
                        }
                        
                        if contraindication["severity"] == "ABSOLUTE":
                            alert["message"] = f"⛔ ABSOLUTE CONTRAINDICATION: {proposed_drug} contraindicated in {condition_name}"
                            result["absolute"].append(alert)
                        else:  # RELATIVE
                            alert["message"] = f"⚠️ RELATIVE CONTRAINDICATION: Use {proposed_drug} with caution in {condition_name}"
                            result["relative"].append(alert)
        
        return result
    
    def _check_organ_function(self, patient, proposed_drug):
        """Check if drug dosing needs adjustment based on renal/hepatic function"""
        result = {"warnings": [], "considerations": [], "monitoring": []}
        
        vitals = patient.get("vitals", {})
        renal_function = vitals.get("renal_function") or patient.get("renal_function", "Normal")
        hepatic_function = vitals.get("hepatic_function") or patient.get("hepatic_function", "Normal")
        
        # Renal function checks
        if "Severe" in renal_function or "Moderate" in renal_function:
            renally_cleared_drugs = ["Metformin", "Digoxin", "Lithium", "Gabapentin", "ACE Inhibitors", 
                                     "ARBs", "NSAIDs", "Antibiotics"]
            
            for drug_class in renally_cleared_drugs:
                if drug_class.lower() in proposed_drug.lower():
                    if "Severe" in renal_function and "Metformin" in drug_class:
                        result["warnings"].append({
                            "type": "RENAL CONTRAINDICATION",
                            "message": f"⛔ {proposed_drug} contraindicated in severe renal impairment",
                            "severity": "CRITICAL",
                            "action": "Do not prescribe. Use alternative diabetes medication"
                        })
                    else:
                        result["considerations"].append({
                            "type": "RENAL DOSING",
                            "message": f"⚠️ {proposed_drug} requires dose adjustment for {renal_function}",
                            "action": "Reduce dose by 25-50% or use alternative"
                        })
                        result["monitoring"].append({
                            "test": "Serum creatinine, eGFR",
                            "reason": f"Monitor renal function on {proposed_drug}",
                            "frequency": "Every 3-6 months"
                        })
        
        # Hepatic function checks
        if "Impairment" in hepatic_function:
            hepatically_metabolized = ["Statins", "Warfarin", "Sulfonylureas", "Benzodiazepines"]
            
            for drug_class in hepatically_metabolized:
                if drug_class.lower() in proposed_drug.lower():
                    result["considerations"].append({
                        "type": "HEPATIC DOSING",
                        "message": f"⚠️ {proposed_drug} metabolism affected by {hepatic_function}",
                        "action": "Start with low dose and titrate carefully"
                    })
                    result["monitoring"].append({
                        "test": "Liver function tests (AST, ALT, albumin)",
                        "reason": f"Monitor hepatic function on {proposed_drug}",
                        "frequency": "Every 3-6 months"
                    })
        
        return result if result["warnings"] or result["considerations"] else None
    
    def _check_beers_criteria(self, proposed_drug):
        """Check if drug is potentially inappropriate for elderly (Beers Criteria)"""
        result = {"warnings": [], "considerations": []}
        
        for drug_class, criteria in self.beers_criteria.items():
            # Check if proposed drug is in this Beers category
            if any(drug.lower() in proposed_drug.lower() or proposed_drug.lower() in drug.lower() 
                   for drug in criteria["drugs"]):
                result["warnings"].append({
                    "type": "BEERS CRITERIA",
                    "message": f"👴 ELDERLY CONCERN: {proposed_drug} on Beers Criteria (potentially inappropriate for age 65+)",
                    "severity": "MAJOR",
                    "concern": criteria["concern"],
                    "risks": criteria["risks"],
                    "recommendation": criteria["recommendation"],
                    "alternatives": criteria.get("alternatives", [])
                })
                
                result["considerations"].append({
                    "type": "AGE-RELATED RISK",
                    "message": f"Increased risk in elderly: {', '.join(criteria['risks'][:2])}",
                    "action": criteria["recommendation"]
                })
                break
        
        return result if result["warnings"] else None
    
    def _check_food_interactions(self, proposed_drug):
        """Check for significant food-drug interactions"""
        warnings = []
        
        for (drug, food_type), interaction in self.drug_food_interactions.items():
            if drug.lower() in proposed_drug.lower() or proposed_drug.lower() in drug.lower():
                warnings.append({
                    "type": "DRUG-FOOD INTERACTION",
                    "message": f"🍽️ DIETARY WARNING: {proposed_drug} interacts with {food_type}",
                    "severity": interaction["severity"],
                    "foods": interaction["foods"],
                    "effect": interaction["effect"],
                    "management": interaction["management"]
                })
        
        return warnings
    
    def _calculate_safety_score(self, report):
        """Calculate overall safety score 0-100"""
        score = 100
        
        # Critical alerts: -50 each
        score -= len(report["critical_alerts"]) * 50
        
        # Major warnings: -20 each
        score -= len(report["major_warnings"]) * 20
        
        # Moderate warnings: -10 each
        score -= len(report["moderate_warnings"]) * 10
        
        return max(0, min(100, score))
    
    def _determine_risk_level(self, safety_score):
        """Determine risk level based on safety score"""
        if safety_score >= 90:
            return "LOW"
        elif safety_score >= 70:
            return "MODERATE"
        elif safety_score >= 40:
            return "HIGH"
        else:
            return "CRITICAL"


# Initialize the Oracle
contraindication_oracle = ContraindicationOracle()
