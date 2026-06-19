import random
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self):
        self.conditions = [
            "Hypertension", "Type 2 Diabetes", "Asthma", "COPD", 
            "Atrial Fibrillation", "Hyperlipidemia", "Osteoarthritis"
        ]
        self.medications = [
            {"name": "Lisinopril", "class": "ACE Inhibitor", "cost": 10},
            {"name": "Metformin", "class": "Biguanide", "cost": 5},
            {"name": "Atorvastatin", "class": "Statin", "cost": 15},
            {"name": "Albuterol", "class": "Beta-2 Agonist", "cost": 25},
            {"name": "Amlodipine", "class": "Calcium Channel Blocker", "cost": 12},
            {"name": "Losartan", "class": "ARB", "cost": 18},
            {"name": "Levothyroxine", "class": "Thyroid Hormone", "cost": 8}
        ]
        self.allergies = ["Penicillin", "Sulfa", "Peanuts", "Latex", "None", "None", "None"]
        
        # Initialize stateful storage
        self.patients = {}
        self._initialize_mock_data()

    def _initialize_mock_data(self):
        for i in range(1, 21):
            self.patients[i] = self._generate_mock_patient(i)

    def generate_vitals(self):
        return {
            "bp_systolic": random.randint(110, 160),
            "bp_diastolic": random.randint(70, 100),
            "heart_rate": random.randint(60, 100),
            "temp": round(random.uniform(97.5, 99.5), 1),
            "spO2": random.randint(94, 100),
            "weight": random.randint(60, 120), # kg
            "recorded_at": (datetime.now() - timedelta(minutes=random.randint(15, 1440))).strftime("%Y-%m-%d %H:%M")
        }

    def generate_history(self):
        history = []
        start_date = datetime.now() - timedelta(days=365*2)
        for _ in range(random.randint(3, 8)):
            event_date = start_date + timedelta(days=random.randint(0, 700))
            history.append({
                "date": event_date.strftime("%Y-%m-%d"),
                "event": random.choice(["Routine Checkup", "Emergency Visit", "Lab Work", "Prescription Refill"]),
                "notes": "Patient reported stable symptoms.",
                "attachments": []  # NEW: Store file attachments for timeline events
            })
        return sorted(history, key=lambda x: x['date'], reverse=True)
    
    def _generate_random_supplements(self):
        """Generate random supplements/Ayurvedic medicines for mock patients"""
        # Common supplements in India and worldwide
        ayurvedic_herbs = [
            {"name": "Ashwagandha", "category": "Ayurvedic", "purpose": "Stress & Anxiety", "cyp_impact": "CYP3A4 inhibitor"},
            {"name": "Brahmi (Bacopa)", "category": "Ayurvedic", "purpose": "Cognitive Enhancement", "cyp_impact": "CYP2C19 inhibitor"},
            {"name": "Triphala", "category": "Ayurvedic", "purpose": "Digestive Health", "cyp_impact": "Minor CYP3A4 effect"},
            {"name": "Arjuna", "category": "Ayurvedic", "purpose": "Cardiac Support", "cyp_impact": "May potentiate cardiac glycosides"},
            {"name": "Guggul", "category": "Ayurvedic", "purpose": "Cholesterol", "cyp_impact": "CYP3A4 inducer - reduces many drug levels"},
            {"name": "Tulsi (Holy Basil)", "category": "Ayurvedic", "purpose": "Immunity & Stress", "cyp_impact": "CYP2C9 inhibitor"},
            {"name": "Shilajit", "category": "Ayurvedic", "purpose": "Energy & Vitality", "cyp_impact": "Enhances drug absorption"},
            {"name": "Shatavari", "category": "Ayurvedic", "purpose": "Women's Health", "cyp_impact": "Mild CYP3A4 effect"},
            {"name": "Neem", "category": "Ayurvedic", "purpose": "Blood Purification", "cyp_impact": "CYP1A2 inducer"},
            {"name": "Giloy (Tinospora)", "category": "Ayurvedic", "purpose": "Immunity", "cyp_impact": "CYP3A4 inhibitor"},
        ]
        
        western_supplements = [
            {"name": "Fish Oil (Omega-3)", "category": "Supplement", "purpose": "Cardiovascular", "cyp_impact": "Antiplatelet - bleeding risk with anticoagulants"},
            {"name": "Vitamin D3", "category": "Supplement", "purpose": "Bone Health", "cyp_impact": "CYP24A1 substrate"},
            {"name": "St. John's Wort", "category": "Herbal", "purpose": "Depression", "cyp_impact": "STRONG CYP3A4 inducer - MAJOR INTERACTION RISK"},
            {"name": "Ginkgo Biloba", "category": "Herbal", "purpose": "Memory", "cyp_impact": "CYP2C9 inhibitor, bleeding risk"},
            {"name": "Garlic Extract", "category": "Supplement", "purpose": "Cardiovascular", "cyp_impact": "CYP3A4 inducer, antiplatelet effect"},
            {"name": "Turmeric/Curcumin", "category": "Supplement", "purpose": "Anti-inflammatory", "cyp_impact": "CYP1A2, 3A4 inhibitor"},
            {"name": "Ginseng", "category": "Herbal", "purpose": "Energy", "cyp_impact": "CYP3A4 inhibitor, hypoglycemia risk"},
            {"name": "Echinacea", "category": "Herbal", "purpose": "Immunity", "cyp_impact": "CYP3A4 modulator"},
            {"name": "Valerian Root", "category": "Herbal", "purpose": "Sleep", "cyp_impact": "CNS depression with sedatives"},
            {"name": "Milk Thistle", "category": "Herbal", "purpose": "Liver Support", "cyp_impact": "CYP2C9 inhibitor"},
        ]
        
        all_supplements = ayurvedic_herbs + western_supplements
        
        # 40% chance of no supplements, otherwise 1-3 supplements
        if random.random() < 0.4:
            return []
        
        num_supplements = random.randint(1, 3)
        selected = random.sample(all_supplements, min(num_supplements, len(all_supplements)))
        
        # Add start date to each
        for supp in selected:
            supp["start_date"] = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
        
        return selected
    
    def _generate_sample_lab_reports(self, patient_id, condition):
        """Generate realistic sample lab reports for mock patients"""
        random.seed(patient_id + 1000)  # Different seed for labs
        
        lab_reports = []
        
        # Sample lab test templates
        cbc_template = {
            "test_name": "Complete Blood Count",
            "tests": [
                {"name": "WBC", "value": f"{round(random.uniform(4.0, 12.0), 1)}", "unit": "x10^9/L", "reference_range": "4.0-11.0", "status": ""},
                {"name": "RBC", "value": f"{round(random.uniform(4.0, 5.8), 2)}", "unit": "x10^12/L", "reference_range": "4.5-5.5", "status": ""},
                {"name": "Hemoglobin", "value": f"{round(random.uniform(11.0, 17.0), 1)}", "unit": "g/dL", "reference_range": "12.0-16.0", "status": ""},
                {"name": "Hematocrit", "value": f"{round(random.uniform(35, 50), 1)}", "unit": "%", "reference_range": "36-46", "status": ""},
                {"name": "Platelets", "value": f"{random.randint(150, 400)}", "unit": "x10^9/L", "reference_range": "150-400", "status": ""},
                {"name": "MCV", "value": f"{round(random.uniform(80, 100), 1)}", "unit": "fL", "reference_range": "80-100", "status": ""},
            ]
        }
        
        metabolic_template = {
            "test_name": "Comprehensive Metabolic Panel",
            "tests": [
                {"name": "Glucose", "value": f"{random.randint(70, 180)}", "unit": "mg/dL", "reference_range": "70-100", "status": ""},
                {"name": "BUN", "value": f"{random.randint(7, 35)}", "unit": "mg/dL", "reference_range": "7-20", "status": ""},
                {"name": "Creatinine", "value": f"{round(random.uniform(0.6, 2.0), 2)}", "unit": "mg/dL", "reference_range": "0.7-1.3", "status": ""},
                {"name": "eGFR", "value": f"{random.randint(45, 120)}", "unit": "mL/min/1.73m²", "reference_range": ">60", "status": ""},
                {"name": "Sodium", "value": f"{random.randint(135, 148)}", "unit": "mEq/L", "reference_range": "136-145", "status": ""},
                {"name": "Potassium", "value": f"{round(random.uniform(3.5, 5.5), 1)}", "unit": "mEq/L", "reference_range": "3.5-5.0", "status": ""},
                {"name": "AST", "value": f"{random.randint(15, 60)}", "unit": "U/L", "reference_range": "10-40", "status": ""},
                {"name": "ALT", "value": f"{random.randint(15, 55)}", "unit": "U/L", "reference_range": "7-56", "status": ""},
            ]
        }
        
        lipid_template = {
            "test_name": "Lipid Panel",
            "tests": [
                {"name": "Total Cholesterol", "value": f"{random.randint(150, 280)}", "unit": "mg/dL", "reference_range": "<200", "status": ""},
                {"name": "LDL", "value": f"{random.randint(70, 180)}", "unit": "mg/dL", "reference_range": "<100", "status": ""},
                {"name": "HDL", "value": f"{random.randint(35, 80)}", "unit": "mg/dL", "reference_range": ">40", "status": ""},
                {"name": "Triglycerides", "value": f"{random.randint(80, 300)}", "unit": "mg/dL", "reference_range": "<150", "status": ""},
            ]
        }
        
        hba1c_template = {
            "test_name": "Hemoglobin A1c",
            "tests": [
                {"name": "HbA1c", "value": f"{round(random.uniform(5.0, 9.5), 1)}", "unit": "%", "reference_range": "<5.7", "status": ""},
                {"name": "Estimated Average Glucose", "value": f"{random.randint(100, 230)}", "unit": "mg/dL", "reference_range": "N/A", "status": ""},
            ]
        }
        
        # Mark abnormal values
        def mark_abnormals(tests):
            for test in tests:
                val = float(test["value"])
                ref = test["reference_range"]
                try:
                    if "-" in ref:
                        low, high = ref.split("-")
                        if val < float(low):
                            test["status"] = "LOW"
                        elif val > float(high):
                            test["status"] = "HIGH"
                        else:
                            test["status"] = "Normal"
                    elif "<" in ref:
                        if val >= float(ref.replace("<", "")):
                            test["status"] = "HIGH"
                        else:
                            test["status"] = "Normal"
                    elif ">" in ref:
                        if val <= float(ref.replace(">", "")):
                            test["status"] = "LOW"
                        else:
                            test["status"] = "Normal"
                    else:
                        test["status"] = "Normal"
                except:
                    test["status"] = "Normal"
            return tests
        
        # Generate 2-4 lab reports per patient
        num_reports = random.randint(2, 4)
        
        templates = [cbc_template, metabolic_template, lipid_template]
        if "Diabetes" in condition:
            templates.append(hba1c_template)
        
        for i in range(num_reports):
            template = random.choice(templates)
            report_date = (datetime.now() - timedelta(days=random.randint(7, 180))).strftime("%Y-%m-%d")
            
            report = {
                "id": i + 1,
                "date": report_date,
                "tests": mark_abnormals([{**t} for t in template["tests"]]),  # Deep copy
                "overall_summary": f"{template['test_name']} completed. Results reviewed.",
                "interpretation": f"Panel shows overall stable results with some values requiring monitoring." if i % 2 == 0 else "Results within expected ranges for patient's condition.",
                "analyzed_at": (datetime.now() - timedelta(days=random.randint(7, 180))).isoformat(),
                "raw_text": ""
            }
            
            lab_reports.append(report)
        
        # Sort by date
        lab_reports = sorted(lab_reports, key=lambda x: x['date'], reverse=True)
        return lab_reports
    
    def _generate_sample_radiology_studies(self, patient_id, condition):
        """Generate realistic sample radiology studies for mock patients"""
        random.seed(patient_id + 2000)  # Different seed for radiology
        
        radiology_studies = []
        
        # Sample radiology findings based on conditions
        ct_chest_findings = [
            {"finding": "No acute cardiopulmonary abnormality", "critical": False},
            {"finding": "Mild cardiomegaly", "critical": False},
            {"finding": "Small bilateral pleural effusions", "critical": False},
            {"finding": "Calcified granuloma in right middle lobe - benign appearance", "critical": False},
            {"finding": "Coronary artery calcifications noted", "critical": False},
            {"finding": "Clear lung fields bilaterally", "critical": False},
        ]
        
        ct_abdomen_findings = [
            {"finding": "Hepatic steatosis (fatty liver)", "critical": False},
            {"finding": "Gallstones without acute cholecystitis", "critical": False},
            {"finding": "Simple renal cyst, left kidney, 2.1cm", "critical": False},
            {"finding": "No bowel obstruction or free fluid", "critical": False},
            {"finding": "Atherosclerotic calcifications of abdominal aorta", "critical": False},
            {"finding": "Normal liver, spleen, pancreas, and kidneys", "critical": False},
        ]
        
        xray_chest_findings = [
            {"finding": "No acute infiltrate or consolidation", "critical": False},
            {"finding": "Mild hyperinflation suggesting COPD", "critical": False},
            {"finding": "Heart size within normal limits", "critical": False},
            {"finding": "Clear lungs, no pleural effusion", "critical": False},
            {"finding": "Calcified lymph nodes in hilum - chronic granulomatous disease", "critical": False},
        ]
        
        mri_brain_findings = [
            {"finding": "No acute intracranial abnormality", "critical": False},
            {"finding": "Age-appropriate white matter changes", "critical": False},
            {"finding": "No mass effect or midline shift", "critical": False},
            {"finding": "Mild chronic small vessel ischemic disease", "critical": False},
            {"finding": "Normal brain parenchyma and ventricles", "critical": False},
        ]
        
        ecg_findings = [
            {"finding": "Normal sinus rhythm, rate 72 bpm", "critical": False},
            {"finding": "Left ventricular hypertrophy by voltage criteria", "critical": False},
            {"finding": "Nonspecific ST-T changes", "critical": False},
            {"finding": "Sinus bradycardia, rate 55 bpm", "critical": False},
            {"finding": "First-degree AV block", "critical": False},
        ]
        
        study_types = [
            {"type": "CT Chest", "findings": ct_chest_findings, "indication": "Chest pain / shortness of breath workup"},
            {"type": "CT Abdomen/Pelvis", "findings": ct_abdomen_findings, "indication": "Abdominal pain evaluation"},
            {"type": "Chest X-ray", "findings": xray_chest_findings, "indication": "Routine screening / cough evaluation"},
            {"type": "MRI Brain", "findings": mri_brain_findings, "indication": "Headache / cognitive changes workup"},
            {"type": "ECG/EKG", "findings": ecg_findings, "indication": "Cardiac screening"},
        ]
        
        # Generate 1-3 radiology studies per patient
        num_studies = random.randint(1, 3)
        
        selected_types = random.sample(study_types, min(num_studies, len(study_types)))
        
        for i, study_type in enumerate(selected_types):
            study_date = (datetime.now() - timedelta(days=random.randint(14, 365))).strftime("%Y-%m-%d")
            
            # Select 2-4 findings
            num_findings = random.randint(2, 4)
            selected_findings = random.sample(study_type["findings"], min(num_findings, len(study_type["findings"])))
            
            findings_text = [f["finding"] for f in selected_findings]
            critical = [f["finding"] for f in selected_findings if f["critical"]]
            
            study = {
                "id": i + 1,
                "study_type": study_type["type"],
                "study_date": study_date,
                "indication": study_type["indication"],
                "findings": findings_text,
                "impression": f"{study_type['type']} demonstrates {len(findings_text)} findings. Overall impression: {'Critical findings require immediate attention.' if critical else 'No acute abnormality requiring immediate intervention.'}",
                "critical_findings": critical if critical else [],
                "recommendations": ["Clinical correlation recommended", "Follow-up as clinically indicated"] if random.random() > 0.5 else ["No follow-up imaging required"],
                "analyzed_at": (datetime.now() - timedelta(days=random.randint(14, 365))).isoformat(),
                "raw_text": ""
            }
            
            radiology_studies.append(study)
        
        # Sort by date
        radiology_studies = sorted(radiology_studies, key=lambda x: x['study_date'], reverse=True)
        return radiology_studies

    def _generate_mock_patient(self, patient_id):
        # Deterministic seed for consistent initial data
        random.seed(patient_id)
        
        age = random.randint(25, 85)
        sex = random.choice(["M", "F"])
        condition = random.choice(self.conditions)
        
        # Generate weight and height based on age and sex
        if sex == "M":
            weight = random.randint(60, 110)  # kg
            height = random.randint(165, 190)  # cm
        else:
            weight = random.randint(50, 90)  # kg
            height = random.randint(150, 175)  # cm
        
        # Pharmacogenomics - CYP450 Enzyme Metabolizer Status
        # Real-world distribution approximations
        metabolizer_statuses = ["Poor", "Intermediate", "Normal", "Rapid", "Ultra-Rapid"]
        metabolizer_weights = [0.07, 0.15, 0.60, 0.12, 0.06]  # Approximate population frequencies
        
        def get_metabolizer_status():
            return random.choices(metabolizer_statuses, weights=metabolizer_weights, k=1)[0]
        
        pharmacogenomics = {
            # Major CYP enzymes affecting drug metabolism
            "CYP2D6": get_metabolizer_status(),   # Codeine, Tramadol, Tamoxifen, many antidepressants
            "CYP2C19": get_metabolizer_status(),  # Clopidogrel, PPIs, some antidepressants
            "CYP2C9": get_metabolizer_status(),   # Warfarin, NSAIDs, phenytoin
            "CYP3A4": get_metabolizer_status(),   # 50%+ of all drugs (statins, calcium blockers, etc.)
            "CYP1A2": get_metabolizer_status(),   # Caffeine, theophylline, clozapine
            # Other pharmacogenomic markers
            "VKORC1": random.choice(["Normal Sensitivity", "High Sensitivity", "Low Sensitivity"]),  # Warfarin sensitivity
            "HLA_B5701": random.choice([False, False, False, False, True]),  # Abacavir hypersensitivity (5-8% positive)
            "TPMT": random.choice(["Normal", "Intermediate", "Deficient"]),  # Thiopurines (azathioprine, 6-MP)
            "tested_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d") if random.random() > 0.3 else None
        }
        
        # Generate sample lab and radiology data
        lab_reports = self._generate_sample_lab_reports(patient_id, condition)
        radiology_studies = self._generate_sample_radiology_studies(patient_id, condition)
        
        return {
            "id": patient_id,
            "name": f"Patient_{patient_id}", 
            "age": age,
            "sex": sex,
            "weight": weight,  # kg
            "height": height,  # cm
            "bmi": round(weight / ((height/100) ** 2), 1),  # Calculate BMI
            "bsa": round(0.007184 * (weight ** 0.425) * (height ** 0.725), 2),  # Body Surface Area (Mosteller formula)
            "condition": condition,
            "status": random.choice(["Stable", "Critical", "Monitoring"]) if age > 60 else "Stable",
            "vitals": self.generate_vitals(),
            "allergies": [random.choice(self.allergies)],
            "renal_function": random.choice(["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"]),
            "hepatic_function": random.choice(["Normal", "Mild Impairment", "Moderate Impairment"]),
            "pharmacogenomics": pharmacogenomics,  # NEW: CYP450 enzyme data
            "current_medications": [{**med, "start_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")} for med in random.sample(self.medications, random.randint(1, 3))],
            "supplements": self._generate_random_supplements(),  # NEW: Ayurvedic/herbal medicines
            "medication_history": [],
            "history": self.generate_history(),
            "lab_reports": lab_reports,  # POPULATED: Sample lab test results
            "radiology_studies": radiology_studies,  # POPULATED: Sample imaging studies
            "last_visit": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        }

    def get_patient(self, patient_id):
        return self.patients.get(patient_id)

    def get_all_patients(self):
        return list(self.patients.values())

    def add_patient(self, data):
        new_id = max(self.patients.keys()) + 1 if self.patients else 1
        
        # Calculate weight-based metrics
        weight = float(data.get("weight", 70))  # Default 70kg
        height = float(data.get("height", 170))  # Default 170cm
        
        # Default pharmacogenomics (not tested)
        pharmacogenomics = data.get("pharmacogenomics", {
            "CYP2D6": "Not Tested",
            "CYP2C19": "Not Tested",
            "CYP2C9": "Not Tested",
            "CYP3A4": "Not Tested",
            "CYP1A2": "Not Tested",
            "VKORC1": "Not Tested",
            "HLA_B5701": None,
            "TPMT": "Not Tested",
            "tested_date": None
        })
        
        # Basic validation/defaults
        patient = {
            "id": new_id,
            "name": data.get("name", "Unknown"),
            "age": int(data.get("age", 0)),
            "sex": data.get("sex", "U"),
            "weight": weight,
            "height": height,
            "bmi": round(weight / ((height/100) ** 2), 1),
            "bsa": round(0.007184 * (weight ** 0.425) * (height ** 0.725), 2),
            "condition": data.get("condition", "None"),
            "status": data.get("status", "Stable"),
            "vitals": self.generate_vitals(),
            "allergies": data.get("allergies", ["None"]) if isinstance(data.get("allergies"), list) else [data.get("allergies", "None")],
            "renal_function": data.get("renal_function", "Normal"),
            "hepatic_function": data.get("hepatic_function", "Normal"),
            "pharmacogenomics": pharmacogenomics,
            "current_medications": [],
            "supplements": data.get("supplements", []),  # NEW: Ayurvedic/herbal medicines
            "medication_history": [],
            "history": [],
            "lab_reports": [],
            "radiology_studies": [],
            "last_visit": datetime.now().strftime("%Y-%m-%d")
        }
        self.patients[new_id] = patient
        return patient

    def update_patient(self, patient_id, data):
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        # Update fields if present
        if "name" in data: patient["name"] = data["name"]
        if "age" in data: patient["age"] = int(data["age"])
        if "sex" in data: patient["sex"] = data["sex"]
        if "condition" in data: patient["condition"] = data["condition"]
        if "status" in data: patient["status"] = data["status"]
        if "allergies" in data: 
            patient["allergies"] = data["allergies"] if isinstance(data["allergies"], list) else [data["allergies"]]
        if "renal_function" in data: patient["renal_function"] = data["renal_function"]
        if "hepatic_function" in data: patient["hepatic_function"] = data["hepatic_function"]
        
        # Update weight/height and recalculate derived metrics
        if "weight" in data:
            patient["weight"] = float(data["weight"])
        if "height" in data:
            patient["height"] = float(data["height"])
        
        # Recalculate BMI and BSA if weight or height changed
        if "weight" in data or "height" in data:
            w = patient.get("weight", 70)
            h = patient.get("height", 170)
            patient["bmi"] = round(w / ((h/100) ** 2), 1)
            patient["bsa"] = round(0.007184 * (w ** 0.425) * (h ** 0.725), 2)
        
        # Update pharmacogenomics
        if "pharmacogenomics" in data:
            if "pharmacogenomics" not in patient:
                patient["pharmacogenomics"] = {}
            patient["pharmacogenomics"].update(data["pharmacogenomics"])
        
        if "vitals" in data:
            patient["vitals"].update(data["vitals"])
            patient["vitals"]["recorded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Update supplements
        if "supplements" in data:
            patient["supplements"] = data["supplements"]
        
        return patient
    
    def add_medication(self, patient_id, med_data):
        """Add a new medication to patient's current meds"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        new_med = {
            "name": med_data.get("name"),
            "class": med_data.get("class"),
            "cost": med_data.get("cost", 0),
            "start_date": datetime.now().strftime("%Y-%m-%d")
        }
        patient["current_medications"].append(new_med)
        return patient
    
    def stop_medication(self, patient_id, med_name, reason=""):
        """Stop a medication and move it to history"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        
        # Find and remove from current meds
        med_to_stop = None
        for i, med in enumerate(patient["current_medications"]):
            if med.get("name") == med_name or (isinstance(med, dict) and med.get("name") == med_name):
                med_to_stop = patient["current_medications"].pop(i)
                break
        
        if med_to_stop:
            # Add to history
            if "medication_history" not in patient:
                patient["medication_history"] = []
            
            history_entry = {
                **med_to_stop,
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "reason_stopped": reason
            }
            patient["medication_history"].append(history_entry)
        
        return patient
    
    def add_timeline_event(self, patient_id, event_data):
        """Add a new event to patient's medical timeline"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        new_event = {
            "date": event_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "event": event_data.get("event"),
            "notes": event_data.get("notes", ""),
            "attachments": event_data.get("attachments", [])  # NEW: Support attachments
        }
        patient["history"].append(new_event)
        # Re-sort history by date (most recent first)
        patient["history"] = sorted(patient["history"], key=lambda x: x['date'], reverse=True)
        return patient
    
    def update_timeline_event(self, patient_id, event_index, event_data):
        """Update an existing timeline event"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        if event_index < 0 or event_index >= len(patient["history"]):
            return None
        
        if "date" in event_data:
            patient["history"][event_index]["date"] = event_data["date"]
        if "event" in event_data:
            patient["history"][event_index]["event"] = event_data["event"]
        if "notes" in event_data:
            patient["history"][event_index]["notes"] = event_data["notes"]
        
        # Re-sort after update
        patient["history"] = sorted(patient["history"], key=lambda x: x['date'], reverse=True)
        return patient
    
    def delete_timeline_event(self, patient_id, event_index):
        """Delete a timeline event"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        if event_index < 0 or event_index >= len(patient["history"]):
            return None
        
        patient["history"].pop(event_index)
        return patient
    
    def add_lab_report(self, patient_id, lab_data):
        """Add a lab report to patient's records"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        
        # Ensure lab_reports exists
        if "lab_reports" not in patient:
            patient["lab_reports"] = []
        
        lab_report = {
            "id": len(patient["lab_reports"]) + 1,
            "date": lab_data.get("test_date", datetime.now().strftime("%Y-%m-%d")),
            "tests": lab_data.get("tests", []),
            "overall_summary": lab_data.get("overall_summary", ""),
            "interpretation": lab_data.get("interpretation", ""),
            "analyzed_at": lab_data.get("analyzed_at", datetime.now().isoformat()),
            "raw_text": lab_data.get("raw_text", ""),
            "image_files": lab_data.get("image_files", [])  # NEW: Store uploaded images
        }
        
        patient["lab_reports"].append(lab_report)
        # Sort by date (most recent first)
        patient["lab_reports"] = sorted(patient["lab_reports"], key=lambda x: x['date'], reverse=True)
        return patient
    
    def add_radiology_study(self, patient_id, radiology_data):
        """Add a radiology study to patient's records"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        
        # Ensure radiology_studies exists
        if "radiology_studies" not in patient:
            patient["radiology_studies"] = []
        
        study = {
            "id": len(patient["radiology_studies"]) + 1,
            "study_type": radiology_data.get("study_type", "Unknown"),
            "study_date": radiology_data.get("study_date", datetime.now().strftime("%Y-%m-%d")),
            "indication": radiology_data.get("indication", ""),
            "findings": radiology_data.get("findings", []),
            "impression": radiology_data.get("impression", ""),
            "critical_findings": radiology_data.get("critical_findings", []),
            "recommendations": radiology_data.get("recommendations", []),
            "analyzed_at": radiology_data.get("analyzed_at", datetime.now().isoformat()),
            "raw_text": radiology_data.get("raw_text", ""),
            "image_files": radiology_data.get("image_files", [])  # NEW: Store uploaded images
        }
        
        patient["radiology_studies"].append(study)
        # Sort by date (most recent first)
        patient["radiology_studies"] = sorted(patient["radiology_studies"], key=lambda x: x['study_date'], reverse=True)
        return patient
    
    def get_lab_reports(self, patient_id):
        """Get all lab reports for a patient"""
        patient = self.get_patient(patient_id)
        if not patient:
            return []
        return patient.get("lab_reports", [])
    
    
    def get_radiology_studies(self, patient_id):
        """Get all radiology studies for a patient"""
        patient = self.get_patient(patient_id)
        if not patient:
            return []
        return patient.get("radiology_studies", [])
    
    def add_timeline_attachment(self, patient_id, event_index, file_info):
        """Add an attachment to a timeline event"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        if event_index < 0 or event_index >= len(patient["history"]):
            return None
        
        # Ensure attachments array exists
        if "attachments" not in patient["history"][event_index]:
            patient["history"][event_index]["attachments"] = []
        
        patient["history"][event_index]["attachments"].append(file_info)
        return patient
    
    def get_timeline_attachments(self, patient_id, event_index):
        """Get all attachments for a timeline event"""
        patient = self.get_patient(patient_id)
        if not patient or event_index < 0 or event_index >= len(patient["history"]):
            return []
        return patient["history"][event_index].get("attachments", [])
    
    def add_supplement(self, patient_id, supplement_data):
        """Add a supplement/herbal medicine to patient's current list"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        if "supplements" not in patient:
            patient["supplements"] = []
        
        new_supplement = {
            "name": supplement_data.get("name"),
            "category": supplement_data.get("category", "Supplement"),
            "purpose": supplement_data.get("purpose", ""),
            "cyp_impact": supplement_data.get("cyp_impact", "Unknown"),
            "start_date": supplement_data.get("start_date", datetime.now().strftime("%Y-%m-%d"))
        }
        patient["supplements"].append(new_supplement)
        return patient
    
    def remove_supplement(self, patient_id, supplement_name):
        """Remove a supplement from patient's list"""
        if patient_id not in self.patients:
            return None
        
        patient = self.patients[patient_id]
        if "supplements" not in patient:
            return patient
        
        patient["supplements"] = [s for s in patient["supplements"] if s.get("name") != supplement_name]
        return patient
    
    def get_supplements(self, patient_id):
        """Get all supplements for a patient"""
        patient = self.get_patient(patient_id)
        if not patient:
            return []
        return patient.get("supplements", [])

    def delete_patient(self, patient_id):
        if patient_id in self.patients:
            del self.patients[patient_id]
            return True
        return False


generator = DataGenerator()
