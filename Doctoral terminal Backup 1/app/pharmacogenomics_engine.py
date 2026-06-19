"""
Pharmacogenomics Engine - CYP450 Drug Metabolism & Weight-Based Dosing

Provides clinical decision support for:
1. CYP450 enzyme metabolizer status → Drug dose adjustments
2. Weight-based dosing calculations for critical medications
3. Drug-gene interaction alerts
4. Personalized dosing recommendations

This is a professional healthcare tool for clinical use.
"""

from typing import Dict, List, Optional, Any


class PharmacogenomicsEngine:
    """
    Clinical pharmacogenomics analysis engine.
    
    Analyzes patient genetic data to:
    - Flag drugs affected by CYP450 variants
    - Calculate weight-adjusted dosing
    - Provide evidence-based dosing recommendations
    """
    
    # CYP450 enzyme -> drugs metabolized by that enzyme
    CYP_DRUG_SUBSTRATES = {
        "CYP2D6": {
            "drugs": [
                "Codeine", "Tramadol", "Hydrocodone", "Oxycodone",  # Opioids
                "Metoprolol", "Carvedilol", "Propranolol",  # Beta-blockers
                "Fluoxetine", "Paroxetine", "Venlafaxine", "Amitriptyline",  # Antidepressants
                "Risperidone", "Haloperidol", "Aripiprazole",  # Antipsychotics
                "Tamoxifen",  # Oncology
                "Ondansetron",  # Antiemetic
            ],
            "clinical_impact": {
                "Poor": {
                    "effect": "Reduced drug activation or increased toxicity",
                    "recommendation": "Reduce dose 25-50% or use alternative",
                    "alert_level": "HIGH"
                },
                "Intermediate": {
                    "effect": "Reduced metabolism, potential toxicity",
                    "recommendation": "Consider 25% dose reduction and monitor",
                    "alert_level": "MODERATE"
                },
                "Normal": {
                    "effect": "Standard metabolism",
                    "recommendation": "Standard dosing",
                    "alert_level": None
                },
                "Rapid": {
                    "effect": "Increased metabolism, reduced efficacy",
                    "recommendation": "May need higher doses or more frequent dosing",
                    "alert_level": "MODERATE"
                },
                "Ultra-Rapid": {
                    "effect": "Very rapid metabolism, drug failure or toxicity (prodrugs)",
                    "recommendation": "USE ALTERNATIVE DRUG. Codeine contraindicated.",
                    "alert_level": "CRITICAL"
                }
            },
            "special_notes": {
                "Codeine": "Ultra-rapid metabolizers: ⚠️ CONTRAINDICATED (risk of fatal respiratory depression)",
                "Tamoxifen": "Poor metabolizers: Reduced conversion to active metabolite, consider aromatase inhibitor"
            }
        },
        "CYP2C19": {
            "drugs": [
                "Clopidogrel",  # Antiplatelet
                "Omeprazole", "Esomeprazole", "Pantoprazole", "Lansoprazole",  # PPIs
                "Escitalopram", "Citalopram", "Sertraline",  # Antidepressants
                "Diazepam", "Clobazam",  # Benzodiazepines
                "Voriconazole",  # Antifungal
            ],
            "clinical_impact": {
                "Poor": {
                    "effect": "Clopidogrel: REDUCED EFFICACY (prodrug). PPIs: Increased levels.",
                    "recommendation": "Clopidogrel: Use prasugrel or ticagrelor. PPIs: Reduce dose.",
                    "alert_level": "HIGH"
                },
                "Intermediate": {
                    "effect": "Reduced Clopidogrel activation",
                    "recommendation": "Consider alternative antiplatelet if stent",
                    "alert_level": "MODERATE"
                },
                "Normal": {
                    "effect": "Standard metabolism",
                    "recommendation": "Standard dosing",
                    "alert_level": None
                },
                "Rapid": {
                    "effect": "PPIs: Reduced efficacy",
                    "recommendation": "May need higher PPI doses",
                    "alert_level": "LOW"
                },
                "Ultra-Rapid": {
                    "effect": "PPIs: Significantly reduced efficacy",
                    "recommendation": "Use higher PPI doses or alternative acid suppression",
                    "alert_level": "MODERATE"
                }
            }
        },
        "CYP2C9": {
            "drugs": [
                "Warfarin",  # Anticoagulant
                "Phenytoin",  # Antiepileptic
                "Losartan",  # ARB
                "Celecoxib", "Ibuprofen", "Naproxen",  # NSAIDs
                "Glipizide", "Tolbutamide",  # Sulfonylureas
            ],
            "clinical_impact": {
                "Poor": {
                    "effect": "Warfarin: HIGH BLEEDING RISK",
                    "recommendation": "Warfarin: Start 50% lower dose. Use alternative NSAIDs.",
                    "alert_level": "CRITICAL"
                },
                "Intermediate": {
                    "effect": "Increased drug exposure",
                    "recommendation": "Reduce initial dose 25%, monitor closely",
                    "alert_level": "HIGH"
                },
                "Normal": {
                    "effect": "Standard metabolism",
                    "recommendation": "Standard dosing",
                    "alert_level": None
                },
                "Rapid": {
                    "effect": "Faster clearance",
                    "recommendation": "Standard dosing, may need higher maintenance",
                    "alert_level": None
                },
                "Ultra-Rapid": {
                    "effect": "Very fast clearance",
                    "recommendation": "May need higher doses for efficacy",
                    "alert_level": "LOW"
                }
            }
        },
        "CYP3A4": {
            "drugs": [
                "Atorvastatin", "Simvastatin", "Lovastatin",  # Statins
                "Amlodipine", "Diltiazem", "Verapamil", "Nifedipine",  # CCBs
                "Midazolam", "Alprazolam", "Triazolam",  # Benzodiazepines
                "Cyclosporine", "Tacrolimus",  # Immunosuppressants
                "Fentanyl",  # Opioid
                "Apixaban", "Rivaroxaban",  # DOACs
                "Clarithromycin", "Erythromycin",  # Macrolides
            ],
            "clinical_impact": {
                "Poor": {
                    "effect": "Significantly increased drug levels",
                    "recommendation": "Reduce doses 50%, avoid CYP3A4 inhibitors",
                    "alert_level": "HIGH"
                },
                "Intermediate": {
                    "effect": "Moderately increased levels",
                    "recommendation": "Consider dose reduction, monitor for toxicity",
                    "alert_level": "MODERATE"
                },
                "Normal": {
                    "effect": "Standard metabolism",
                    "recommendation": "Standard dosing",
                    "alert_level": None
                },
                "Rapid": {
                    "effect": "Reduced drug levels",
                    "recommendation": "May need higher doses",
                    "alert_level": "LOW"
                },
                "Ultra-Rapid": {
                    "effect": "Very low drug levels, therapeutic failure risk",
                    "recommendation": "Use higher doses or alternative drugs",
                    "alert_level": "MODERATE"
                }
            }
        },
        "CYP1A2": {
            "drugs": [
                "Theophylline", "Caffeine",
                "Clozapine", "Olanzapine",
                "Duloxetine",
                "Tizanidine",
            ],
            "clinical_impact": {
                "Poor": {
                    "effect": "Elevated drug levels, toxicity risk",
                    "recommendation": "Reduce doses, especially theophylline and clozapine",
                    "alert_level": "HIGH"
                },
                "Intermediate": {
                    "effect": "Increased drug exposure",
                    "recommendation": "Consider dose reduction",
                    "alert_level": "MODERATE"
                },
                "Normal": {
                    "effect": "Standard metabolism",
                    "recommendation": "Standard dosing",
                    "alert_level": None
                },
                "Rapid": {
                    "effect": "Faster drug clearance",
                    "recommendation": "May need higher doses (smokers often induce CYP1A2)",
                    "alert_level": "LOW"
                },
                "Ultra-Rapid": {
                    "effect": "Very rapid clearance",
                    "recommendation": "Higher doses may be needed",
                    "alert_level": "MODERATE"
                }
            }
        }
    }
    
    # VKORC1 affects warfarin sensitivity
    VKORC1_WARFARIN = {
        "Normal Sensitivity": {"effect": "Standard warfarin dose", "adjustment": 1.0, "alert_level": None},
        "High Sensitivity": {"effect": "REDUCED WARFARIN DOSE NEEDED", "adjustment": 0.5, "alert_level": "HIGH"},
        "Low Sensitivity": {"effect": "Higher warfarin dose needed", "adjustment": 1.5, "alert_level": "MODERATE"}
    }
    
    # HLA-B*57:01 - Abacavir hypersensitivity
    HLA_B5701_DRUGS = {
        "Abacavir": {
            True: {"effect": "⚠️ CONTRAINDICATED - HIGH RISK OF HYPERSENSITIVITY REACTION", "alert_level": "CRITICAL"},
            False: {"effect": "Standard dosing permitted", "alert_level": None}
        }
    }
    
    # TPMT - Thiopurine drugs
    TPMT_DRUGS = {
        "drugs": ["Azathioprine", "Mercaptopurine", "Thioguanine"],
        "impact": {
            "Normal": {"effect": "Standard metabolism", "recommendation": "Standard dosing", "alert_level": None},
            "Intermediate": {"effect": "Reduced TPMT activity", "recommendation": "Reduce dose 50%", "alert_level": "HIGH"},
            "Deficient": {"effect": "⚠️ SEVERE MYELOSUPPRESSION RISK", "recommendation": "Reduce dose 90% or AVOID", "alert_level": "CRITICAL"}
        }
    }
    
    # Weight-based dosing for critical medications (dose per kg)
    WEIGHT_BASED_DRUGS = {
        # Anticoagulants
        "Enoxaparin": {
            "indication": "Treatment dose",
            "dose_mg_per_kg": 1.0,
            "frequency": "q12h",
            "max_dose_mg": 150,
            "renal_adjustment": {"CrCl<30": "1 mg/kg q24h"},
            "obesity_note": "Use actual body weight up to 144 kg; beyond that, consult anti-Xa levels"
        },
        "Heparin": {
            "indication": "Loading dose",
            "dose_units_per_kg": 80,
            "max_dose_units": 10000,
            "frequency": "bolus",
            "obesity_note": "Use adjusted body weight for morbidly obese"
        },
        # Antibiotics
        "Vancomycin": {
            "indication": "Loading dose",
            "dose_mg_per_kg": 25,
            "max_dose_mg": 3000,
            "frequency": "loading",
            "target": "Trough 15-20 for serious infections",
            "obesity_note": "Use actual body weight; max single dose 3g"
        },
        "Gentamicin": {
            "indication": "Extended interval",
            "dose_mg_per_kg": 5,
            "max_dose_mg": 700,
            "frequency": "q24h",
            "obesity_note": "Use adjusted body weight = IBW + 0.4(ABW-IBW)"
        },
        "Amikacin": {
            "indication": "Extended interval",
            "dose_mg_per_kg": 15,
            "max_dose_mg": 1500,
            "frequency": "q24h",
            "obesity_note": "Use adjusted body weight"
        },
        # Oncology (BSA-based examples, noted here)
        "Carboplatin": {
            "indication": "Chemotherapy",
            "note": "Calculated by Calvert formula: Dose = AUC × (GFR + 25)",
            "bsa_based": True
        },
        # Pediatric/geriatric
        "Acetaminophen": {
            "indication": "Pediatric",
            "dose_mg_per_kg": 15,
            "max_dose_mg": 1000,
            "frequency": "q4-6h",
            "max_daily_mg": 4000
        },
        "Ibuprofen": {
            "indication": "Pediatric",
            "dose_mg_per_kg": 10,
            "max_dose_mg": 400,
            "frequency": "q6-8h"
        }
    }
    
    def __init__(self):
        pass
    
    def analyze_patient_pharmacogenomics(self, patient: Dict) -> Dict[str, Any]:
        """
        Comprehensive pharmacogenomics analysis for a patient.
        
        Returns structured alerts and recommendations.
        """
        pgx = patient.get("pharmacogenomics", {})
        
        if not pgx:
            return {
                "status": "NOT_TESTED",
                "message": "No pharmacogenomics data available",
                "alerts": [],
                "recommendations": []
            }
        
        alerts = []
        recommendations = []
        affected_drugs = []
        
        # Check each CYP enzyme
        for enzyme, data in self.CYP_DRUG_SUBSTRATES.items():
            status = pgx.get(enzyme, "Not Tested")
            if status == "Not Tested" or status not in data["clinical_impact"]:
                continue
            
            impact = data["clinical_impact"][status]
            
            if impact.get("alert_level") in ["HIGH", "CRITICAL", "MODERATE"]:
                alerts.append({
                    "enzyme": enzyme,
                    "status": status,
                    "effect": impact["effect"],
                    "recommendation": impact["recommendation"],
                    "alert_level": impact["alert_level"],
                    "affected_drugs": data["drugs"]
                })
                affected_drugs.extend(data["drugs"])
        
        # Check VKORC1 for warfarin
        vkorc1 = pgx.get("VKORC1", "Not Tested")
        if vkorc1 in self.VKORC1_WARFARIN:
            info = self.VKORC1_WARFARIN[vkorc1]
            if info.get("alert_level"):
                alerts.append({
                    "marker": "VKORC1",
                    "status": vkorc1,
                    "effect": info["effect"],
                    "dose_adjustment": info["adjustment"],
                    "alert_level": info["alert_level"],
                    "affected_drugs": ["Warfarin"]
                })
        
        # Check HLA-B*57:01 for Abacavir
        hla = pgx.get("HLA_B5701")
        if hla is True:
            alerts.append({
                "marker": "HLA-B*57:01",
                "status": "Positive",
                "effect": "⚠️ ABACAVIR CONTRAINDICATED",
                "alert_level": "CRITICAL",
                "affected_drugs": ["Abacavir"]
            })
        
        # Check TPMT
        tpmt = pgx.get("TPMT", "Not Tested")
        if tpmt in self.TPMT_DRUGS["impact"]:
            info = self.TPMT_DRUGS["impact"][tpmt]
            if info.get("alert_level"):
                alerts.append({
                    "marker": "TPMT",
                    "status": tpmt,
                    "effect": info["effect"],
                    "recommendation": info["recommendation"],
                    "alert_level": info["alert_level"],
                    "affected_drugs": self.TPMT_DRUGS["drugs"]
                })
        
        # Sort alerts by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        alerts.sort(key=lambda x: severity_order.get(x.get("alert_level", "LOW"), 4))
        
        return {
            "status": "ANALYZED",
            "tested_date": pgx.get("tested_date"),
            "alerts": alerts,
            "total_affected_drugs": list(set(affected_drugs)),
            "critical_count": sum(1 for a in alerts if a.get("alert_level") == "CRITICAL"),
            "high_count": sum(1 for a in alerts if a.get("alert_level") == "HIGH")
        }
    
    def check_drug_pgx_interaction(self, patient: Dict, drug_name: str) -> Optional[Dict]:
        """
        Check if a specific drug has pharmacogenomic implications for this patient.
        
        Returns alert if found, None otherwise.
        """
        pgx = patient.get("pharmacogenomics", {})
        drug_lower = drug_name.lower()
        
        # Check each CYP enzyme
        for enzyme, data in self.CYP_DRUG_SUBSTRATES.items():
            drugs_lower = [d.lower() for d in data["drugs"]]
            if any(drug_lower in d or d in drug_lower for d in drugs_lower):
                status = pgx.get(enzyme, "Not Tested")
                if status != "Not Tested" and status in data["clinical_impact"]:
                    impact = data["clinical_impact"][status]
                    if impact.get("alert_level"):
                        # Check for special notes
                        special = None
                        for special_drug, note in data.get("special_notes", {}).items():
                            if special_drug.lower() in drug_lower:
                                special = note
                        
                        return {
                            "drug": drug_name,
                            "enzyme": enzyme,
                            "patient_status": status,
                            "effect": impact["effect"],
                            "recommendation": impact["recommendation"],
                            "alert_level": impact["alert_level"],
                            "special_note": special
                        }
        
        # Check specific markers
        if "abacavir" in drug_lower and pgx.get("HLA_B5701") is True:
            return {
                "drug": drug_name,
                "marker": "HLA-B*57:01",
                "patient_status": "Positive",
                "effect": "⚠️ CONTRAINDICATED - HIGH RISK OF HYPERSENSITIVITY",
                "recommendation": "DO NOT USE. Select alternative antiretroviral.",
                "alert_level": "CRITICAL"
            }
        
        tpmt_drugs = [d.lower() for d in self.TPMT_DRUGS["drugs"]]
        if any(d in drug_lower for d in tpmt_drugs):
            tpmt_status = pgx.get("TPMT", "Not Tested")
            if tpmt_status in self.TPMT_DRUGS["impact"]:
                info = self.TPMT_DRUGS["impact"][tpmt_status]
                if info.get("alert_level"):
                    return {
                        "drug": drug_name,
                        "marker": "TPMT",
                        "patient_status": tpmt_status,
                        "effect": info["effect"],
                        "recommendation": info["recommendation"],
                        "alert_level": info["alert_level"]
                    }
        
        return None
    
    def calculate_weight_based_dose(self, patient: Dict, drug_name: str) -> Optional[Dict]:
        """
        Calculate weight-based dose for applicable medications.
        
        Returns dosing recommendation or None if drug not in weight-based list.
        """
        drug_lower = drug_name.lower()
        weight = patient.get("weight", 70)
        bsa = patient.get("bsa", 1.73)
        renal_function = patient.get("renal_function", "Normal")
        
        for drug, info in self.WEIGHT_BASED_DRUGS.items():
            if drug.lower() in drug_lower or drug_lower in drug.lower():
                result = {
                    "drug": drug,
                    "patient_weight_kg": weight,
                    "patient_bsa": bsa,
                    "indication": info.get("indication"),
                }
                
                if info.get("bsa_based"):
                    result["note"] = info.get("note")
                    result["calculation_type"] = "BSA-based"
                elif "dose_mg_per_kg" in info:
                    dose_per_kg = info["dose_mg_per_kg"]
                    calculated = weight * dose_per_kg
                    max_dose = info.get("max_dose_mg")
                    
                    if max_dose and calculated > max_dose:
                        calculated = max_dose
                        result["capped_at_max"] = True
                    
                    result["calculated_dose_mg"] = round(calculated, 1)
                    result["frequency"] = info.get("frequency", "")
                    result["max_dose_mg"] = max_dose
                    result["calculation_type"] = "Weight-based"
                    
                    # Renal adjustment
                    if "renal_adjustment" in info and "Severe" in renal_function:
                        result["renal_adjustment"] = info["renal_adjustment"]
                        result["renal_warning"] = "⚠️ Dose adjustment needed for renal impairment"
                    
                elif "dose_units_per_kg" in info:
                    dose_per_kg = info["dose_units_per_kg"]
                    calculated = weight * dose_per_kg
                    max_dose = info.get("max_dose_units")
                    
                    if max_dose and calculated > max_dose:
                        calculated = max_dose
                        result["capped_at_max"] = True
                    
                    result["calculated_dose_units"] = round(calculated)
                    result["frequency"] = info.get("frequency", "")
                    result["calculation_type"] = "Weight-based (units)"
                
                # Add obesity note if relevant
                if weight > 100 and "obesity_note" in info:
                    result["obesity_warning"] = info["obesity_note"]
                
                return result
        
        return None
    
    def get_clinical_safety_summary(self, patient: Dict) -> Dict:
        """
        Generate a comprehensive clinical safety summary for display.
        
        Combines pharmacogenomics, weight-based considerations, and organ function.
        """
        pgx_analysis = self.analyze_patient_pharmacogenomics(patient)
        
        weight = patient.get("weight", 70)
        height = patient.get("height", 170)
        bmi = patient.get("bmi", 24)
        age = patient.get("age", 50)
        renal = patient.get("renal_function", "Normal")
        hepatic = patient.get("hepatic_function", "Normal")
        
        # Determine patient category flags
        flags = []
        
        if age >= 65:
            flags.append({
                "type": "GERIATRIC",
                "message": "Elderly patient - Start low, go slow",
                "impact": "Increased drug sensitivity, fall risk"
            })
        
        if age >= 80:
            flags.append({
                "type": "VERY_ELDERLY",
                "message": "Very elderly - Beers Criteria applies",
                "impact": "Avoid anticholinergics, long-acting benzodiazepines"
            })
        
        if bmi >= 35:
            flags.append({
                "type": "OBESE",
                "message": f"Obesity (BMI {bmi}) - Weight-based dosing may need adjustment",
                "impact": "Use adjusted body weight for aminoglycosides, actual weight for enoxaparin"
            })
        
        if bmi >= 40:
            flags.append({
                "type": "MORBID_OBESITY", 
                "message": "Morbid obesity - Drug distribution significantly altered",
                "impact": "Consult clinical pharmacist for complex dosing"
            })
        
        if bmi < 18.5:
            flags.append({
                "type": "UNDERWEIGHT",
                "message": f"Underweight (BMI {bmi}) - Increased risk of toxicity",
                "impact": "Use conservative dosing"
            })
        
        if "Severe" in renal:
            flags.append({
                "type": "RENAL_IMPAIRMENT",
                "message": "Severe renal impairment",
                "impact": "Reduce doses of renally excreted drugs, avoid nephrotoxins"
            })
        
        if "Moderate" in hepatic or "Severe" in hepatic:
            flags.append({
                "type": "HEPATIC_IMPAIRMENT",
                "message": f"{hepatic} hepatic function",
                "impact": "Reduce doses of hepatically metabolized drugs"
            })
        
        return {
            "pharmacogenomics": pgx_analysis,
            "flags": flags,
            "weight_kg": weight,
            "height_cm": height,
            "bmi": bmi,
            "bsa": patient.get("bsa", round(0.007184 * (weight ** 0.425) * (height ** 0.725), 2)),
            "age": age,
            "renal_function": renal,
            "hepatic_function": hepatic,
            "summary": self._generate_summary_text(pgx_analysis, flags)
        }
    
    def _generate_summary_text(self, pgx_analysis: Dict, flags: List) -> str:
        """Generate human-readable summary."""
        parts = []
        
        critical_pgx = pgx_analysis.get("critical_count", 0)
        if critical_pgx > 0:
            parts.append(f"⚠️ {critical_pgx} CRITICAL pharmacogenomic alert(s)")
        
        high_pgx = pgx_analysis.get("high_count", 0)
        if high_pgx > 0:
            parts.append(f"🟠 {high_pgx} high-priority pharmacogenomic alert(s)")
        
        for flag in flags:
            if flag["type"] in ["GERIATRIC", "MORBID_OBESITY", "RENAL_IMPAIRMENT"]:
                parts.append(f"⚡ {flag['message']}")
        
        if not parts:
            return "✅ No major pharmacogenomic or dosing concerns identified"
        
        return " | ".join(parts)


# Singleton instance
pharmacogenomics_engine = PharmacogenomicsEngine()
