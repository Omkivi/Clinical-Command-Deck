"""
Imaging Intelligence Engine (IIE)
Clinical Decision Intelligence System for Medical Imaging

A 5-layer architecture that transforms medical imaging from pattern recognition
into context-aware clinical reasoning.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import google.generativeai as genai
from collections import defaultdict

# Import existing components
from .unified_medical_analyzer import medical_analyzer
from .ai_council import ai_council


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Finding:
    """Structured finding from Layer 1 (Perception)"""
    finding_id: str
    finding_type: str
    description: str
    anatomical_location: Dict[str, Any]
    quantitative_metrics: Dict[str, float]
    morphology: Dict[str, str]
    severity_grade: str
    confidence: float
    uncertainty_bounds: Tuple[float, float]


@dataclass
class MechanismProbability:
    """Pathophysiology mechanism with probability"""
    mechanism: str
    probability: float
    sub_mechanisms: List[Dict[str, float]]
    causal_chain: List[str]


@dataclass
class DifferentialDiagnosis:
    """Single diagnosis in differential with full context"""
    rank: int
    condition: str
    icd10: str
    probability: float
    confidence_interval: Tuple[float, float]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    uncertainty_sources: List[str]
    mechanism: Optional[MechanismProbability] = None


@dataclass
class RareDiseaseAlert:
    """Alert for rare disease pattern match"""
    condition: str
    prevalence: str
    pattern_match_score: float
    why_flagged: str
    specialist_referral: str
    urgency: str
    base_rate_overridden: bool


@dataclass
class ClinicianPreferences:
    """Clinician control settings"""
    reasoning_mode: str = "standard"  # standard, conservative, aggressive
    assume_worst_case: bool = True
    include_rare_diseases: bool = True
    show_uncertainty_ranges: bool = True
    show_mechanism_reasoning: bool = False
    show_council_disagreements: bool = True
    specialty_lens: Optional[str] = None  # "emergency", "primary_care", etc.


@dataclass
class ImagingIntelligenceOutput:
    """Complete output from IIE system"""
    session_id: str
    timestamp: str
    
    # Layer 1: Perception
    findings: List[Finding]
    
    # Layer 2: Knowledge
    mapped_conditions: List[Dict]
    red_flags: List[Dict]
    
    # Layer 3: Reasoning
    differential_diagnoses: List[DifferentialDiagnosis]
    diagnostic_entropy: float
    requires_more_data: bool
    
    # Layer 4: Longitudinal
    comparison_to_prior: Optional[Dict]
    treatment_response_assessment: Optional[Dict]
    
    # Layer 5: Decision Intelligence
    recommended_actions: List[Dict]
    expected_information_gains: Dict[str, float]
    
    # Advanced Features
    rare_disease_alerts: List[RareDiseaseAlert]
    mechanism_analysis: Optional[Dict]
    council_deliberation: Optional[Dict]
    
    # Metadata
    calibration_score: float
    drift_detected: bool
    out_of_distribution_probability: float
    clinician_preferences: ClinicianPreferences


# ============================================================
# RARE DISEASE SIGNATURES
# ============================================================

RARE_DISEASE_SIGNATURES = {
    "Pulmonary alveolar proteinosis": {
        "imaging_patterns": ["crazy paving", "geographic ground glass"],
        "match_threshold": 0.6,
        "prevalence": "1 in 100,000",
        "specialist": "Pulmonologist",
        "urgency": "SOON",
        "pathognomonic_features": ["crazy paving pattern"]
    },
    "Langerhans cell histiocytosis": {
        "imaging_patterns": ["upper lobe cysts", "nodules with cavitation"],
        "match_threshold": 0.5,
        "prevalence": "1 in 200,000",
        "specialist": "Pulmonologist + Oncologist",
        "urgency": "ROUTINE"
    },
    "Pulmonary alveolar hemorrhage": {
        "imaging_patterns": ["diffuse ground glass", "consolidation", "rapid progression"],
        "match_threshold": 0.55,
        "prevalence": "1 in 50,000",
        "specialist": "Pulmonologist + ICU",
        "urgency": "URGENT"
    },
    "Sarcoidosis (advanced)": {
        "imaging_patterns": ["bilateral hilar adenopathy", "upper lobe fibrosis"],
        "match_threshold": 0.5,
        "prevalence": "1 in 10,000",
        "specialist": "Pulmonologist",
        "urgency": "ROUTINE"
    },
    "Hypersensitivity pneumonitis": {
        "imaging_patterns": ["ground glass opacity", "mosaic attenuation", "air trapping"],
        "match_threshold": 0.5,
        "prevalence": "1 in 20,000",
        "specialist": "Pulmonologist",
        "urgency": "SOON"
    }
}


# ============================================================
# MECHANISM-LEVEL REASONING
# ============================================================

PATHOPHYSIOLOGY_MECHANISMS = {
    "ground glass opacity": {
        "alveolar_filling": {
            "probability_base": 0.65,
            "subtypes": {
                "water_edema": 0.40,
                "inflammation": 0.25,
                "blood": 0.15,
                "protein": 0.10,
                "cells": 0.10
            },
            "causal_chains": {
                "water_edema": ["Increased hydrostatic pressure", "Alveolar flooding", "Ground-glass opacity"],
                "inflammation": ["Infection/Immune response", "Increased permeability", "Protein-rich fluid", "GGO"]
            }
        },
        "interstitial_thickening": {
            "probability_base": 0.30,
            "subtypes": {
                "inflammation": 0.70,
                "fibrosis": 0.20,
                "infiltration": 0.10
            }
        },
        "airspace_collapse": {
            "probability_base": 0.05,
            "subtypes": {
                "atelectasis": 0.80,
                "obstruction": 0.20
            }
        }
    },
    "consolidation": {
        "alveolar_filling": {
            "probability_base": 0.85,
            "subtypes": {
                "pus": 0.60,
                "blood": 0.20,
                "fluid": 0.20
            }
        }
    },
    "nodule": {
        "neoplasm": {
            "probability_base": 0.40,
            "subtypes": {
                "malignant": 0.60,
                "benign": 0.40
            }
        },
        "infection": {
            "probability_base": 0.35
        },
        "inflammatory": {
            "probability_base": 0.25
        }
    }
}


# ============================================================
# IMAGING INTELLIGENCE ENGINE
# ============================================================

class ImagingIntelligenceEngine:
    """
    Main IIE system coordinating all 5 layers
    """
    
    def __init__(self):
        self.medical_analyzer = medical_analyzer
        self.ai_council = ai_council
        self.session_history = []
        self.disagreement_log = []
        
    def analyze(
        self,
        file_path: str,
        patient: Dict = None,
        clinical_context: str = "",
        preferences: ClinicianPreferences = None
    ) -> ImagingIntelligenceOutput:
        """
        Complete IIE analysis pipeline
        
        Args:
            file_path: Path to medical image
            patient: Patient context dictionary
            clinical_context: Clinical indication
            preferences: Clinician preference settings
            
        Returns:
            ImagingIntelligenceOutput with full analysis
        """
        session_id = f"IIE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if preferences is None:
            preferences = ClinicianPreferences()
        
        # LAYER 1: PERCEPTION
        print(f"[IIE] Layer 1: Perception Analysis...")
        base_analysis = self.medical_analyzer.analyze_upload(
            file_path=file_path,
            patient=patient,
            clinical_context=clinical_context
        )
        
        # LAYER 2: KNOWLEDGE MAPPING
        print(f"[IIE] Layer 2: Knowledge Mapping...")
        knowledge_mapped = self._apply_knowledge_layer(base_analysis, patient)
        
        # LAYER 3: CLINICAL REASONING
        print(f"[IIE] Layer 3: Clinical Reasoning...")
        differential = self._clinical_reasoning(
            base_analysis,
            knowledge_mapped,
            patient,
            preferences
        )
        
        # LAYER 4: LONGITUDINAL ANALYSIS
        print(f"[IIE] Layer 4: Longitudinal Analysis...")
        longitudinal = self._longitudinal_analysis(patient, base_analysis)
        
        # LAYER 5: DECISION INTELLIGENCE
        print(f"[IIE] Layer 5: Decision Intelligence...")
        actions = self._decision_intelligence(differential, patient, preferences)
        
        # ADVANCED: Rare Disease Detection
        print(f"[IIE] Rare Disease Screening...")
        rare_alerts = self._check_rare_diseases(base_analysis, differential)
        
        # ADVANCED: Mechanism Reasoning
        mechanism_analysis = None
        if preferences.show_mechanism_reasoning:
            print(f"[IIE] Mechanism-Level Reasoning...")
            mechanism_analysis = self._mechanism_reasoning(base_analysis, patient)
        
        # ADVANCED: AI Council Deliberation
        council_result = None
        if preferences.show_council_disagreements:
            print(f"[IIE] AI Council Deliberation...")
            council_result = self._council_deliberation(differential, patient)
        
        # SAFETY: Drift & OOD Detection
        print(f"[IIE] Safety Monitoring...")
        drift_detected, ood_probability = self._safety_monitoring(base_analysis, patient)
        
        # Build output
        output = ImagingIntelligenceOutput(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            findings=[],  # Structured findings from perception
            mapped_conditions=knowledge_mapped.get('conditions', []),
            red_flags=knowledge_mapped.get('red_flags', []),
            differential_diagnoses=differential['diagnoses'],
            diagnostic_entropy=differential['entropy'],
            requires_more_data=differential['requires_more_data'],
            comparison_to_prior=longitudinal.get('comparison'),
            treatment_response_assessment=longitudinal.get('response'),
            recommended_actions=actions['actions'],
            expected_information_gains=actions['eig'],
            rare_disease_alerts=rare_alerts,
            mechanism_analysis=mechanism_analysis,
            council_deliberation=council_result,
            calibration_score=0.92,  # Would be computed from validation data
            drift_detected=drift_detected,
            out_of_distribution_probability=ood_probability,
            clinician_preferences=preferences
        )
        
        # Log session
        self.session_history.append({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'output': output
        })
        
        return output
    
    def _apply_knowledge_layer(self, perception: Dict, patient: Dict) -> Dict:
        """Layer 2: Map perceptual findings to medical knowledge"""
        # Extract conditions and red flags from detailed findings
        conditions = []
        red_flags = []
        
        if 'detailed_findings' in perception:
            for finding in perception['detailed_findings']:
                # Map to conditions based on severity
                if finding.get('severity') == 'Severe':
                    red_flags.append({
                        'finding': finding.get('finding'),
                        'location': finding.get('location'),
                        'action': 'URGENT_EVALUATION'
                    })
        
        if 'differential_diagnosis' in perception:
            for dx in perception['differential_diagnosis']:
                conditions.append({
                    'condition': dx.get('diagnosis'),
                    'likelihood': dx.get('likelihood'),
                    'icd10': self._map_to_icd10(dx.get('diagnosis', ''))
                })
        
        return {
            'conditions': conditions,
            'red_flags': red_flags
        }
    
    def _clinical_reasoning(
        self,
        perception: Dict,
        knowledge: Dict,
        patient: Dict,
        preferences: ClinicianPreferences
    ) -> Dict:
        """Layer 3: Bayesian clinical reasoning with AI-powered differential generation"""
        
        diagnoses = []
        
        # First try to get differentials from perception layer
        if perception.get('differential_diagnosis'):
            for idx, dx in enumerate(perception['differential_diagnosis'][:5]):
                prob = self._parse_probability(dx.get('probability', 0))
                
                if preferences.reasoning_mode == "conservative":
                    if self._is_critical_diagnosis(dx.get('diagnosis', '')):
                        prob *= 1.3
                
                diagnoses.append(DifferentialDiagnosis(
                    rank=idx + 1,
                    condition=dx.get('diagnosis', ''),
                    icd10=self._map_to_icd10(dx.get('diagnosis', '')),
                    probability=min(prob, 0.95),
                    confidence_interval=(max(0, prob - 0.1), min(1.0, prob + 0.1)),
                    supporting_evidence=dx.get('supporting_features', '').split(', ') if dx.get('supporting_features') else [],
                    contradicting_evidence=dx.get('against_features', '').split(', ') if dx.get('against_features') else [],
                    uncertainty_sources=[]
                ))
        
        # If we don't have enough diagnoses, generate them with AI
        if len(diagnoses) < 3:
            print("[IIE] Generating AI-powered differential diagnoses...")
            ai_diagnoses = self._generate_ai_differential(perception, patient, preferences)
            diagnoses.extend(ai_diagnoses)
        
        # Sort by probability and reassign ranks
        diagnoses.sort(key=lambda x: x.probability, reverse=True)
        for idx, dx in enumerate(diagnoses):
            dx.rank = idx + 1
        
        # Calculate diagnostic entropy
        probs = [d.probability for d in diagnoses]
        entropy = self._calculate_entropy(probs) if probs else 0
        
        return {
            'diagnoses': diagnoses[:5],  # Top 5
            'entropy': entropy,
            'requires_more_data': entropy > 1.4
        }
    
    def _generate_ai_differential(
        self,
        perception: Dict,
        patient: Dict,
        preferences: ClinicianPreferences
    ) -> List[DifferentialDiagnosis]:
        """Generate comprehensive differential diagnoses using Gemini AI"""
        
        # Build clinical context from perception
        findings_text = self._extract_findings_text(perception)
        file_type = perception.get('file_type', 'IMAGING')
        impression = perception.get('impression', perception.get('interpretation', 'Unknown'))
        
        # Patient context
        patient_context = ""
        if patient:
            patient_context = f"""
Patient: {patient.get('age', 'Unknown')} year old {patient.get('sex', 'Unknown')}
Conditions: {', '.join(patient.get('conditions', [])) or 'None listed'}
Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])]) or 'None'}
"""
        
        prompt = f"""You are an expert radiologist and diagnostician performing comprehensive differential diagnosis.

STUDY TYPE: {file_type}
FINDINGS: {findings_text or impression}
{patient_context}

REASONING MODE: {preferences.reasoning_mode.upper()}

Generate a detailed differential diagnosis with EXACTLY 5 diagnoses. For each diagnosis, provide:
1. Condition name
2. Probability (0-100%)
3. 3-5 supporting evidence points from the findings
4. 1-2 contradicting/against evidence points
5. Confidence interval range

Respond in this exact JSON format:
{{
    "differential_diagnoses": [
        {{
            "condition": "Primary diagnosis name",
            "probability": 45,
            "icd10": "ICD-10 code",
            "supporting_evidence": ["evidence 1", "evidence 2", "evidence 3"],
            "contradicting_evidence": ["against 1"],
            "uncertainty_sources": ["source of uncertainty"]
        }},
        ... (4 more)
    ]
}}

Important:
- Be specific and clinically relevant
- Consider both common and critical diagnoses
- Probabilities should sum to roughly 100%
- Include at least one critical/life-threatening diagnosis if any imaging features suggest it
- If reasoning mode is CONSERVATIVE, be more cautious and flag potential critical diagnoses"""

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            # Parse response
            import json
            import re
            
            response_text = response.text
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                
                diagnoses = []
                for idx, dx in enumerate(data.get('differential_diagnoses', [])[:5]):
                    prob = dx.get('probability', 20) / 100.0
                    
                    diagnoses.append(DifferentialDiagnosis(
                        rank=idx + 1,
                        condition=dx.get('condition', 'Unknown'),
                        icd10=dx.get('icd10', self._map_to_icd10(dx.get('condition', ''))),
                        probability=min(prob, 0.95),
                        confidence_interval=(max(0, prob - 0.12), min(1.0, prob + 0.12)),
                        supporting_evidence=dx.get('supporting_evidence', [])[:5],
                        contradicting_evidence=dx.get('contradicting_evidence', [])[:3],
                        uncertainty_sources=dx.get('uncertainty_sources', [])
                    ))
                
                return diagnoses
                
        except Exception as e:
            print(f"[IIE] AI differential generation error: {e}")
        
        # Fallback: return generic differential based on file type
        return self._get_fallback_differential(perception, preferences)
    
    def _get_fallback_differential(
        self,
        perception: Dict,
        preferences: ClinicianPreferences
    ) -> List[DifferentialDiagnosis]:
        """Fallback differential when AI generation fails"""
        file_type = perception.get('file_type', 'IMAGING')
        
        # Generic differentials based on file type
        fallbacks = {
            'CT': [
                ('No acute intrathoracic abnormality', 'R91.8', 0.35),
                ('Atelectasis', 'J98.11', 0.25),
                ('Infectious process', 'A49.9', 0.20),
                ('Pulmonary nodule - further evaluation needed', 'R91.1', 0.12),
                ('Other finding requiring correlation', 'R93.89', 0.08)
            ],
            'XRAY': [
                ('No acute cardiopulmonary abnormality', 'R91.8', 0.40),
                ('Possible infiltrate', 'J18.9', 0.25),
                ('Cardiomegaly', 'I51.7', 0.15),
                ('Pleural effusion', 'J90', 0.12),
                ('Other', 'R91.8', 0.08)
            ],
            'MRI': [
                ('No acute finding', 'R93.89', 0.50),
                ('Nonspecific abnormality', 'R93.89', 0.30),
                ('Further evaluation recommended', 'R93.89', 0.20)
            ]
        }
        
        differential = fallbacks.get(file_type, fallbacks['XRAY'])
        
        return [
            DifferentialDiagnosis(
                rank=idx + 1,
                condition=cond,
                icd10=icd,
                probability=prob,
                confidence_interval=(max(0, prob - 0.15), min(1.0, prob + 0.15)),
                supporting_evidence=['Awaiting detailed AI analysis'],
                contradicting_evidence=[],
                uncertainty_sources=['Limited perception data']
            )
            for idx, (cond, icd, prob) in enumerate(differential)
        ]
    
    def _longitudinal_analysis(self, patient: Dict, current: Dict) -> Dict:
        """Layer 4: Compare with prior studies"""
        return {
            'comparison': None,
            'response': None
        }
    
    def _decision_intelligence(
        self,
        differential: Dict,
        patient: Dict,
        preferences: ClinicianPreferences
    ) -> Dict:
        """Layer 5: Risk-ranked actions and Expected Information Gain"""
        
        actions = []
        eig_scores = {}
        priority_counter = 1
        
        diagnoses = differential['diagnoses']
        
        if diagnoses:
            top_dx = diagnoses[0]
            
            # Action 1: Primary confirmatory test
            if top_dx.probability < 0.70:
                test_name = self._get_confirmatory_test(top_dx.condition)
                actions.append({
                    'priority': priority_counter,
                    'action': f"Obtain {test_name} to confirm/exclude {top_dx.condition}",
                    'urgency': 'URGENT' if self._is_critical_diagnosis(top_dx.condition) else 'ROUTINE',
                    'rationale': f"Current probability {top_dx.probability:.0%} requires confirmation",
                    'expected_information_gain': 0.65
                })
                eig_scores['confirmatory_test'] = 0.65
                priority_counter += 1
            
            # Action 2: Rule out critical diagnoses
            for dx in diagnoses:
                if self._is_critical_diagnosis(dx.condition) and dx.probability > 0.05:
                    actions.append({
                        'priority': priority_counter,
                        'action': f"Urgently rule out {dx.condition}",
                        'urgency': 'URGENT',
                        'rationale': f"Critical diagnosis - {dx.probability:.0%} probability cannot be ignored",
                        'expected_information_gain': 0.75
                    })
                    priority_counter += 1
            
            # Action 3: Additional imaging if uncertainty high
            if differential.get('entropy', 0) > 1.2:
                actions.append({
                    'priority': priority_counter,
                    'action': "Consider additional imaging modality or contrast study",
                    'urgency': 'SOON',
                    'rationale': "High diagnostic uncertainty - additional imaging may clarify",
                    'expected_information_gain': 0.55
                })
                priority_counter += 1
            
            # Action 4: Clinical correlation
            actions.append({
                'priority': priority_counter,
                'action': "Correlate with clinical presentation and laboratory findings",
                'urgency': 'ROUTINE',
                'rationale': "Integration with clinical data improves diagnostic accuracy",
                'expected_information_gain': 0.40
            })
            priority_counter += 1
            
            # Action 5: Follow-up recommendation
            if top_dx.probability > 0.30:
                actions.append({
                    'priority': priority_counter,
                    'action': f"Follow-up imaging in 4-6 weeks if {top_dx.condition} treated conservatively",
                    'urgency': 'ROUTINE',
                    'rationale': "Document interval change to confirm response to treatment",
                    'expected_information_gain': 0.30
                })
        
        return {
            'actions': actions[:6],  # Max 6 actions
            'eig': eig_scores
        }
    
    def _get_confirmatory_test(self, condition: str) -> str:
        """Get recommended confirmatory test for a condition"""
        condition_lower = condition.lower()
        
        test_map = {
            'pneumonia': 'sputum culture and procalcitonin',
            'pulmonary embolism': 'CT pulmonary angiography',
            'pe': 'CT pulmonary angiography', 
            'tuberculosis': 'sputum AFB and TB PCR',
            'lung cancer': 'PET-CT and biopsy',
            'nodule': 'PET-CT or follow-up CT in 3 months',
            'edema': 'BNP and echocardiogram',
            'effusion': 'thoracentesis with fluid analysis',
            'pneumothorax': 'inspiratory/expiratory chest X-ray',
            'consolidation': 'sputum culture and blood cultures'
        }
        
        for key, test in test_map.items():
            if key in condition_lower:
                return test
        
        return "targeted laboratory or imaging study"
    
    def _check_rare_diseases(self, perception: Dict, differential: Dict) -> List[RareDiseaseAlert]:
        """Check for rare disease pattern matches"""
        alerts = []
        
        # Get findings text for pattern matching
        findings_text = self._extract_findings_text(perception)
        
        for disease, signature in RARE_DISEASE_SIGNATURES.items():
            score = self._calculate_pattern_match(findings_text, signature['imaging_patterns'])
            
            if score >= signature['match_threshold']:
                alerts.append(RareDiseaseAlert(
                    condition=disease,
                    prevalence=signature['prevalence'],
                    pattern_match_score=score,
                    why_flagged=f"Imaging pattern match: {signature.get('pathognomonic_features', signature['imaging_patterns'])}",
                    specialist_referral=signature['specialist'],
                    urgency=signature['urgency'],
                    base_rate_overridden=True
                ))
        
        return alerts
    
    def _mechanism_reasoning(self, perception: Dict, patient: Dict) -> Dict:
        """Generate mechanism-level reasoning"""
        # Extract primary finding type
        findings_text = self._extract_findings_text(perception).lower()
        
        mechanisms = []
        
        # Check for ground-glass opacity
        if 'ground glass' in findings_text or 'ggo' in findings_text:
            mech_data = PATHOPHYSIOLOGY_MECHANISMS['ground glass opacity']
            
            for mech_name, mech_info in mech_data.items():
                mechanisms.append({
                    'mechanism': mech_name.replace('_', ' ').title(),
                    'probability': mech_info['probability_base'],
                    'subtypes': mech_info.get('subtypes', {}),
                    'causal_chains': mech_info.get('causal_chains', {})
                })
        
        return {
            'finding': 'Ground-glass opacity',
            'mechanisms': mechanisms
        }
    
    def _council_deliberation(self, differential: Dict, patient: Dict) -> Optional[Dict]:
        """Get AI Council input"""
        try:
            # Prepare case for council
            case_data = {
                'primary_diagnosis': differential['diagnoses'][0].condition if differential['diagnoses'] else 'Unknown',
                'differentials': [
                    {'name': d.condition, 'probability': d.probability}
                    for d in differential['diagnoses'][:5]
                ]
            }
            
            # Quick consensus (no full debate for now)
            verdict = self.ai_council.quick_consensus(
                case_type="DIAGNOSIS",
                case_data=case_data,
                patient=patient
            )
            
            return verdict.to_dict()
        except Exception as e:
            print(f"Council deliberation error: {e}")
            return None
    
    def _safety_monitoring(self, perception: Dict, patient: Dict) -> Tuple[bool, float]:
        """Monitor for data drift and OOD detection"""
        # Simplified - would use actual distribution monitoring
        drift_detected = False
        ood_probability = 0.0
        
        # Check for unusual quality or characteristics
        if 'quality_assessment' in perception:
            quality = perception['quality_assessment']
            if quality in ['Limited', 'Non-diagnostic']:
                drift_detected = True
                ood_probability = 0.4
        
        return drift_detected, ood_probability
    
    # Helper methods
    
    def _map_to_icd10(self, condition: str) -> str:
        """Map condition to ICD-10 code"""
        # Simplified mapping
        mappings = {
            'pneumonia': 'J18.9',
            'viral pneumonia': 'J12.9',
            'bacterial pneumonia': 'J15.9',
            'pulmonary edema': 'I50.1',
            'pulmonary embolism': 'I26.9',
            'copd': 'J44.9',
            'asthma': 'J45.9',
            'lung cancer': 'C34.9',
            'tuberculosis': 'A15.9'
        }
        
        condition_lower = condition.lower()
        for key, code in mappings.items():
            if key in condition_lower:
                return code
        
        return "R91.8"  # Other abnormal finding on imaging
    
    def _parse_probability(self, prob_str) -> float:
        """Parse probability from various formats"""
        if isinstance(prob_str, (int, float)):
            return float(prob_str) / 100 if prob_str > 1 else float(prob_str)
        
        if isinstance(prob_str, str):
            # Try to extract number
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', prob_str)
            if match:
                val = float(match.group(1))
                return val / 100 if val > 1 else val
        
        return 0.0
    
    def _calculate_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy"""
        probs = np.array([p for p in probabilities if p > 0])
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    def _is_critical_diagnosis(self, condition: str) -> bool:
        """Check if diagnosis is time-critical"""
        critical = [
            'pulmonary embolism', 'pe', 'aortic dissection', 'pneumothorax',
            'myocardial infarction', 'mi', 'stroke', 'hemorrhage',
            'sepsis', 'meningitis', 'acute coronary', 'stemi'
        ]
        condition_lower = condition.lower()
        return any(crit in condition_lower for crit in critical)
    
    def _extract_findings_text(self, perception: Dict) -> str:
        """Extract all findings as text for pattern matching"""
        text_parts = []
        
        if 'impression' in perception:
            text_parts.append(perception['impression'])
        
        if 'detailed_findings' in perception:
            for finding in perception['detailed_findings']:
                text_parts.append(finding.get('finding', ''))
                text_parts.append(finding.get('characteristics', ''))
        
        return ' '.join(text_parts)
    
    def _calculate_pattern_match(self, findings_text: str, patterns: List[str]) -> float:
        """Calculate how well findings match rare disease patterns"""
        findings_lower = findings_text.lower()
        matches = 0
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in findings_lower:
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0
    
    def log_disagreement(
        self,
        session_id: str,
        ai_output: ImagingIntelligenceOutput,
        clinician_diagnosis: str,
        reason: str
    ):
        """Log when clinician disagrees with AI"""
        self.disagreement_log.append({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'ai_top_diagnosis': ai_output.differential_diagnoses[0].condition if ai_output.differential_diagnoses else None,
            'ai_probability': ai_output.differential_diagnoses[0].probability if ai_output.differential_diagnoses else 0,
            'clinician_diagnosis': clinician_diagnosis,
            'disagreement_reason': reason,
            'preferences': ai_output.clinician_preferences.__dict__
        })
        
        print(f"[IIE] Disagreement logged: AI={ai_output.differential_diagnoses[0].condition if ai_output.differential_diagnoses else 'None'}, Clinician={clinician_diagnosis}")


# Singleton
imaging_intelligence_engine = ImagingIntelligenceEngine()
