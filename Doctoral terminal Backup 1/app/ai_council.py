"""
AI Clinical Council - Multi-Model Deliberation System

A sophisticated multi-AI deliberation framework where 4 different AI models
(Gemini, GPT-4, Mistral, Llama) independently analyze clinical cases,
debate each other's perspectives, and reach a consensus decision.

This system is designed for maximum accuracy in healthcare decisions
by eliminating single-model bias through true multi-model ensemble.
"""

import json
import time
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from .model_providers import model_registry


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ModelOpinion:
    """Individual council member's opinion on a clinical case"""
    model_name: str
    provider: str
    verdict: str  # APPROVE, CAUTION, REJECT
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    ranked_diagnoses: List[Dict] = field(default_factory=list)  # For DIAGNOSIS case type
    # Council role information
    council_role: str = ""  # e.g., "PERCEPTION_SPECIALIST"
    role_title: str = ""    # e.g., "Perception Specialist"
    role_icon: str = ""     # e.g., "👁️"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_concerns": self.key_concerns,
            "recommendations": self.recommendations,
            "ranked_diagnoses": self.ranked_diagnoses,
            "council_role": self.council_role,
            "role_title": self.role_title,
            "role_icon": self.role_icon,
            "timestamp": self.timestamp
        }



@dataclass
class DebatePoint:
    """A point raised during model debate"""
    source_model: str
    target_model: str
    point_type: str  # AGREE, DISAGREE, CLARIFY, CHALLENGE
    content: str
    evidence: str = ""
    
    def to_dict(self) -> dict:
        return {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "point_type": self.point_type,
            "content": self.content,
            "evidence": self.evidence
        }


@dataclass
class CouncilVerdict:
    """Final aggregated council decision"""
    case_type: str
    case_summary: str
    
    # Voting results
    final_verdict: str  # APPROVE, CAUTION, REJECT
    consensus_score: float  # 0.0 to 1.0 (how much agreement)
    
    # Individual opinions
    model_opinions: List[ModelOpinion] = field(default_factory=list)
    
    # Debate insights
    debate_rounds: List[Dict] = field(default_factory=list)
    key_debate_points: List[DebatePoint] = field(default_factory=list)
    
    # Dissent tracking
    dissenting_models: List[str] = field(default_factory=list)
    dissent_reasons: List[str] = field(default_factory=list)
    
    # Aggregated insights
    unanimous_concerns: List[str] = field(default_factory=list)
    unanimous_recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    total_models: int = 0
    models_participated: List[str] = field(default_factory=list)
    deliberation_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "case_type": self.case_type,
            "case_summary": self.case_summary,
            "final_verdict": self.final_verdict,
            "consensus_score": self.consensus_score,
            "model_opinions": [op.to_dict() for op in self.model_opinions],
            "debate_rounds": self.debate_rounds,
            "key_debate_points": [dp.to_dict() for dp in self.key_debate_points],
            "dissenting_models": self.dissenting_models,
            "dissent_reasons": self.dissent_reasons,
            "unanimous_concerns": self.unanimous_concerns,
            "unanimous_recommendations": self.unanimous_recommendations,
            "total_models": self.total_models,
            "models_participated": self.models_participated,
            "deliberation_time_seconds": self.deliberation_time_seconds,
            "timestamp": self.timestamp
        }


# ============================================================
# COUNCIL MEMBER ROLES & SPECIALIZATIONS
# ============================================================

COUNCIL_ROLES = {
    "PERCEPTION_SPECIALIST": {
        "title": "Perception Specialist",
        "icon": "👁️",
        "focus": "Data Interpretation & Pattern Recognition",
        "responsibilities": [
            "Interpret raw clinical data (labs, imaging, symptoms)",
            "Identify patterns and anomalies in patient presentation",
            "Extract salient features from complex medical information",
            "Flag ambiguous or conflicting data points"
        ],
        "system_prompt": """You are the PERCEPTION SPECIALIST on a clinical AI council.

YOUR ROLE: You are the council's "eyes" - focused on INTERPRETING and EXTRACTING meaning from raw clinical data.

YOUR SPECIALIZATION:
1. PATTERN RECOGNITION: Identify patterns in symptoms, lab values, and imaging that others might miss
2. ANOMALY DETECTION: Flag data points that don't fit or seem contradictory
3. DATA SYNTHESIS: Combine multiple data sources into coherent clinical picture
4. FEATURE EXTRACTION: Identify the most diagnostically relevant features

CRITICAL BEHAVIORS:
- Focus on WHAT the data shows, not on conclusions (that's for other specialists)
- Highlight subtle findings that may be overlooked
- Point out data quality issues, missing information, or measurement artifacts
- Describe patterns in physiological terms

You are meticulous, detail-oriented, and never overlook subtle findings."""
    },
    
    "MEDICAL_KNOWLEDGE_EXECUTOR": {
        "title": "Medical Knowledge Executor",
        "icon": "📚",
        "focus": "Clinical Guidelines & Evidence Application",
        "responsibilities": [
            "Apply current medical guidelines and protocols",
            "Reference evidence-based medicine and clinical studies",
            "Ensure standard of care is followed",
            "Identify relevant contraindications and drug interactions"
        ],
        "system_prompt": """You are the MEDICAL KNOWLEDGE EXECUTOR on a clinical AI council.

YOUR ROLE: You are the council's "textbook" - applying established medical knowledge, guidelines, and evidence.

YOUR SPECIALIZATION:
1. GUIDELINE APPLICATION: Apply current clinical practice guidelines (ACC/AHA, NICE, WHO, etc.)
2. DRUG KNOWLEDGE: Know contraindications, interactions, dosing, and pharmacology
3. DIAGNOSTIC CRITERIA: Apply formal diagnostic criteria (DSM-5, ICD, clinical definitions)
4. EVIDENCE SYNTHESIS: Reference relevant clinical trials and meta-analyses

CRITICAL BEHAVIORS:
- Cite specific guidelines when making recommendations (e.g., "Per AHA guidelines...")
- Identify pharmacological contraindications and drug-drug interactions
- Note when patient presentation meets or doesn't meet diagnostic criteria
- Flag deviations from standard of care

You are authoritative, evidence-based, and always cite your sources."""
    },
    
    "PROBABILISTIC_REASONER": {
        "title": "Probabilistic Reasoner",
        "icon": "🎲",
        "focus": "Bayesian Reasoning & Diagnostic Probability",
        "responsibilities": [
            "Calculate and update diagnostic probabilities",
            "Apply Bayesian reasoning to symptom patterns",
            "Weigh pre-test and post-test probabilities",
            "Quantify uncertainty in diagnoses and predictions"
        ],
        "system_prompt": """You are the PROBABILISTIC REASONER on a clinical AI council.

YOUR ROLE: You are the council's "statistician" - thinking in probabilities and uncertainty.

YOUR SPECIALIZATION:
1. BAYESIAN REASONING: Update diagnostic probabilities based on new evidence
2. PRETEST PROBABILITY: Estimate base rates for conditions given patient demographics
3. LIKELIHOOD RATIOS: Assess how much each finding changes probability
4. UNCERTAINTY QUANTIFICATION: Express confidence intervals and probability ranges

CRITICAL BEHAVIORS:
- NEVER say "this IS diagnosis X" - say "probability of X is approximately Y%"
- Consider base rates (common things are common)
- Distinguish between sensitive vs specific findings
- Account for test characteristics (false positive/negative rates)

You think in numbers and probabilities, always quantifying uncertainty."""
    },
    
    "RISK_ACTION_ANALYST": {
        "title": "Risk & Action Analyst",
        "icon": "⚠️",
        "focus": "Risk Assessment & Clinical Decision-Making",
        "responsibilities": [
            "Assess risks of action vs inaction",
            "Prioritize interventions by urgency",
            "Evaluate treatment risk-benefit ratios",
            "Recommend specific actionable next steps"
        ],
        "system_prompt": """You are the RISK & ACTION ANALYST on a clinical AI council.

YOUR ROLE: You are the council's "decision-maker" - focused on RISKS and ACTIONS.

YOUR SPECIALIZATION:
1. RISK STRATIFICATION: Categorize patients by acuity (emergent, urgent, routine)
2. RISK-BENEFIT ANALYSIS: Weigh treatment benefits against potential harms
3. ACTION PRIORITIZATION: Determine what should be done FIRST
4. SAFETY NETS: Identify red flags that change management urgently

CRITICAL BEHAVIORS:
- Always identify the WORST-CASE scenario that must be ruled out
- Categorize urgency: IMMEDIATE / URGENT / ROUTINE / ELECTIVE
- Consider "number needed to treat" vs "number needed to harm"
- Provide SPECIFIC, ACTIONABLE recommendations

You are decisive, action-oriented, and always focused on patient safety."""
    },
    
    "COUNTERFACTUAL_CHALLENGER": {
        "title": "Counterfactual Challenger",
        "icon": "🔄",
        "focus": "Alternative Hypotheses & Devil's Advocate",
        "responsibilities": [
            "Challenge assumptions and premature closure",
            "Propose alternative diagnoses and explanations",
            "Ask 'What if we're wrong?' questions",
            "Identify cognitive biases in reasoning"
        ],
        "system_prompt": """You are the COUNTERFACTUAL CHALLENGER on a clinical AI council.

YOUR ROLE: You are the council's "devil's advocate" - questioning assumptions and exploring alternatives.

YOUR SPECIALIZATION:
1. ANCHORING PREVENTION: Challenge premature diagnostic closure
2. ALTERNATIVE HYPOTHESES: Propose diagnoses others haven't considered
3. COGNITIVE BIAS DETECTION: Identify anchoring, availability, confirmation biases
4. 'WHAT IF' ANALYSIS: Explore "what if we're wrong?" scenarios

CRITICAL BEHAVIORS:
- Ask: "What else could explain these findings?"
- Challenge: "Are we anchoring on an obvious but possibly wrong diagnosis?"
- Probe: "What would we expect to see if this were actually X instead?"
- Warn: "We might be missing Y because of Z bias"

You are skeptical, thorough, and never afraid to challenge the consensus."""
    }
}

# Role assignment to models (maps model names to council roles)
MODEL_ROLE_ASSIGNMENTS = {
    "Gemini 2.0 Flash": "PERCEPTION_SPECIALIST",      # Good at visual/pattern tasks
    "GPT-4o": "MEDICAL_KNOWLEDGE_EXECUTOR",            # Strong medical knowledge
    "Mistral Large": "PROBABILISTIC_REASONER",         # Good at structured reasoning
    "Llama 3.1 70B": "RISK_ACTION_ANALYST",           # Practical, action-oriented
    # 5th member rotates or uses primary model
}

def get_role_for_model(model_name: str, role_override: str = None) -> dict:
    """Get the council role assigned to a model"""
    if role_override and role_override in COUNCIL_ROLES:
        return COUNCIL_ROLES[role_override]
    
    assigned_role = MODEL_ROLE_ASSIGNMENTS.get(model_name, "MEDICAL_KNOWLEDGE_EXECUTOR")
    return COUNCIL_ROLES.get(assigned_role, COUNCIL_ROLES["MEDICAL_KNOWLEDGE_EXECUTOR"])


# ============================================================
# BASE SYSTEM PROMPTS (used with role-specific prompts)
# ============================================================

CLINICAL_ANALYSIS_SYSTEM_PROMPT = """You are a specialist on a multi-model clinical AI council.
Your role is to provide independent, evidence-based analysis of clinical cases from YOUR specialized perspective.

CRITICAL GUIDELINES:
1. Prioritize patient safety above all else
2. Stay in your role - focus on YOUR specialization
3. Be thorough but focused on your area of expertise
4. Consider the patient's individual characteristics (age, allergies, conditions, medications)

Your analysis will be compared with other council members to reach a consensus.
Bring YOUR unique perspective to the case."""


DEBATE_SYSTEM_PROMPT = """You are participating in a multi-specialist clinical council deliberation.
You have seen other specialists' opinions on this case.

REMEMBER YOUR ROLE: You are a specialist. Focus on YOUR area of expertise.

Your task is to:
1. Identify points of agreement with other specialists
2. Challenge points that conflict with YOUR expertise area
3. Add perspectives from YOUR specialization that weren't covered
4. Consider if you should update your position based on other experts' input

This deliberation is about reaching the BEST clinical decision through diverse expert perspectives."""


# ============================================================
# AI COUNCIL CLASS
# ============================================================

class AICouncil:
    """
    Multi-Model AI Council for Clinical Decision Making
    
    Uses 4 different AI models (Gemini, GPT-4, Mistral, Llama) to:
    1. Independently analyze clinical cases
    2. Debate each other's perspectives
    3. Reach a weighted consensus
    
    This eliminates single-model bias and improves accuracy.
    """
    
    def __init__(self):
        self.models = model_registry.get_available_models()
        self.model_weights = {
            # Weights based on model capabilities (can be adjusted)
            "GPT-4o": 1.2,           # Strong clinical knowledge
            "Gemini 2.0 Flash": 1.0, # Good general reasoning
            "Mistral Large": 1.0,    # European perspective
            "Llama 3.1 70B": 0.9,    # Fast, good support
        }
        print(f"[COUNCIL] AI Council initialized with {len(self.models)} models")
    
    def get_active_models(self) -> List[str]:
        """Get list of active model names"""
        return [m.name for m in self.models]
    
    def deliberate(
        self, 
        case_type: str,
        case_data: Dict[str, Any],
        patient: Dict[str, Any] = None
    ) -> CouncilVerdict:
        """
        Run full council deliberation on a clinical case
        
        Args:
            case_type: Type of case (SIMULATION, OPTIMIZATION, DIAGNOSIS, REPORT)
            case_data: Case-specific data (drug info, symptoms, etc.)
            patient: Patient data dictionary
            
        Returns:
            CouncilVerdict with full deliberation results
        """
        start_time = time.time()
        
        # Build case summary for all models
        case_summary = self._build_case_summary(case_type, case_data, patient)
        
        # Phase 1: Parallel Independent Assessment (all models simultaneously)
        print("Phase 1: Independent Assessment...")
        initial_opinions = self._parallel_assessment(case_type, case_summary, patient)
        
        # Phase 2: Debate Rounds (2 rounds of inter-model discussion)
        print("Phase 2: Debate Rounds...")
        debate_results, final_opinions = self._conduct_debate(
            initial_opinions, case_summary, rounds=2
        )
        
        # Phase 3: Calculate Consensus
        print("Phase 3: Calculating Consensus...")
        verdict = self._calculate_consensus(
            case_type, case_summary, final_opinions, debate_results
        )
        
        verdict.deliberation_time_seconds = time.time() - start_time
        
        print(f"Council deliberation complete in {verdict.deliberation_time_seconds:.2f}s")
        print(f"   Verdict: {verdict.final_verdict} | Consensus: {verdict.consensus_score:.0%}")
        
        return verdict
    
    def _build_case_summary(
        self, 
        case_type: str, 
        case_data: Dict, 
        patient: Dict
    ) -> str:
        """Build comprehensive case summary for all models"""
        
        summary_parts = [f"CASE TYPE: {case_type}\n"]
        
        # Patient information
        if patient:
            summary_parts.append("PATIENT PROFILE:")
            summary_parts.append(f"  - Name: {patient.get('name', 'Unknown')}")
            summary_parts.append(f"  - Age: {patient.get('age', 'Unknown')} years")
            summary_parts.append(f"  - Sex: {patient.get('sex', 'Unknown')}")
            summary_parts.append(f"  - Primary Condition: {patient.get('condition', 'Unknown')}")
            
            # Allergies
            allergies = patient.get('allergies', [])
            if allergies:
                summary_parts.append(f"  - Allergies: {', '.join(allergies)}")
            else:
                summary_parts.append("  - Allergies: None reported")
            
            # Current medications
            current_meds = patient.get('current_medications', [])
            if current_meds:
                med_names = [m.get('name', str(m)) if isinstance(m, dict) else str(m) for m in current_meds]
                summary_parts.append(f"  - Current Medications: {', '.join(med_names)}")
            
            # Medical history
            history = patient.get('medical_history', [])
            if history:
                conditions = [h.get('condition', str(h)) if isinstance(h, dict) else str(h) for h in history]
                summary_parts.append(f"  - Medical History: {', '.join(conditions)}")
            
            # Organ function
            vitals = patient.get('vitals', {})
            renal = vitals.get('renal_function', patient.get('renal_function', 'Normal'))
            hepatic = vitals.get('hepatic_function', patient.get('hepatic_function', 'Normal'))
            summary_parts.append(f"  - Renal Function: {renal}")
            summary_parts.append(f"  - Hepatic Function: {hepatic}")
        
        summary_parts.append("")
        
        # Case-specific data
        summary_parts.append("CASE DETAILS:")
        if case_type == "SIMULATION":
            summary_parts.append(f"  - Proposed Drug: {case_data.get('drug', 'Unknown')}")
            summary_parts.append(f"  - Dosage: {case_data.get('dosage', 'Unknown')}")
            summary_parts.append(f"  - Duration: {case_data.get('duration', 'Unknown')}")
            if case_data.get('safety_report'):
                sr = case_data['safety_report']
                summary_parts.append(f"  - Safety Score: {sr.get('safety_score', 'N/A')}/100")
                summary_parts.append(f"  - Critical Alerts: {len(sr.get('critical_alerts', []))}")
                summary_parts.append(f"  - Major Warnings: {len(sr.get('major_warnings', []))}")
                
        elif case_type == "OPTIMIZATION":
            summary_parts.append(f"  - Condition to Treat: {case_data.get('condition', 'Unknown')}")
            summary_parts.append(f"  - Optimization Priority: {case_data.get('priority', 'balanced')}")
            if case_data.get('proposed_drugs'):
                summary_parts.append(f"  - Proposed Drug Regimen: {case_data['proposed_drugs']}")
                
        elif case_type == "DIAGNOSIS":
            symptoms = case_data.get('symptoms', [])
            summary_parts.append(f"  - Reported Symptoms: {', '.join(symptoms)}")
            if case_data.get('primary_diagnosis'):
                summary_parts.append(f"  - Preliminary Diagnosis: {case_data['primary_diagnosis']}")
            if case_data.get('differentials'):
                diffs = [d.get('name', str(d)) if isinstance(d, dict) else str(d) for d in case_data['differentials']]
                summary_parts.append(f"  - Differential Diagnoses: {', '.join(diffs)}")
                
        elif case_type == "REPORT_ANALYSIS":
            summary_parts.append(f"  - Report Type: {case_data.get('report_type', 'Unknown')}")
            summary_parts.append(f"  - Key Findings: {case_data.get('findings', 'None')}")
            if case_data.get('abnormal_values'):
                summary_parts.append(f"  - Abnormal Values: {', '.join(case_data['abnormal_values'])}")
        
        return "\n".join(summary_parts)
    
    def _parallel_assessment(
        self, 
        case_type: str, 
        case_summary: str, 
        patient: Dict
    ) -> List[ModelOpinion]:
        """
        Phase 1: All models analyze the case independently in parallel
        """
        opinions = []
        
        prompt = self._build_assessment_prompt(case_type, case_summary)
        
        # Run all models in parallel for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_model = {
                executor.submit(self._get_model_opinion, model, prompt): model 
                for model in self.models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    opinion = future.result()
                    if opinion:
                        opinions.append(opinion)
                        print(f"   [OK] {model.name}: {opinion.verdict} ({opinion.confidence:.0%})")
                except Exception as e:
                    print(f"   [FAIL] {model.name} failed: {e}")
        
        return opinions
    
    def _build_assessment_prompt(self, case_type: str, case_summary: str) -> str:
        """Build the assessment prompt for all models"""
        
        # DIAGNOSIS requires a specialized prompt for ranked differentials
        if case_type == "DIAGNOSIS":
            return self._build_diagnosis_prompt(case_summary)
        
        verdict_guidance = ""
        if case_type == "SIMULATION":
            verdict_guidance = """
VERDICT OPTIONS:
- APPROVE: Drug is safe and appropriate for this patient
- CAUTION: Drug can be used but with specific modifications or monitoring
- REJECT: Drug should NOT be used for this patient"""
        elif case_type == "OPTIMIZATION":
            verdict_guidance = """
VERDICT OPTIONS:
- APPROVE: Drug regimen is optimal for this patient
- CAUTION: Regimen acceptable but consider alternatives or modifications
- REJECT: Regimen is not appropriate, recommend different approach"""
        else:
            verdict_guidance = """
VERDICT OPTIONS:
- APPROVE: Findings are clinically significant and require action
- CAUTION: Findings need monitoring but not immediate action
- REJECT: Findings are not clinically significant"""
        
        prompt = f"""{case_summary}

{verdict_guidance}

Provide your clinical assessment in the following JSON format:
{{
    "verdict": "APPROVE" or "CAUTION" or "REJECT",
    "confidence": 0.0 to 1.0 (your confidence in this verdict),
    "reasoning": "Detailed explanation of your clinical reasoning",
    "key_concerns": ["concern 1", "concern 2", ...],
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}}

Be thorough. Patient safety is the highest priority.
Return ONLY valid JSON, no other text."""

        return prompt
    
    def _build_diagnosis_prompt(self, case_summary: str) -> str:
        """Build specialized prompt for differential diagnosis with ranked probabilities."""
        
        prompt = f"""{case_summary}

YOUR TASK: Provide a differential diagnosis with EXPLICIT PROBABILITY RANKINGS.

CRITICAL: You must assign DIFFERENT probability percentages to each diagnosis based on how well the symptoms match. Do NOT assign equal probabilities - differentiate based on:
1. How pathognomonic (characteristic) the symptoms are for each condition
2. The base rate/prevalence of each condition
3. How many of the key symptoms for each condition are present vs missing

Provide your clinical assessment in the following JSON format:
{{
    "verdict": "CAUTION",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief clinical reasoning",
    "ranked_diagnoses": [
        {{
            "name": "Most Likely Diagnosis",
            "probability": 45,
            "rationale": "Why this is most likely - which symptoms strongly support it"
        }},
        {{
            "name": "Second Most Likely",
            "probability": 25,
            "rationale": "Why this is considered - what supports and what's missing"
        }},
        {{
            "name": "Third Possibility",
            "probability": 15,
            "rationale": "Why still on differential despite lower probability"
        }},
        {{
            "name": "Fourth Possibility",
            "probability": 10,
            "rationale": "Must rule out because critical/dangerous"
        }},
        {{
            "name": "Fifth Possibility",
            "probability": 5,
            "rationale": "Less likely but worth considering"
        }}
    ],
    "key_concerns": ["concern 1", "concern 2"],
    "recommendations": ["recommendation 1", "recommendation 2"]
}}

RULES:
1. Probabilities should sum to approximately 100%
2. The primary diagnosis should have the HIGHEST probability (not equal to others)
3. Mark any life-threatening conditions that must be ruled out urgently
4. Be specific about which symptoms support or argue against each diagnosis

Return ONLY valid JSON, no other text."""

        return prompt
    
    def _get_model_opinion(self, model, prompt: str) -> Optional[ModelOpinion]:
        """Get a single council member's opinion using their specialized role"""
        try:
            # Get the role for this model
            role = get_role_for_model(model.name)
            role_key = MODEL_ROLE_ASSIGNMENTS.get(model.name, "MEDICAL_KNOWLEDGE_EXECUTOR")
            
            # Build role-specific system prompt
            role_system_prompt = f"""{role['system_prompt']}

{CLINICAL_ANALYSIS_SYSTEM_PROMPT}"""
            
            response = model.generate(prompt, role_system_prompt)
            
            if not response.get('success'):
                return None
            
            # Parse JSON response
            content = response['content'].strip()
            
            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            # Clean up content
            content = content.strip()
            
            # Basic sanitization for common control characters that break JSON
            # Replace tabs with spaces, remove other control chars
            content = content.replace('\t', ' ')
            content = ''.join(c for c in content if c >= ' ')
            
            data = json.loads(content, strict=False)
            
            return ModelOpinion(
                model_name=model.name,
                provider=model.provider,
                verdict=data.get('verdict', 'CAUTION'),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', ''),
                key_concerns=data.get('key_concerns', []),
                recommendations=data.get('recommendations', []),
                ranked_diagnoses=data.get('ranked_diagnoses', []),
                council_role=role_key,
                role_title=role['title'],
                role_icon=role['icon']
            )
            
        except json.JSONDecodeError as e:
            print(f"   JSON parse error for {model.name}: {e}")
            return None
        except Exception as e:
            print(f"   Error getting opinion from {model.name}: {e}")
            return None
    
    def _conduct_debate(
        self, 
        initial_opinions: List[ModelOpinion], 
        case_summary: str,
        rounds: int = 2
    ) -> tuple:
        """
        Phase 2: Models debate each other's opinions
        
        Each model sees all other models' opinions and can:
        - Agree with points
        - Disagree/challenge points
        - Add new perspectives
        - Change their own verdict based on new evidence
        """
        debate_results = []
        current_opinions = initial_opinions.copy()
        
        for round_num in range(1, rounds + 1):
            print(f"   Round {round_num}...")
            round_result = {"round": round_num, "exchanges": []}
            
            # Each model responds to others
            new_opinions = []
            
            for model in self.models:
                # Find this model's current opinion
                my_opinion = next(
                    (op for op in current_opinions if op.model_name == model.name), 
                    None
                )
                
                # Get opinions from other models
                other_opinions = [
                    op for op in current_opinions if op.model_name != model.name
                ]
                
                if not other_opinions:
                    continue
                
                # Build debate prompt
                debate_prompt = self._build_debate_prompt(
                    case_summary, my_opinion, other_opinions, round_num
                )
                
                try:
                    response = model.generate(debate_prompt, DEBATE_SYSTEM_PROMPT)
                    
                    if response.get('success'):
                        debate_response = self._parse_debate_response(
                            response['content'], model.name, model.provider
                        )
                        
                        if debate_response:
                            new_opinions.append(debate_response['opinion'])
                            round_result['exchanges'].extend(
                                debate_response.get('debate_points', [])
                            )
                except Exception as e:
                    print(f"   Debate error for {model.name}: {e}")
            
            # Update opinions for next round
            if new_opinions:
                current_opinions = new_opinions
            
            debate_results.append(round_result)
        
        return debate_results, current_opinions
    
    def _build_debate_prompt(
        self, 
        case_summary: str, 
        my_opinion: ModelOpinion, 
        other_opinions: List[ModelOpinion],
        round_num: int
    ) -> str:
        """Build prompt for debate round with role context"""
        
        others_text = []
        for op in other_opinions:
            role_label = f"{op.role_icon} {op.role_title}" if op.role_title else op.model_name
            others_text.append(f"""
{role_label} ({op.model_name}):
  ROLE: {op.role_title if op.role_title else 'General Analyst'}
  Verdict: {op.verdict} (Confidence: {op.confidence:.0%})
  Reasoning: {op.reasoning}
  Concerns: {', '.join(op.key_concerns[:3])}
  Recommendations: {', '.join(op.recommendations[:3])}""")
        
        my_role = ""
        my_verdict_text = ""
        if my_opinion:
            my_role = f"YOUR ROLE: {my_opinion.role_icon} {my_opinion.role_title}" if my_opinion.role_title else ""
            my_verdict_text = f"""
YOUR PREVIOUS POSITION:
  {my_role}
  Verdict: {my_opinion.verdict} (Confidence: {my_opinion.confidence:.0%})
  Reasoning: {my_opinion.reasoning}"""
        
        prompt = f"""COUNCIL DELIBERATION - ROUND {round_num}

{case_summary}

{my_verdict_text}

OTHER COUNCIL MEMBERS' OPINIONS:
{chr(10).join(others_text)}

INSTRUCTIONS (Stay in YOUR role as a specialist):
1. Review each specialist's opinion from their perspective
2. Identify points of AGREEMENT (acknowledge valid insights)
3. Challenge points that conflict with YOUR area of expertise
4. Consider if you should UPDATE your position based on other specialists' input
5. Add insights from YOUR specialization that weren't covered

Respond in JSON format:
{{
    "verdict": "APPROVE" or "CAUTION" or "REJECT" (may change from your previous),
    "confidence": 0.0 to 1.0,
    "reasoning": "Updated reasoning from YOUR specialist perspective",
    "key_concerns": ["concerns from YOUR area of expertise"],
    "recommendations": ["recommendations from YOUR specialist viewpoint"],
    "debate_points": [
        {{
            "target_model": "Specialist Role/Name",
            "point_type": "AGREE" or "DISAGREE" or "CLARIFY" or "CHALLENGE",
            "content": "Your specific point from YOUR specialty",
            "evidence": "Supporting evidence if disagreeing"
        }}
    ]
}}

Return ONLY valid JSON."""

        return prompt
    
    def _parse_debate_response(
        self, 
        content: str, 
        model_name: str, 
        provider: str
    ) -> Optional[Dict]:
        """Parse a council member's debate response"""
        try:
            content = content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            data = json.loads(content)
            
            # Get role for this model
            role = get_role_for_model(model_name)
            role_key = MODEL_ROLE_ASSIGNMENTS.get(model_name, "MEDICAL_KNOWLEDGE_EXECUTOR")
            
            opinion = ModelOpinion(
                model_name=model_name,
                provider=provider,
                verdict=data.get('verdict', 'CAUTION'),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', ''),
                key_concerns=data.get('key_concerns', []),
                recommendations=data.get('recommendations', []),
                council_role=role_key,
                role_title=role['title'],
                role_icon=role['icon']
            )
            
            debate_points = []
            for dp in data.get('debate_points', []):
                debate_points.append(DebatePoint(
                    source_model=f"{role['icon']} {role['title']} ({model_name})",
                    target_model=dp.get('target_model', ''),
                    point_type=dp.get('point_type', 'CLARIFY'),
                    content=dp.get('content', ''),
                    evidence=dp.get('evidence', '')
                ))
            
            return {
                "opinion": opinion,
                "debate_points": debate_points
            }
            
        except Exception as e:
            print(f"   Parse error for {model_name}: {e}")
            return None
    
    def _calculate_consensus(
        self, 
        case_type: str,
        case_summary: str,
        final_opinions: List[ModelOpinion],
        debate_results: List[Dict]
    ) -> CouncilVerdict:
        """
        Phase 3: Calculate weighted consensus from all opinions
        """
        if not final_opinions:
            return CouncilVerdict(
                case_type=case_type,
                case_summary=case_summary,
                final_verdict="CAUTION",
                consensus_score=0.0
            )
        
        # Weighted voting
        verdict_scores = {"APPROVE": 0.0, "CAUTION": 0.0, "REJECT": 0.0}
        total_weight = 0.0
        
        for opinion in final_opinions:
            weight = self.model_weights.get(opinion.model_name, 1.0)
            weighted_conf = weight * opinion.confidence
            verdict_scores[opinion.verdict] += weighted_conf
            total_weight += weighted_conf
        
        # Normalize scores
        if total_weight > 0:
            for v in verdict_scores:
                verdict_scores[v] /= total_weight
        
        # Determine winning verdict
        final_verdict = max(verdict_scores, key=verdict_scores.get)
        consensus_score = verdict_scores[final_verdict]
        
        # Find dissenting models
        dissenting_models = []
        dissent_reasons = []
        for opinion in final_opinions:
            if opinion.verdict != final_verdict:
                dissenting_models.append(opinion.model_name)
                dissent_reasons.append(f"{opinion.model_name}: {opinion.reasoning[:100]}...")
        
        # Find unanimous concerns and recommendations
        all_concerns = {}
        all_recommendations = {}
        
        for opinion in final_opinions:
            for concern in opinion.key_concerns:
                concern_lower = concern.lower()
                all_concerns[concern_lower] = all_concerns.get(concern_lower, 0) + 1
            for rec in opinion.recommendations:
                rec_lower = rec.lower()
                all_recommendations[rec_lower] = all_recommendations.get(rec_lower, 0) + 1
        
        model_count = len(final_opinions)
        unanimous_concerns = [c for c, count in all_concerns.items() if count >= model_count * 0.75]
        unanimous_recommendations = [r for r, count in all_recommendations.items() if count >= model_count * 0.75]
        
        # Collect all debate points
        all_debate_points = []
        for round_result in debate_results:
            for exchange in round_result.get('exchanges', []):
                if isinstance(exchange, DebatePoint):
                    all_debate_points.append(exchange)
                elif isinstance(exchange, dict):
                    all_debate_points.append(DebatePoint(**exchange))
        
        return CouncilVerdict(
            case_type=case_type,
            case_summary=case_summary,
            final_verdict=final_verdict,
            consensus_score=consensus_score,
            model_opinions=final_opinions,
            debate_rounds=debate_results,
            key_debate_points=all_debate_points,
            dissenting_models=dissenting_models,
            dissent_reasons=dissent_reasons,
            unanimous_concerns=unanimous_concerns[:5],
            unanimous_recommendations=unanimous_recommendations[:5],
            total_models=len(self.models),
            models_participated=[op.model_name for op in final_opinions]
        )
    
    def quick_consensus(
        self, 
        case_type: str, 
        case_data: Dict, 
        patient: Dict = None
    ) -> CouncilVerdict:
        """
        Quick consensus without debate (for less critical decisions)
        Only Phase 1 (parallel assessment) and Phase 3 (consensus)
        No debate rounds for faster response.
        """
        start_time = time.time()
        
        case_summary = self._build_case_summary(case_type, case_data, patient)
        
        # Phase 1 only
        print("Quick Assessment (no debate)...")
        opinions = self._parallel_assessment(case_type, case_summary, patient)
        
        # Phase 3: Calculate Consensus
        verdict = self._calculate_consensus(case_type, case_summary, opinions, [])
        verdict.deliberation_time_seconds = time.time() - start_time
        
        return verdict


# ============================================================
# SINGLETON INSTANCE
# ============================================================

print("\n[COUNCIL] Initializing AI Clinical Council...")
ai_council = AICouncil()
