"""
Disagreement Learning System
Captures and learns from clinician-AI disagreements
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class DisagreementLearner:
    """
    System for capturing, analyzing, and learning from disagreements
    """
    
    def __init__(self):
        self.disagreements = []
        self.pattern_cache = {}
        self.adjustment_history = []
    
    def capture(
        self,
        case_id: str,
        ai_prediction: Dict,
        clinician_action: str,
        final_diagnosis: str,
        disagreement_reason: str,
        outcome: str
    ):
        """
        Capture a disagreement instance
        """
        disagreement = {
            'case_id': case_id,
            'timestamp': datetime.now().isoformat(),
            'ai_prediction': ai_prediction,
            'clinician_action': clinician_action,
            'final_diagnosis': final_diagnosis,
            'disagreement_reason': disagreement_reason,
            'outcome': outcome,
            'resolved': final_diagnosis != ''
        }
        
        self.disagreements.append(disagreement)
        
        print(f"[DISAGREEMENT] Logged: AI={ai_prediction.get('top_diagnosis')}, Clinician={final_diagnosis}")
        
        return disagreement
    
    def analyze_patterns(self, time_window_days: int = 7) -> List[Dict]:
        """
        Analyze disagreement patterns
        """
        cutoff = datetime.now() - timedelta(days=time_window_days)
        
        recent = [
            d for d in self.disagreements
            if datetime.fromisoformat(d['timestamp']) > cutoff and d['resolved']
        ]
        
        if len(recent) < 5:
            return []
        
        # Cluster by features
        patterns = self._cluster_disagreements(recent)
        
        # Identify systematic errors
        systematic_patterns = [
            p for p in patterns
            if p['frequency'] > 3 and p['clinician_correct_rate'] > 0.7
        ]
        
        return systematic_patterns
    
    def _cluster_disagreements(self, disagreements: List[Dict]) -> List[Dict]:
        """
        Cluster disagreements by common features
        """
        # Group by diagnosis type
        by_diagnosis = defaultdict(list)
        
        for d in disagreements:
            ai_dx = d['ai_prediction'].get('top_diagnosis', 'Unknown')
            clinic_dx = d['final_diagnosis']
            
            if ai_dx != clinic_dx:
                key = f"{ai_dx}_vs_{clinic_dx}"
                by_diagnosis[key].append(d)
        
        patterns = []
        
        for key, cases in by_diagnosis.items():
            if len(cases) >= 3:  # Minimum frequency
                # Check if clinician is usually correct
                clinician_correct = sum(
                    1 for c in cases
                    if c['outcome'] in ['resolved', 'good', 'recovered']
                )
                
                patterns.append({
                    'pattern': key,
                    'description': f"AI predicts {key.split('_vs_')[0]}, actually {key.split('_vs_')[1]}",
                    'frequency': len(cases),
                    'clinician_correct_rate': clinician_correct / len(cases),
                    'cases': cases,
                    'suggested_adjustment': self._infer_adjustment(cases)
                })
        
        return patterns
    
    def _infer_adjustment(self, cases: List[Dict]) -> str:
        """
        Infer what adjustment might fix this pattern
        """
        # Extract common reasons
        reasons = [c.get('disagreement_reason', '') for c in cases]
        
        if any('missed' in r.lower() for r in reasons):
            return "Increase sensitivity for this pattern"
        elif any('risk' in r.lower() or 'factor' in r.lower() for r in reasons):
            return "Adjust risk factor weights"
        elif any('age' in r.lower() for r in reasons):
            return "Apply age-specific priors"
        else:
            return "Review feature importance"
    
    def propose_model_update(self, pattern: Dict) -> Dict:
        """
        Propose a model update based on pattern
        """
        proposal = {
            'pattern_id': f"ADJ-{len(self.adjustment_history) + 1}",
            'timestamp': datetime.now().isoformat(),
            'pattern': pattern['description'],
            'frequency': pattern['frequency'],
            'suggested_adjustment': pattern['suggested_adjustment'],
            'expected_improvement': self._simulate_improvement(pattern),
            'requires_review': True,
            'status': 'PENDING'
        }
        
        return proposal
    
    def _simulate_improvement(self, pattern: Dict) -> str:
        """
        Estimate improvement if adjustment is applied
        """
        freq = pattern['frequency']
        correct_rate = pattern['clinician_correct_rate']
        
        if correct_rate > 0.8 and freq > 10:
            return f"+{int(freq * correct_rate)} correct diagnoses"
        elif correct_rate > 0.7:
            return f"+{int(freq * 0.6)} correct diagnoses (estimated)"
        else:
            return "Uncertain - manual review recommended"
    
    def apply_adjustment(self, proposal_id: str, adjustment_params: Dict):
        """
        Apply an approved adjustment
        """
        self.adjustment_history.append({
            'proposal_id': proposal_id,
            'timestamp': datetime.now().isoformat(),
            'adjustment_params': adjustment_params,
            'status': 'APPLIED'
        })
        
        print(f"[ADJUSTMENT] Applied: {proposal_id}")
    
    def get_analytics(self, time_window_days: int = 30) -> Dict:
        """
        Get disagreement analytics dashboard
        """
        cutoff = datetime.now() - timedelta(days=time_window_days)
        
        recent = [
            d for d in self.disagreements
            if datetime.fromisoformat(d['timestamp']) > cutoff
        ]
        
        # Calculate agreement rate
        total = len(recent)
        if total == 0:
            return {
                'time_window_days': time_window_days,
                'total_cases': 0,
                'agreement_rate': 1.0
            }
        
        # Count outcomes
        ai_correct = sum(
            1 for d in recent
            if d.get('outcome') in ['ai_correct', 'ai_right']
        )
        clinician_correct = sum(
            1 for d in recent
            if d.get('outcome') in ['clinician_correct', 'good', 'resolved']
        )
        both_wrong = sum(
            1 for d in recent
            if d.get('outcome') in ['both_wrong', 'poor_outcome']
        )
        unknown = total - ai_correct - clinician_correct - both_wrong
        
        # Find top patterns
        patterns = self.analyze_patterns(time_window_days)
        top_patterns = sorted(
            patterns,
            key=lambda p: p['frequency'],
            reverse=True
        )[:5]
        
        # Format top patterns
        formatted_patterns = []
        for p in top_patterns:
            formatted_patterns.append({
                'pattern': p['description'],
                'frequency': p['frequency'],
                'action_taken': p['suggested_adjustment'],
                'improvement_after_fix': 'Pending monitoring'
            })
        
        return {
            'period': f"Last {time_window_days} days",
            'total_cases': total,
            'ai_clinician_agreement': f"{((total - len(recent)) / total * 100):.0f}%" if total > 0 else "100%",
            'disagreement_breakdown': {
                'ai_correct_clinician_wrong': ai_correct,
                'clinician_correct_ai_wrong': clinician_correct,
                'both_wrong': both_wrong,
                'outcome_unknown': unknown
            },
            'top_disagreement_patterns': formatted_patterns,
            'adjustments_applied': len(self.adjustment_history)
        }
    
    def should_retrain(self) -> Dict:
        """
        Determine if retraining is recommended
        """
        recent_agreement = self.get_agreement_rate(days=30)
        
        should_retrain = recent_agreement < 0.80
        
        if should_retrain:
            patterns = self.analyze_patterns(30)
            return {
                'retrain_recommended': True,
                'current_agreement_rate': recent_agreement,
                'threshold': 0.80,
                'top_patterns': patterns[:5],
                'estimated_improvement_potential': self._estimate_improvement_potential(patterns)
            }
        
        return {
            'retrain_recommended': False,
            'current_agreement_rate': recent_agreement,
            'status': 'Performance acceptable'
        }
    
    def get_agreement_rate(self, days: int = 30) -> float:
        """
        Calculate clinician-AI agreement rate
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [
            d for d in self.disagreements
            if datetime.fromisoformat(d['timestamp']) > cutoff
        ]
        
        if len(recent) == 0:
            return 1.0
        
        # Agreement is when AI and clinician match
        agreements = sum(
            1 for d in recent
            if d['ai_prediction'].get('top_diagnosis', '').lower() in d['final_diagnosis'].lower()
        )
        
        return agreements / len(recent)
    
    def _estimate_improvement_potential(self, patterns: List[Dict]) -> str:
        """
        Estimate how much improvement is possible
        """
        total_fixable = sum(
            p['frequency'] * p['clinician_correct_rate']
            for p in patterns
        )
        
        if total_fixable > 20:
            return f"High (est. +{int(total_fixable)} correct diagnoses)"
        elif total_fixable > 10:
            return f"Moderate (est. +{int(total_fixable)} correct diagnoses)"
        else:
            return "Low (incremental gains)"


class FeedbackCollector:
    """
    Collects structured feedback from clinicians
    """
    
    DISAGREEMENT_REASONS = [
        "AI missed key clinical information",
        "Imaging findings were misinterpreted",
        "Clinical gestalt / experience",
        "Recent similar case",
        "Risk factors not weighted properly",
        "Age/demographics not considered",
        "Rare disease pattern recognized",
        "Laboratory results contradicted",
        "Other"
    ]
    
    @staticmethod
    def collect_feedback_prompt(
        ai_suggestion: str,
        clinician_action: str
    ) -> Dict:
        """
        Generate feedback collection prompt for clinician
        """
        return {
            'title': 'Why did you disagree with the AI suggestion?',
            'ai_suggested': ai_suggestion,
            'your_action': clinician_action,
            'reasons': FeedbackCollector.DISAGREEMENT_REASONS,
            'additional_comments': '',
            'outcome_follow_up': True
        }
    
    @staticmethod
    def parse_feedback(feedback: Dict) -> Dict:
        """
        Parse and structure feedback
        """
        return {
            'reason': feedback.get('selected_reason', 'Unknown'),
            'comments': feedback.get('additional_comments', ''),
            'timestamp': datetime.now().isoformat(),
            'structured': True
        }


# Singleton
disagreement_learner = DisagreementLearner()
feedback_collector = FeedbackCollector()
