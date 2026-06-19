"""
Model Calibration & Safety Monitoring
Handles probability calibration, drift detection, and OOD monitoring
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta
import json


class CalibrationMetrics:
    """Calculate and track calibration metrics"""
    
    @staticmethod
    def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Calculate Brier Score
        
        Range: 0 (perfect) to 1 (worst)
        Target: < 0.15 for clinical acceptability
        """
        if len(predictions) != len(outcomes):
            return 1.0
        
        return float(np.mean((predictions - outcomes) ** 2))
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        Target: < 0.05 (5% calibration error)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = predictions[in_bin].mean()
                avg_accuracy = outcomes[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
        
        return float(ece)
    
    @staticmethod
    def reliability_data(
        predictions: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """
        Generate data for reliability diagram
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins_data = []
        
        for i in range(n_bins):
            in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            
            if in_bin.sum() > 0:
                bins_data.append({
                    'bin_center': (bin_boundaries[i] + bin_boundaries[i + 1]) / 2,
                    'avg_confidence': float(predictions[in_bin].mean()),
                    'avg_accuracy': float(outcomes[in_bin].mean()),
                    'count': int(in_bin.sum())
                })
        
        return {
            'bins': bins_data,
            'perfect_calibration_line': [(0, 0), (1, 1)]
        }


class DriftDetector:
    """
    Detect distribution drift in inputs
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.reference_stats = None
        self.current_window = deque(maxlen=window_size)
        self.drift_history = []
    
    def set_reference(self, reference_data: Dict):
        """Set reference distribution statistics"""
        self.reference_stats = reference_data
    
    def check_covariate_drift(self, patient_data: Dict) -> Dict:
        """
        Detect shifts in patient demographics
        """
        if self.reference_stats is None:
            return {'drift_detected': False, 'reason': 'No reference data'}
        
        # Add to window
        self.current_window.append(patient_data)
        
        if len(self.current_window) < 100:  # Need minimum samples
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Calculate current statistics
        current_stats = self._calculate_patient_stats(list(self.current_window))
        
        # Compare distributions
        kl_div = self._kl_divergence(current_stats, self.reference_stats)
        
        drift_detected = kl_div > 0.3  # Threshold for drift
        
        result = {
            'drift_detected': drift_detected,
            'drift_score': float(kl_div),
            'drift_magnitude': 'SEVERE' if kl_div > 0.6 else ('MODERATE' if kl_div > 0.3 else 'MINOR'),
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            self.drift_history.append(result)
        
        return result
    
    def check_concept_drift(
        self,
        recent_predictions: List[float],
        recent_outcomes: List[float],
        historical_accuracy: float
    ) -> Dict:
        """
        Detect when relationship between findings and outcomes changes
        """
        if len(recent_predictions) < 100:
            return {'drift_detected': False, 'reason': 'Insufficient recent data'}
        
        recent_accuracy = np.mean(
            [1 if p > 0.5 and o == 1 or p <= 0.5 and o == 0 else 0
             for p, o in zip(recent_predictions, recent_outcomes)]
        )
        
        accuracy_drop = historical_accuracy - recent_accuracy
        
        drift_detected = accuracy_drop > 0.10  # 10% drop threshold
        
        return {
            'drift_detected': drift_detected,
            'type': 'CONCEPT_DRIFT',
            'severity': 'HIGH' if accuracy_drop > 0.15 else 'MODERATE',
            'accuracy_drop': float(accuracy_drop),
            'recent_accuracy': float(recent_accuracy),
            'historical_accuracy': float(historical_accuracy),
            'recommendation': 'URGENT: Model retraining required' if drift_detected else 'Continue monitoring'
        }
    
    def _calculate_patient_stats(self, patients: List[Dict]) -> Dict:
        """Calculate population statistics"""
        ages = [p.get('age', 50) for p in patients]
        sexes = [p.get('sex', 'U') for p in patients]
        
        return {
            'age_mean': float(np.mean(ages)),
            'age_std': float(np.std(ages)),
            'sex_ratio_male': float(sum(1 for s in sexes if s == 'M') / len(sexes)),
            'sample_size': len(patients)
        }
    
    def _kl_divergence(self, current: Dict, reference: Dict) -> float:
        """Calculate KL divergence (simplified)"""
        # Simplified KL divergence based on age and sex distributions
        age_diff = abs(current.get('age_mean', 50) - reference.get('age_mean', 50)) / 50
        sex_diff = abs(current.get('sex_ratio_male', 0.5) - reference.get('sex_ratio_male', 0.5))
        
        return (age_diff + sex_diff) / 2


class OODDetector:
    """
    Out-of-Distribution detector
    """
    
    def __init__(self):
        self.detection_threshold = 0.7
    
    def detect(self, image_analysis: Dict, patient_context: Dict) -> Dict:
        """
        Detect if input is out-of-distribution
        """
        ood_scores = {}
        
        # Check image quality
        quality_score = self._assess_quality(image_analysis)
        ood_scores['quality'] = quality_score
        
        # Check patient demographics
        demographic_score = self._assess_demographics(patient_context)
        ood_scores['demographics'] = demographic_score
        
        # Check finding characteristics
        findings_score = self._assess_findings(image_analysis)
        ood_scores['findings'] = findings_score
        
        # Ensemble score
        ood_probability = np.mean(list(ood_scores.values()))
        
        is_ood = ood_probability > self.detection_threshold
        
        result = {
            'is_ood': bool(is_ood),
            'ood_probability': float(ood_probability),
            'component_scores': ood_scores,
            'confidence_cap': 0.30 if is_ood else 1.0,
            'warning': "⚠️ This case appears unusual. Predictions may be unreliable." if is_ood else None,
            'action': 'MANDATORY_HUMAN_REVIEW' if is_ood else 'CONTINUE'
        }
        
        return result
    
    def _assess_quality(self, analysis: Dict) -> float:
        """Assess if image quality is unusual"""
        quality = analysis.get('quality_assessment', 'Good')
        
        quality_scores = {
            'Excellent': 0.0,
            'Good': 0.1,
            'Adequate': 0.3,
            'Limited': 0.6,
            'Non-diagnostic': 0.9
        }
        
        return quality_scores.get(quality, 0.5)
    
    def _assess_demographics(self, patient: Dict) -> float:
        """Assess if patient demographics are within training distribution"""
        if not patient:
            return 0.5
        
        age = patient.get('age', 50)
        
        # Flag if age is extreme
        if age < 1 or age > 100:
            return 0.8
        elif age < 10 or age > 90:
            return 0.4
        
        return 0.1
    
    def _assess_findings(self, analysis: Dict) -> float:
        """Assess if findings are unusual"""
        # Check for very high number of findings (atypical)
        findings_count = len(analysis.get('detailed_findings', []))
        
        if findings_count > 10:
            return 0.6
        elif findings_count == 0:
            return 0.7
        
        return 0.1


class CriticalDiagnosisTracker:
    """
    Track missed critical diagnoses
    """
    
    CRITICAL_DIAGNOSES = [
        "Pulmonary embolism",
        "Aortic dissection",
        "Acute MI",
        "Myocardial infarction",
        "Tension pneumothorax",
        "Pneumothorax",
        "Bowel perforation",
        "Intracranial hemorrhage",
        "Stroke",
        "Sepsis",
        "Meningitis"
    ]
    
    def __init__(self):
        self.missed_critical_log = []
    
    def track(
        self,
        system_output: Dict,
        final_diagnosis: str,
        patient_outcome: str
    ):
        """
        Track critical diagnosis performance
        """
        if self._is_critical(final_diagnosis):
            system_ranking = self._get_ranking(system_output, final_diagnosis)
            system_probability = self._get_probability(system_output, final_diagnosis)
            
            # Flag if critical diagnosis was missed or deprioritized
            if system_ranking is None or system_ranking > 3:
                self.missed_critical_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'diagnosis': final_diagnosis,
                    'system_ranking': system_ranking,
                    'probability_assigned': system_probability,
                    'patient_outcome': patient_outcome,
                    'severity': 'CRITICAL' if patient_outcome in ['death', 'severe_harm'] else 'MODERATE',
                    'root_cause_analysis_required': True
                })
                
                print(f"[CRITICAL MISS] {final_diagnosis} ranked #{system_ranking} (prob: {system_probability:.1%})")
    
    def _is_critical(self, diagnosis: str) -> bool:
        """Check if diagnosis is critical"""
        diagnosis_lower = diagnosis.lower()
        return any(crit.lower() in diagnosis_lower for crit in self.CRITICAL_DIAGNOSES)
    
    def _get_ranking(self, system_output: Dict, diagnosis: str) -> Optional[int]:
        """Get ranking of diagnosis in system output"""
        differentials = system_output.get('differential_diagnoses', [])
        
        for rank, dx in enumerate(differentials, 1):
            if diagnosis.lower() in dx.get('condition', '').lower():
                return rank
        
        return None
    
    def _get_probability(self, system_output: Dict, diagnosis: str) -> float:
        """Get probability assigned to diagnosis"""
        differentials = system_output.get('differential_diagnoses', [])
        
        for dx in differentials:
            if diagnosis.lower() in dx.get('condition', '').lower():
                return dx.get('probability', 0.0)
        
        return 0.0
    
    def get_miss_rate(self, time_window_days: int = 30) -> Dict:
        """Calculate missed critical diagnosis rate"""
        cutoff = datetime.now() - timedelta(days=time_window_days)
        
        recent_misses = [
            m for m in self.missed_critical_log
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
        return {
            'time_window_days': time_window_days,
            'missed_critical_count': len(recent_misses),
            'miss_rate': len(recent_misses),  # Would normalize by total critical cases
            'by_diagnosis': self._group_by_diagnosis(recent_misses)
        }
    
    def _group_by_diagnosis(self, misses: List[Dict]) -> Dict:
        """Group misses by diagnosis type"""
        grouped = {}
        for miss in misses:
            dx = miss['diagnosis']
            if dx not in grouped:
                grouped[dx] = 0
            grouped[dx] += 1
        return grouped


# Singleton instances
calibration_metrics = CalibrationMetrics()
drift_detector = DriftDetector()
ood_detector = OODDetector()
critical_tracker = CriticalDiagnosisTracker()
