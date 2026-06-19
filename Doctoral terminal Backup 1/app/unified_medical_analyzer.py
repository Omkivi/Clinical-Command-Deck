"""
Unified Medical Imaging & Report Analysis Engine

Handles ALL medical files in one place:
1. ECG/EKG interpretation
2. Chest X-rays
3. CT scans
4. MRI scans
5. Lab reports
6. Pathology reports
7. Any medical document

AI automatically detects file type and routes to appropriate analyzer.
"""

import os
import base64
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

load_dotenv()

class UnifiedMedicalAnalyzer:
    """
    Single upload point for all medical imaging and reports.
    Automatically detects type and analyzes accordingly.
    """
    
    def __init__(self):
        # Initialize Gemini with vision capabilities
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            # Use 2.0-flash as primary, but fall back gracefully if needed
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.vision_model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
            self.vision_model = None

    def _generate_with_retry(self, model, contents):
        """Helper to generate content with retry logic for 429 errors"""
        import time
        import random
        
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries + 1):
            try:
                return model.generate_content(contents)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt == max_retries:
                        raise Exception("API Quota/Rate Limit Exceeded. Please wait 30s and try again.")
                    
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                    print(f"Rate limit hit, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise e
    
    def _load_image(self, file_path: str) -> Image.Image:
        """Load image safely, handling webp and other formats"""
        try:
            with Image.open(file_path) as img:
                img.load()  # Force load image data
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    rgb_img = img.convert('RGB')
                    return rgb_img.copy()
                return img.copy()
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            raise
    
    def analyze_upload(self, file_path: str, patient: Dict = None, clinical_context: str = "") -> Dict:
        """
        Universal analyzer - detects file type and routes to appropriate specialist.
        
        Args:
            file_path: Path to uploaded file
            patient: Optional patient context
            clinical_context: Clinical indication/reason for study
        
        Returns:
            Comprehensive analysis with detected type and findings
        """
        if not self.model:
            return {"error": "AI model not available"}
        
        try:
            # Detect file type
            file_type = self._detect_file_type(file_path)
            
            # Route to appropriate analyzer
            if file_type == "ECG":
                return self.analyze_ecg(file_path, patient, clinical_context)
            elif file_type in ["XRAY", "CT", "MRI", "ULTRASOUND"]:
                return self.analyze_imaging(file_path, file_type, patient, clinical_context)
            elif file_type == "LAB_REPORT":
                return self.analyze_lab_report(file_path, patient)
            elif file_type == "PATHOLOGY":
                return self.analyze_pathology(file_path, patient)
            elif file_type == "DOCUMENT":
                return self.analyze_medical_document(file_path, patient)
            elif file_type == "IMAGING":
                # Generic imaging - use AI to detect type
                detected_type = self._ai_detect_imaging_type(file_path)
                return self.analyze_imaging(file_path, detected_type, patient, clinical_context)
            else:
                return self._generic_analysis(file_path, patient, clinical_context)
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                "error": str(e),
                "recommendation": "Manual review recommended"
            }
    
    def analyze_ecg(self, file_path: str, patient: Dict = None, indication: str = "") -> Dict:
        """
        ECG/EKG interpretation with AI
        
        Detects:
        - Rate and rhythm
        - Axis
        - Intervals (PR, QRS, QT, QTc)
        - ST-segment changes
        - T-wave abnormalities
        - Chamber enlargement
        - Conduction blocks
        - STEMI/NSTEMI
        - Arrhythmias
        """
        if not self.vision_model:
            return self._fallback_ecg_analysis("AI Vision Model not initialized - Check API Key")
        
        try:
            # Load image
            image = self._load_image(file_path)
            
            # Build patient context
            patient_context = ""
            if patient:
                patient_context = f"""
Patient Context:
- Age: {patient.get('age', 'Unknown')}
- Sex: {patient.get('sex', 'Unknown')}
- Current Medications: {', '.join([m.get('name', '') for m in patient.get('current_medications', [])])}
- Known Cardiac History: {', '.join([c for c in patient.get('conditions', []) if 'heart' in c.lower() or 'cardiac' in c.lower()])}
"""
            
            prompt = f"""
You are an expert cardiologist interpreting an ECG.

{patient_context}

Clinical Indication: {indication if indication else 'Not provided'}

Analyze this ECG systematically:

1. RATE & RHYTHM:
   - Heart rate (bpm)
   - Rhythm (sinus, afib, aflutter, etc)
   - Regularity

2. INTERVALS:
   - PR interval (normal 120-200ms)
   - QRS duration (normal <120ms)
   - QT interval
   - QTc (corrected QT, normal <440ms men, <460ms women)

3. AXIS:
   - Frontal plane axis (normal -30 to +90)
   - Any axis deviation

4. CHAMBER ABNORMALITIES:
   - LVH (left ventricular hypertrophy)
   - RVH (right ventricular hypertrophy)
   - Atrial enlargement

5. CONDUCTION ABNORMALITIES:
   - Bundle branch blocks (RBBB, LBBB)
   - Fascicular blocks
   - AV blocks

6. ISCHEMIA/INFARCTION:
   - ST elevation (STEMI?)
   - ST depression
   - T wave inversions
   - Q waves (old MI?)

7. OTHER FINDINGS:
   - Arrhythmias
   - Pre-excitation (WPW)
   - QTc prolongation
   - Electrolyte abnormalities

Return ONLY valid JSON:
{{
    "interpretation_type": "ECG",
    "heart_rate": <number>,
    "rhythm": "description",
    "intervals": {{
        "pr_interval": "normal|prolonged|short",
        "qrs_duration": "normal|prolonged",
        "qtc": "normal|prolonged|borderline"
    }},
    "axis": "normal|left deviation|right deviation",
    "st_segment": "normal|elevated|depressed",
    "critical_findings": ["STEMI in V2-V4", "etc"] or [],
    "abnormalities": ["LVH", "T wave inversion", "etc"] or [],
    "clinical_impression": "Brief summary",
    "urgency_level": "CRITICAL|URGENT|ROUTINE",
    "recommendations": ["Immediate cath lab activation", "Cardiology consult", "etc"]
}}

If STEMI detected, mark urgency_level as CRITICAL and include cath lab activation.
"""
            
            response = self._generate_with_retry(self.vision_model, [prompt, image])
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            analysis = json.loads(text)
            analysis['file_type'] = 'ECG'
            analysis['analyzed_at'] = '2025-12-16'
            
            return analysis
            
        except Exception as e:
            print(f"ECG analysis error: {e}")
            return self._fallback_ecg_analysis(str(e))
    
    def analyze_imaging(self, file_path: str, modality: str, patient: Dict = None, indication: str = "") -> Dict:
        """
        Analyze radiology images (X-ray, CT, MRI, Ultrasound)
        
        Provides comprehensive, clinically actionable findings
        """
        if not self.vision_model:
            return self._fallback_imaging_analysis(modality, "AI Vision Model not initialized - Check API Key")
        
        try:
            image = self._load_image(file_path)
            
            # Build comprehensive patient context
            patient_context = ""
            if patient:
                conditions = patient.get('conditions', []) or patient.get('medical_conditions', []) or []
                medications = patient.get('current_medications', [])
                med_names = [m.get('name', '') if isinstance(m, dict) else str(m) for m in medications]
                
                patient_context = f"""
PATIENT CONTEXT (Correlate findings with this data):
- Age: {patient.get('age', 'Unknown')} years
- Sex: {patient.get('sex', 'Unknown')}
- Weight: {patient.get('weight', 'Unknown')} kg
- Known Conditions: {', '.join(conditions) if conditions else 'None documented'}
- Current Medications: {', '.join(med_names) if med_names else 'None documented'}
- Allergies: {', '.join(patient.get('allergies', [])) if patient.get('allergies') else 'None documented'}
"""
            
            # Modality-specific analysis guidelines
            modality_guidelines = {
                "XRAY": """
CHEST X-RAY SYSTEMATIC REVIEW:
1. Technical quality (rotation, inspiration, penetration)
2. Cardiac silhouette (CTR, contours)
3. Mediastinum (width, contours, trachea)
4. Lung fields (each zone, consolidation, nodules, masses)
5. Pleural spaces (effusion, pneumothorax)
6. Diaphragms (contour, costophrenic angles)
7. Bones (ribs, spine, clavicles)
8. Soft tissues
9. Lines/tubes/devices
""",
                "CT": """
CT SCAN SYSTEMATIC REVIEW:
1. Primary pathology assessment
2. All organs in field of view
3. Vascular structures (aorta, pulmonary vessels)
4. Lymph nodes
5. Bones/spine
6. Any masses with measurements
7. Enhancement patterns if contrast used
8. Incidental findings
9. Comparison with any prior imaging
""",
                "MRI": """
MRI SYSTEMATIC REVIEW:
1. Sequences available (T1, T2, FLAIR, DWI, etc.)
2. Primary pathology
3. Signal characteristics
4. Enhancement patterns
5. Mass effect/edema
6. Vascular structures
7. Any diffusion restriction
8. Incidental findings
""",
                "ULTRASOUND": """
ULTRASOUND SYSTEMATIC REVIEW:
1. Organ visualization and echogenicity
2. Measurements
3. Doppler findings if applicable
4. Masses or lesions
5. Fluid collections
6. Vascularity
"""
            }
            
            prompt = f"""
You are an expert radiologist providing a COMPREHENSIVE, DETAILED analysis.

{patient_context}

MODALITY: {modality}
CLINICAL INDICATION: {indication if indication else 'Not provided - use your clinical judgment'}

{modality_guidelines.get(modality, 'Perform systematic radiological analysis')}

Provide EXHAUSTIVE analysis with specific anatomical locations, measurements, and clinical correlations.

RETURN ONLY VALID JSON with this structure:
{{
    "modality": "{modality}",
    "study_type": "{modality}",
    "quality_assessment": "Excellent/Good/Adequate/Limited/Non-diagnostic",
    "anatomical_coverage": "Body regions included",
    "technique": "Imaging parameters if visible",
    
    "detailed_findings": [
        {{
            "anatomical_region": "Specific area (e.g., 'Right Upper Lobe')",
            "finding": "Detailed description",
            "location": "Precise location (e.g., 'Right upper lobe, apical segment')",
            "size_measurement": "X.X x Y.Y cm" or "N/A",
            "characteristics": "Density/signal/enhancement description",
            "severity": "Normal/Mild/Moderate/Severe",
            "clinical_significance": "High/Medium/Low/Incidental",
            "change_from_prior": "New/Stable/Improved/Worsened/No prior available"
        }}
    ],
    
    "normal_structures": ["List structures that appear normal"],
    
    "impression": "1. Primary finding\\n2. Secondary finding\\n3. etc",
    
    "critical_findings": [
        {{
            "finding": "Critical finding description",
            "urgency": "Immediate/Urgent/Soon",
            "recommended_action": "Specific action"
        }}
    ],
    
    "differential_diagnosis": [
        {{
            "diagnosis": "Most likely diagnosis",
            "likelihood": "High/Moderate/Low",
            "supporting_features": "Features favoring this",
            "against_features": "Features against this"
        }}
    ],
    
    "patient_specific_correlation": "How findings relate to patient's known conditions and medications",
    
    "recommendations": [
        {{
            "recommendation": "Specific recommendation",
            "priority": "Immediate/Routine/Optional",
            "rationale": "Why this is recommended"
        }}
    ],
    
    "follow_up_imaging": "Recommended follow-up if any",
    "limitations": "Study limitations if any",
    
    "clinical_summary": "2-3 sentence summary for referring clinician",
    
    "urgency_level": "CRITICAL/URGENT/ROUTINE"
}}

Be specific with locations (use proper anatomical terminology) and measurements.
Include at least 3-5 detailed findings even for normal studies (document normal structures).
"""
            
            response = self._generate_with_retry(self.vision_model, [prompt, image])
            import json
            from datetime import datetime
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            analysis = json.loads(text)
            analysis['file_type'] = modality
            analysis['analyzed_at'] = datetime.now().isoformat()
            analysis['patient_context_applied'] = patient is not None
            
            return analysis
            
        except Exception as e:
            print(f"Imaging analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_imaging_analysis(modality, str(e))
    
    def analyze_lab_report(self, file_path: str, patient: Dict = None) -> Dict:
        """
        Analyze lab report (text or image)
        """
        # Use existing lab analysis engine if available
        try:
            from .lab_analysis_engine import lab_engine
            
            # Read report text (simplified - actual implementation would use OCR for images)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()
            
            return lab_engine.analyze_lab_report_with_council(report_text, patient.get('id') if patient else None, patient)
            
        except Exception as e:
            print(f"Lab report analysis error: {e}")
            return {"error": "Lab analysis unavailable", "recommendation": "Manual review"}
    
    def analyze_pathology(self, file_path: str, patient: Dict = None) -> Dict:
        """
        Analyze pathology reports
        """
        if not self.model:
            return {"error": "Pathology analysis unavailable"}
        
        try:
            # Read report
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()
            
            prompt = f"""
Analyze this pathology report:

{report_text}

Extract and summarize:
1. Specimen type and site
2. Diagnosis
3. Grade/stage if applicable
4. Margins (if surgical specimen)
5. Molecular markers
6. Clinical significance
7. Recommended follow-up

Return JSON:
{{
    "specimen": "type and site",
    "diagnosis": "primary diagnosis",
    "grade": "if applicable",
    "stage": "if applicable",
    "margins": "positive|negative|close",
    "markers": {{}},
    "significance": "benign|premalignant|malignant|other",
    "recommendations": []
}}
"""
            
            response = self._generate_with_retry(self.model, prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            analysis = json.loads(text)
            analysis['file_type'] = 'PATHOLOGY'
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_medical_document(self, file_path: str, patient: Dict = None) -> Dict:
        """
        Generic medical document analysis
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if self.model:
                prompt = f"""
Analyze this medical document and extract key information:

{content[:5000]}  # Limit to avoid token overflow

Extract:
1. Document type
2. Key findings
3. Medications mentioned
4. Diagnoses
5. Recommendations
6. Critical alerts

Return JSON.
"""
                response = self._generate_with_retry(self.model, prompt)
                # Parse response
                return {"analysis": response.text, "file_type": "DOCUMENT"}
            else:
                return {"error": "Analysis unavailable"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect what type of medical file this is
        """
        filename = file_path.lower()
        
        # Check filename patterns
        if 'ecg' in filename or 'ekg' in filename:
            return "ECG"
        elif 'xray' in filename or 'chest' in filename or 'cxr' in filename:
            return "XRAY"
        elif 'ct' in filename or 'computed' in filename:
            return "CT"
        elif 'mri' in filename or 'magnetic' in filename:
            return "MRI"
        elif 'ultrasound' in filename or 'doppler' in filename or 'echo' in filename:
            return "ULTRASOUND"
        elif 'lab' in filename or 'cbc' in filename or 'cmp' in filename:
            return "LAB_REPORT"
        elif 'path' in filename or 'biopsy' in filename:
            return "PATHOLOGY"
        else:
            # Try to detect from file content/extension
            if filename.endswith(('.jpg', '.jpeg', '.png', '.dicom', '.dcm', '.webp')):
                return "IMAGING"
            else:
                return "DOCUMENT"
    
    def _ai_detect_imaging_type(self, file_path: str) -> str:
        """Use AI vision to detect what type of medical imaging this is"""
        if not self.vision_model:
            return "CT"  # Default fallback
        
        try:
            image = self._load_image(file_path)
            
            prompt = """
Look at this medical image and identify what type it is.
Reply with ONLY ONE of these options (nothing else):
- XRAY (for chest X-rays, bone X-rays, any radiograph)
- CT (for CT scans, computed tomography)
- MRI (for magnetic resonance imaging)
- ULTRASOUND (for ultrasound/sonogram images)
- ECG (for electrocardiogram tracings)

Just reply with the single word.
"""
            
            response = self._generate_with_retry(self.vision_model, [prompt, image])
            detected = response.text.strip().upper()
            
            # Clean up response
            for valid_type in ["ECG", "XRAY", "CT", "MRI", "ULTRASOUND"]:
                if valid_type in detected:
                    return valid_type
            
            return "CT"  # Default for unrecognized
            
        except Exception as e:
            print(f"AI detection error: {e}")
            return "CT"  # Default fallback
    
    def _generic_analysis(self, file_path: str, patient: Dict = None, context: str = "") -> Dict:
        """Generic analysis when type unknown"""
        return {
            "file_type": "UNKNOWN",
            "recommendation": "Manual review recommended - file type could not be automatically determined",
            "note": "Please specify document type or upload with descriptive filename"
        }
    
    def _fallback_ecg_analysis(self, error_msg: str = "AI interpretation unavailable") -> Dict:
        """Fallback when ECG analysis fails"""
        return {
            "file_type": "ECG",
            "interpretation_type": "ECG",
            "error": error_msg,
            "recommendation": "Manual ECG interpretation by cardiologist required",
            "urgency_level": "ROUTINE"
        }
    
    def _fallback_imaging_analysis(self, modality: str, error_msg: str = "AI analysis unavailable") -> Dict:
        """Fallback for imaging"""
        return {
            "file_type": modality,
            "modality": modality,
            "error": error_msg,
            "recommendation": "Manual radiologist interpretation required",
            "urgency_level": "ROUTINE"
        }


# Singleton instance  
medical_analyzer = UnifiedMedicalAnalyzer()
