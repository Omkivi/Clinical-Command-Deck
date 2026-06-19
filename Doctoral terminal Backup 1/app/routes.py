from flask import Blueprint, render_template, request, redirect, url_for, jsonify
from datetime import datetime
from .data_gen import generator
from .optimizer import optimizer_engine
from .diagnosis_engine import diagnosis_engine

main = Blueprint('main', __name__)

@main.route('/')
def index():
    patients = generator.get_all_patients()
    return render_template('index.html', patients=patients)

@main.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    patient = generator.get_patient(patient_id)
    if not patient:
        return redirect(url_for('main.index'))
    return render_template('patient.html', patient=patient)

@main.route('/simulation')
def simulation():
    patient_id = request.args.get('patient_id', type=int)
    patient = generator.get_patient(patient_id) if patient_id else None
    return render_template('simulation.html', patient=patient)

@main.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.json
    patient_id = int(data.get('patient_id'))
    drug = data.get('drug')
    dosage = data.get('dosage')
    duration = data.get('duration', '4 weeks')
    
    patient = generator.get_patient(patient_id)
    
    # Use Council-enabled simulation for multi-model deliberation
    from .ai_engine import ai_engine
    result = ai_engine.simulate_with_council(patient, drug, dosage, duration)
    return result
    
@main.route('/api/optimize', methods=['POST'])
def api_optimize():
    data = request.json
    condition = data.get('condition')
    priority = data.get('priority', 'balanced')
    patient_id = data.get('patient_id')
    integrative_mode = data.get('integrative_mode', False)
    
    patient = None
    if patient_id:
        patient = generator.get_patient(patient_id)

    # Use Council-enabled optimization
    if hasattr(optimizer_engine, 'optimize_with_council'):
        result = optimizer_engine.optimize_with_council(
            condition=condition, 
            priority=priority, 
            patient=patient,
            integrative_mode=integrative_mode
        )
    else:
        result = optimizer_engine.optimize(
            condition=condition, 
            priority=priority, 
            patient=patient,
            integrative_mode=integrative_mode
        )
    
    return jsonify(result)

@main.route('/optimizer')
def optimizer():
    return render_template('optimizer.html')

@main.route('/diagnosis')
def diagnosis():
    all_symptoms = diagnosis_engine.get_all_symptoms()
    patients = generator.get_all_patients()
    return render_template('diagnosis.html', symptoms=all_symptoms, patients=patients)

@main.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    data = request.json
    symptoms = data.get('symptoms', [])
    results = diagnosis_engine.diagnose(symptoms)
    return jsonify(results)

# CRUD API
@main.route('/api/patients', methods=['GET'])
def get_patients():
    patients = generator.get_all_patients()
    return jsonify(patients)

@main.route('/api/patients', methods=['POST'])
def create_patient():
    data = request.json
    patient = generator.add_patient(data)
    return patient

@main.route('/api/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get a single patient by ID"""
    patient = generator.get_patient(patient_id)
    if not patient:
        return {"error": "Patient not found"}, 404
    return jsonify(patient)

@main.route('/api/patients/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    data = request.json
    patient = generator.update_patient(patient_id, data)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

@main.route('/api/patients/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    success = generator.delete_patient(patient_id)
    if not success:
        return {"error": "Patient not found"}, 404
    return {"success": True}

# Medication Management API
@main.route('/api/patients/<int:patient_id>/medications', methods=['POST'])
def add_medication(patient_id):
    data = request.json
    patient = generator.add_medication(patient_id, data)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

@main.route('/api/patients/<int:patient_id>/medications/<med_name>', methods=['DELETE'])
def stop_medication(patient_id, med_name):
    data = request.json
    reason = data.get('reason', '') if data else ''
    patient = generator.stop_medication(patient_id, med_name, reason)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

# Supplement Management API
@main.route('/api/patients/<int:patient_id>/supplements', methods=['POST'])
def add_supplement(patient_id):
    data = request.json
    patient = generator.add_supplement(patient_id, data)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

@main.route('/api/patients/<int:patient_id>/supplements/<supp_name>', methods=['DELETE'])
def remove_supplement(patient_id, supp_name):
    patient = generator.remove_supplement(patient_id, supp_name)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

# Timeline Management API
@main.route('/api/patients/<int:patient_id>/timeline', methods=['POST'])
def add_timeline_event(patient_id):
    data = request.json
    patient = generator.add_timeline_event(patient_id, data)
    if not patient:
        return {"error": "Patient not found"}, 404
    return patient

@main.route('/api/patients/<int:patient_id>/timeline/<int:event_index>', methods=['PUT'])
def update_timeline_event(patient_id, event_index):
    data = request.json
    patient = generator.update_timeline_event(patient_id, event_index, data)
    if not patient:
        return {"error": "Patient or event not found"}, 404
    return patient

@main.route('/api/patients/<int:patient_id>/timeline/<int:event_index>', methods=['DELETE'])
def delete_timeline_event(patient_id, event_index):
    patient = generator.delete_timeline_event(patient_id, event_index)
    if not patient:
        return {"error": "Patient or event not found"}, 404
    return patient

# Optimizer API
@main.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json
    condition = data.get('condition')
    priority = data.get('priority', 'balanced')
    patient_id = data.get('patient_id')  # Optional
    drug_count_override = data.get('drug_count_override')  # Optional manual override
    integrative_mode = data.get('integrative_mode', False)
    
    # Fetch patient data if provided
    patient = None
    if patient_id:
        patient = generator.get_patient(patient_id)
    
    # Use Council-enabled optimization for multi-model deliberation
    results = optimizer_engine.optimize_with_council(condition, priority, patient, drug_count_override, integrative_mode)
    return jsonify(results)

# ============ LAB REPORTS & DIAGNOSTICS ============

@main.route('/lab-reports')
def lab_reports():
    """Render the lab reports interface"""
    return render_template('lab_reports.html')

@main.route('/api/upload-lab-report', methods=['POST'])
def upload_lab_report():
    """Upload and analyze a lab report with AI Council"""
    data = request.json
    report_text = data.get('report_text', '')
    patient_id = data.get('patient_id')
    
    from .lab_analysis_engine import lab_engine
    
    # Get patient for council if provided
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    # Use Council-enabled analysis for multi-model deliberation
    analysis = lab_engine.analyze_lab_report_with_council(report_text, patient_id, patient)
    
    # If patient_id is provided, save to patient record
    if patient_id:
        generator.add_lab_report(int(patient_id), analysis)
    
    return jsonify(analysis)

@main.route('/api/upload-radiology', methods=['POST'])
def upload_radiology():
    """Upload and analyze a radiology report with AI Council"""
    data = request.json
    report_text = data.get('report_text', '')
    study_type = data.get('study_type', 'Unknown')
    patient_id = data.get('patient_id')
    
    from .lab_analysis_engine import lab_engine
    
    # Get patient for council if provided
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    # Use Council-enabled analysis for multi-model deliberation
    analysis = lab_engine.analyze_radiology_report_with_council(report_text, study_type, patient)
    
    # If patient_id is provided, save to patient record
    if patient_id:
        generator.add_radiology_study(int(patient_id), analysis)
    
    return jsonify(analysis)

@main.route('/api/recommend-tests', methods=['POST'])
def recommend_tests():
    """Get test recommendations for given symptoms"""
    data = request.json
    symptoms = data.get('symptoms', [])
    patient_id = data.get('patient_id')
    
    # Get patient data if provided
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    # Get recommendations
    recommendations = diagnosis_engine.recommend_tests(symptoms, None, patient)
    
    return jsonify(recommendations)

@main.route('/api/patients/<int:patient_id>/lab-reports')
def get_patient_lab_reports(patient_id):
    """Get all lab reports for a patient"""
    reports = generator.get_lab_reports(patient_id)
    return jsonify(reports)

@main.route('/api/patients/<int:patient_id>/radiology-studies')
def get_patient_radiology_studies(patient_id):
    """Get all radiology studies for a patient"""
    studies = generator.get_radiology_studies(patient_id)
    return jsonify(studies)

@main.route('/api/diagnose-patient/<int:patient_id>', methods=['POST'])
def diagnose_patient_with_context(patient_id):
    """Run diagnosis with full patient context and AI Council"""
    data = request.json
    symptoms = data.get('symptoms', [])
    
    patient = generator.get_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    # Use Council-enabled diagnosis for multi-model deliberation
    results = diagnosis_engine.diagnose(symptoms, patient=patient)
    
    return jsonify(results)

# ============ FILE UPLOAD ENDPOINTS ============

@main.route('/api/patients/<int:patient_id>/upload-image', methods=['POST'])
def upload_image(patient_id):
    """Upload an image file for a patient"""
    # Check if patient exists
    patient = generator.get_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    file_type = request.form.get('file_type', 'general')
    
    from .file_handler import file_handler
    
    # Save the file
    result = file_handler.save_uploaded_file(file, patient_id, file_type)
    
    if "error" in result:
        return jsonify(result), 400
    
    return jsonify(result)

# ============ UNIFIED MEDICAL ANALYZER ============

@main.route('/api/analyze-medical-file', methods=['POST'])
def analyze_medical_file():
    """
    Universal medical file analyzer - accepts ANY medical file.
    AI auto-detects type and routes to appropriate analyzer.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    clinical_context = request.form.get('clinical_context', '')
    patient_id = request.form.get('patient_id')
    
    from .file_handler import file_handler
    from .unified_medical_analyzer import medical_analyzer
    
    # Get patient data if provided
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    # Save the uploaded file temporarily
    import os
    import tempfile
    
    # Create temp file with proper extension
    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Analyze with unified analyzer
        analysis = medical_analyzer.analyze_upload(
            file_path=temp_path,
            patient=patient,
            clinical_context=clinical_context
        )
        
        # Clean up temp file safely
        try:
            os.unlink(temp_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temp file {temp_path}: {cleanup_error}")
        
        return jsonify(analysis)
        
    except Exception as e:
        # Clean up temp file safely
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temp file {temp_path} during error handling: {cleanup_error}")
        
        print(f"Medical file analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500

@main.route('/api/analyze-imaging-intelligence', methods=['POST'])
def analyze_imaging_intelligence():
    """
    Advanced Imaging Intelligence Engine analysis
    Provides 5-layer analysis with rare disease detection, mechanism reasoning,
    and clinician preference controls
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    clinical_context = request.form.get('clinical_context', '')
    patient_id = request.form.get('patient_id')
    
    # Parse clinician preferences from request
    preferences_json = request.form.get('preferences', '{}')
    try:
        import json
        from .imaging_intelligence_engine import ClinicianPreferences
        prefs_data = json.loads(preferences_json)
        
        preferences = ClinicianPreferences(
            reasoning_mode=prefs_data.get('reasoning_mode', 'standard'),
            assume_worst_case=prefs_data.get('assume_worst_case', True),
            include_rare_diseases=prefs_data.get('include_rare_diseases', True),
            show_uncertainty_ranges=prefs_data.get('show_uncertainty_ranges', True),
            show_mechanism_reasoning=prefs_data.get('show_mechanism_reasoning', False),
            show_council_disagreements=prefs_data.get('show_council_disagreements', True),
            specialty_lens=prefs_data.get('specialty_lens')
        )
    except:
        from .imaging_intelligence_engine import ClinicianPreferences
        preferences = ClinicianPreferences()
    
    # Get patient data
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    # Save file temporarily
    import os
    import tempfile
    
    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name
    
    try:
        # Run through Imaging Intelligence Engine
        from .imaging_intelligence_engine import imaging_intelligence_engine
        
        result = imaging_intelligence_engine.analyze(
            file_path=temp_path,
            patient=patient,
            clinical_context=clinical_context,
            preferences=preferences
        )
        
        # Clean up
        try:
            os.unlink(temp_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temp file: {cleanup_error}")
        
        # Helper to make values JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, '__float__'):  # numpy types
                return float(obj)
            elif hasattr(obj, '__int__'):
                return int(obj)
            elif isinstance(obj, bool):
                return bool(obj)
            return obj
        
        # Convert to JSON-serializable format
        response = {
            'session_id': result.session_id,
            'timestamp': result.timestamp,
            'differential_diagnoses': [
                {
                    'rank': int(dx.rank),
                    'condition': str(dx.condition),
                    'icd10': str(dx.icd10),
                    'probability': float(dx.probability),
                    'confidence_interval': list(dx.confidence_interval),
                    'supporting_evidence': list(dx.supporting_evidence) if dx.supporting_evidence else [],
                    'contradicting_evidence': list(dx.contradicting_evidence) if dx.contradicting_evidence else [],
                    'uncertainty_sources': list(dx.uncertainty_sources) if dx.uncertainty_sources else []
                }
                for dx in result.differential_diagnoses
            ],
            'diagnostic_entropy': float(result.diagnostic_entropy) if result.diagnostic_entropy else 0.0,
            'requires_more_data': bool(result.requires_more_data),
            'recommended_actions': make_serializable(result.recommended_actions),
            'rare_disease_alerts': [
                {
                    'condition': str(alert.condition),
                    'prevalence': str(alert.prevalence),
                    'pattern_match_score': float(alert.pattern_match_score),
                    'why_flagged': str(alert.why_flagged),
                    'specialist_referral': str(alert.specialist_referral),
                    'urgency': str(alert.urgency),
                    'base_rate_overridden': bool(alert.base_rate_overridden)
                }
                for alert in result.rare_disease_alerts
            ],
            'mechanism_analysis': make_serializable(result.mechanism_analysis),
            'council_deliberation': make_serializable(result.council_deliberation),
            'drift_detected': bool(result.drift_detected),
            'out_of_distribution_probability': float(result.out_of_distribution_probability),
            'calibration_score': float(result.calibration_score)
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Clean up
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temp file: {cleanup_error}")
        
        print(f"IIE analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Advanced imaging analysis failed",
            "message": str(e)
        }), 500

@main.route('/api/iie/log-disagreement', methods=['POST'])
def log_disagreement():
    """Log clinician disagreement with AI for learning"""
    data = request.json
    
    from .disagreement_learning import disagreement_learner
    
    disagreement_learner.capture(
        case_id=data.get('session_id'),
        ai_prediction={
            'top_diagnosis': data.get('ai_diagnosis'),
            'probability': data.get('ai_probability')
        },
        clinician_action=data.get('clinician_action'),
        final_diagnosis=data.get('final_diagnosis', ''),
        disagreement_reason=data.get('reason', ''),
        outcome=data.get('outcome', 'pending')
    )
    
    return jsonify({'status': 'logged', 'message': 'Disagreement recorded for analysis'})

@main.route('/api/iie/analytics', methods=['GET'])
def iie_analytics():
    """Get IIE performance analytics"""
    from .disagreement_learner import disagreement_learner
    from .calibration_monitoring import critical_tracker
    
    time_window = request.args.get('days', default=30, type=int)
    
    analytics = disagreement_learner.get_analytics(time_window)
    miss_rate = critical_tracker.get_miss_rate(time_window)
    retrain_status = disagreement_learner.should_retrain()
    
    return jsonify({
        'disagreement_analytics': analytics,
        'critical_diagnosis_tracking': miss_rate,
        'retrain_recommendation': retrain_status
    })


@main.route('/api/patients/<int:patient_id>/timeline/<int:event_index>/attachments', methods=['POST'])
def add_timeline_attachment(patient_id, event_index):
    """Add an attachment to a timeline event"""
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    from .file_handler import file_handler
    from .ai_engine import ai_engine
    
    # Save the file
    result = file_handler.save_uploaded_file(file, patient_id, 'timeline')
    
    if "error" in result:
        return jsonify(result), 400
    
    # Get event context for better analysis
    patient = generator.get_patient(patient_id)
    if patient and 0 <= event_index < len(patient.get('history', [])):
        event = patient['history'][event_index]
        event_context = {
            'date': event.get('date'),
            'event': event.get('event'),
            'notes': event.get('notes')
        }
    else:
        event_context = None
    
    # Analyze the file with AI
    try:
        file_path = file_handler.get_file_path(patient_id, result['filename'])
        analysis = ai_engine.analyze_timeline_report(file_path, event_context)
        result['analysis'] = analysis
        result['analyzed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Analysis failed: {e}")
        result['analysis'] = {
            "error": "Analysis failed",
            "summary": "File uploaded but analysis unavailable"
        }
    
    # Add attachment with analysis to timeline event
    patient = generator.add_timeline_attachment(patient_id, event_index, result)
    
    if not patient:
        return jsonify({"error": "Patient or event not found"}), 404
    
    return jsonify({"success": True, "file_info": result})

@main.route('/api/patients/<int:patient_id>/timeline/<int:event_index>/attachments', methods=['GET'])
def get_timeline_event_attachments(patient_id, event_index):
    """Get all attachments for a timeline event"""
    attachments = generator.get_timeline_attachments(patient_id, event_index)
    return jsonify(attachments)

# Surgical Risk Assessment
@main.route('/surgical-risk')
def surgical_risk_page():
    return render_template('surgical_risk.html')

@main.route('/api/surgical-risk-assessment', methods=['POST'])
def surgical_risk_assessment():
    from .surgical_risk_engine import surgical_risk_engine
    data = request.json
    print(f"DEBUG: Surgical Risk Request: {data}")
    patient_id = data.get('patient_id')
    patient = generator.get_patient(int(patient_id)) if patient_id else None
    surgery_type = data.get('surgery_type')
    surgery_duration = data.get('surgery_duration_hours', 2.0)
    
    # New procedure parameters
    urgency = data.get('urgency', 'elective')
    anesthesia_type = data.get('anesthesia_type', 'general')
    estimated_blood_loss = data.get('estimated_blood_loss', 'moderate')
    positioning = data.get('positioning', 'supine')
    
    # Custom surgery parameters
    is_custom = data.get('is_custom', False)
    custom_risk_level = data.get('custom_risk_level')
    custom_description = data.get('custom_description')
    
    if not patient or not surgery_type:
        return jsonify({"error": "Patient and surgery type required"}), 400
    
    assessment = surgical_risk_engine.comprehensive_risk_assessment(
        patient=patient, 
        surgery_type=surgery_type, 
        surgery_duration_hours=surgery_duration,
        urgency=urgency,
        anesthesia_type=anesthesia_type,
        estimated_blood_loss=estimated_blood_loss,
        positioning=positioning,
        is_custom=is_custom,
        custom_risk_level=custom_risk_level,
        custom_description=custom_description
    )
    return jsonify(assessment)


# Literature Search
@main.route('/literature')
def literature_page():
    return render_template('literature.html')

@main.route('/api/literature-search', methods=['POST'])
def literature_search():
    from .literature_engine import literature_engine
    data = request.json
    query = data.get('query')
    condition = data.get('condition', '')
    if not query:
        return jsonify({"error": "Query required"}), 400
    results = literature_engine.search_literature(query, condition)
    return jsonify(results)

@main.route('/api/clinical-trials-search', methods=['POST'])
def clinical_trials_search():
    from .literature_engine import literature_engine
    data = request.json
    condition = data.get('condition')
    patient_id = data.get('patient_id')
    patient = generator.get_patient(int(patient_id)) if patient_id else None
    if not condition:
        return jsonify({"error": "Condition required"}), 400
    results = literature_engine.find_clinical_trials(condition, patient)
    return jsonify(results)

@main.route('/api/grade-evidence', methods=['POST'])
def grade_evidence():
    from .literature_engine import literature_engine
    data = request.json
    intervention = data.get('intervention')
    condition = data.get('condition')
    if not intervention or not condition:
        return jsonify({"error": "Intervention and condition required"}), 400
    grading = literature_engine.grade_recommendation(intervention, condition)
    return jsonify(grading)


# ============ ADVANCED LITERATURE FEATURES ============

@main.route('/api/systematic-review', methods=['POST'])
def systematic_review():
    """Generate AI-powered living systematic review with meta-analysis"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    query = data.get('query')
    condition = data.get('condition', '')
    if not query:
        return jsonify({"error": "Query required"}), 400
    results = advanced_literature_engine.generate_living_systematic_review(query, condition)
    return jsonify(results)


@main.route('/api/guideline-reconciliation', methods=['POST'])
def guideline_reconciliation():
    """Compare and reconcile clinical guidelines across sources"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    condition = data.get('condition')
    intervention = data.get('intervention', '')
    if not condition:
        return jsonify({"error": "Condition required"}), 400
    results = advanced_literature_engine.reconcile_guidelines(condition, intervention)
    return jsonify(results)


@main.route('/api/pharmacogenomic-evidence', methods=['POST'])
def pharmacogenomic_evidence():
    """Get CPIC/PharmGKB evidence for drug-gene interactions"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    drug = data.get('drug')
    gene = data.get('gene', '')
    patient_id = data.get('patient_id')
    
    if not drug:
        return jsonify({"error": "Drug required"}), 400
    
    # Get patient pharmacogenomics if patient_id provided
    patient_pgx = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
        if patient:
            patient_pgx = patient.get('pharmacogenomics', {})
    
    results = advanced_literature_engine.get_pharmacogenomic_evidence(drug, gene, patient_pgx)
    return jsonify(results)


@main.route('/api/evidence-timeline', methods=['POST'])
def evidence_timeline():
    """Generate evidence timeline with landmark trials for a condition"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    condition = data.get('condition')
    patient_id = data.get('patient_id')
    
    if not condition:
        return jsonify({"error": "Condition required"}), 400
    
    patient = None
    if patient_id:
        patient = generator.get_patient(int(patient_id))
    
    results = advanced_literature_engine.generate_evidence_timeline(condition, patient)
    return jsonify(results)


@main.route('/api/patients/<int:patient_id>/evidence-timeline')
def patient_evidence_timeline(patient_id):
    """Get evidence timeline specific to patient's condition"""
    from .advanced_literature_engine import advanced_literature_engine
    patient = generator.get_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    condition = patient.get('condition', '')
    results = advanced_literature_engine.generate_evidence_timeline(condition, patient)
    return jsonify(results)


# ============ CROSS-MODULE LITERATURE INTEGRATION ============

@main.route('/api/evidence-for-diagnosis', methods=['POST'])
def evidence_for_diagnosis():
    """Get supporting literature for differential diagnoses"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    differentials = data.get('differentials', [])
    if not differentials:
        return jsonify({"error": "Differentials required"}), 400
    results = advanced_literature_engine.get_evidence_for_diagnosis(differentials)
    return jsonify(results)


@main.route('/api/evidence-for-optimizer', methods=['POST'])
def evidence_for_optimizer():
    """Get evidence supporting drug optimization recommendations"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    combinations = data.get('combinations', [])
    condition = data.get('condition', '')
    if not combinations:
        return jsonify({"error": "Combinations required"}), 400
    results = advanced_literature_engine.get_evidence_for_optimizer(combinations, condition)
    return jsonify(results)


@main.route('/api/evidence-for-simulation', methods=['POST'])
def evidence_for_simulation():
    """Get evidence for drug interactions in simulation"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    drug = data.get('drug')
    interaction_flags = data.get('interaction_flags', [])
    if not drug:
        return jsonify({"error": "Drug required"}), 400
    results = advanced_literature_engine.get_evidence_for_simulation(drug, interaction_flags)
    return jsonify(results)


@main.route('/api/evidence-for-labs', methods=['POST'])
def evidence_for_labs():
    """Get evidence for interpreting abnormal lab values"""
    from .advanced_literature_engine import advanced_literature_engine
    data = request.json
    abnormal_values = data.get('abnormal_values', [])
    condition = data.get('condition', '')
    if not abnormal_values:
        return jsonify({"error": "Abnormal values required"}), 400
    results = advanced_literature_engine.get_evidence_for_labs(abnormal_values, condition)
    return jsonify(results)


# ========== EXPERIMENTAL LAB MODULES ==========

@main.route('/deterioration-lab')
def deterioration_lab_redirect():
    return redirect(url_for('main.index'))


@main.route('/hologram-lab')
def hologram_lab_redirect():
    return redirect(url_for('main.index'))








# ========== CLINICAL DOCUMENTATION ENDPOINTS ==========


@main.route('/api/generate-patient-summary', methods=['POST'])
def generate_patient_summary():
    """Generate a comprehensive patient summary document"""
    from .clinical_note_assistant import note_assistant
    data = request.json
    patient_id = data.get('patient_id')
    special_instructions = data.get('special_instructions', '')
    
    patient = generator.get_patient(patient_id) if patient_id else None
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    # Build comprehensive summary using AI
    try:
        import google.generativeai as genai
        import os
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Generate a comprehensive Patient Summary document for clinical use.

PATIENT DATA:
- Name: {patient.get('name', 'Unknown')}
- ID: {patient.get('id')}
- Age: {patient.get('age')} years
- Sex: {patient.get('sex')}
- Primary Condition: {patient.get('condition')}
- Status: {patient.get('status')}
- Conditions/Comorbidities: {', '.join(patient.get('conditions', []))}
- Current Medications: {', '.join([m.get('name', m) if isinstance(m, dict) else m for m in patient.get('medications', [])])}
- Allergies: {', '.join(patient.get('allergies', ['NKDA']))}
- Vitals: BP {patient.get('vitals', {}).get('bp_systolic', '--')}/{patient.get('vitals', {}).get('bp_diastolic', '--')}, HR {patient.get('vitals', {}).get('heart_rate', '--')}, SpO2 {patient.get('vitals', {}).get('spO2', '--')}%, Temp {patient.get('vitals', {}).get('temp', '--')}°F
- Renal Function: {patient.get('renal_function', 'Not specified')}
- Hepatic Function: {patient.get('hepatic_function', 'Not specified')}

SPECIAL INSTRUCTIONS FROM PHYSICIAN:
{special_instructions if special_instructions else 'None provided - generate standard comprehensive summary'}

Generate a well-formatted HTML summary including:
1. CHIEF CONCERN / REASON FOR ENCOUNTER
2. MEDICAL HISTORY SUMMARY
3. CURRENT MEDICATION REGIMEN
4. RELEVANT VITALS AND LABS
5. CLINICAL ASSESSMENT
6. RECOMMENDATIONS / PLAN

Use proper HTML formatting with <h3>, <p>, <ul>, <li> tags. Keep it professional and suitable for medical records."""

        response = model.generate_content(prompt)
        content = response.text
        
        # Clean up markdown if present
        content = content.replace('```html', '').replace('```', '')
        
        return jsonify({
            "patient_name": patient.get('name'),
            "patient_id": patient.get('id'),
            "age": patient.get('age'),
            "sex": patient.get('sex'),
            "condition": patient.get('condition'),
            "content": content,
            "generated_at": "2025-12-18"
        })
        
    except Exception as e:
        # Fallback to basic summary
        content = f"""
        <h3>PATIENT SUMMARY</h3>
        <p><strong>Chief Concern:</strong> {patient.get('condition')}</p>
        
        <h3>MEDICAL HISTORY</h3>
        <ul>
            {''.join([f'<li>{c}</li>' for c in patient.get('conditions', ['No conditions recorded'])])}
        </ul>
        
        <h3>CURRENT MEDICATIONS</h3>
        <ul>
            {''.join([f'<li>{m.get("name", m) if isinstance(m, dict) else m}</li>' for m in patient.get('medications', ['No medications recorded'])])}
        </ul>
        
        <h3>VITALS</h3>
        <p>BP: {patient.get('vitals', {}).get('bp_systolic', '--')}/{patient.get('vitals', {}).get('bp_diastolic', '--')} mmHg | 
        HR: {patient.get('vitals', {}).get('heart_rate', '--')} BPM | 
        SpO2: {patient.get('vitals', {}).get('spO2', '--')}%</p>
        
        <h3>CLINICAL NOTES</h3>
        <p>Patient is currently {patient.get('status', 'stable')}. Continue current management plan.</p>
        """
        
        return jsonify({
            "patient_name": patient.get('name'),
            "patient_id": patient.get('id'),
            "age": patient.get('age'),
            "sex": patient.get('sex'),
            "condition": patient.get('condition'),
            "content": content,
            "generated_at": "2025-12-18"
        })


@main.route('/api/generate-clinical-notes', methods=['POST'])
def generate_clinical_notes():
    """Generate clinical notes (SOAP format) for patient"""
    from .clinical_note_assistant import note_assistant
    data = request.json
    patient_id = data.get('patient_id')
    special_instructions = data.get('special_instructions', '')
    
    patient = generator.get_patient(patient_id) if patient_id else None
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    try:
        import google.generativeai as genai
        import os
        
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Generate Clinical Notes in SOAP format for this patient encounter.

PATIENT DATA:
- Name: {patient.get('name', 'Unknown')}
- Age/Sex: {patient.get('age')} / {patient.get('sex')}
- Primary Condition: {patient.get('condition')}
- Status: {patient.get('status')}
- Conditions: {', '.join(patient.get('conditions', []))}
- Medications: {', '.join([m.get('name', m) if isinstance(m, dict) else m for m in patient.get('medications', [])])}
- Allergies: {', '.join(patient.get('allergies', ['NKDA']))}
- Vitals: BP {patient.get('vitals', {}).get('bp_systolic', '--')}/{patient.get('vitals', {}).get('bp_diastolic', '--')}, HR {patient.get('vitals', {}).get('heart_rate', '--')}, SpO2 {patient.get('vitals', {}).get('spO2', '--')}%

PHYSICIAN INSTRUCTIONS:
{special_instructions if special_instructions else 'Standard clinical encounter documentation'}

Generate HTML-formatted SOAP note:
<h3>SUBJECTIVE</h3>
- Chief complaint, HPI, review of systems

<h3>OBJECTIVE</h3>
- Vital signs, physical exam findings, lab results

<h3>ASSESSMENT</h3>
- Primary diagnosis, differential diagnoses, clinical reasoning

<h3>PLAN</h3>
- Treatment plan, medications, follow-up, patient education

Use HTML tags for formatting. Be thorough but concise."""

        response = model.generate_content(prompt)
        content = response.text.replace('```html', '').replace('```', '')
        
        return jsonify({
            "patient_name": patient.get('name'),
            "patient_id": patient.get('id'),
            "age": patient.get('age'),
            "sex": patient.get('sex'),
            "condition": patient.get('condition'),
            "content": content,
            "generated_at": "2025-12-18"
        })
        
    except Exception as e:
        # Fallback
        content = f"""
        <h3>SUBJECTIVE</h3>
        <p>Patient presents with {patient.get('condition')}. Current status: {patient.get('status')}.</p>
        
        <h3>OBJECTIVE</h3>
        <p><strong>Vitals:</strong> BP {patient.get('vitals', {}).get('bp_systolic', '--')}/{patient.get('vitals', {}).get('bp_diastolic', '--')}, HR {patient.get('vitals', {}).get('heart_rate', '--')}, SpO2 {patient.get('vitals', {}).get('spO2', '--')}%</p>
        <p><strong>Conditions:</strong> {', '.join(patient.get('conditions', ['None recorded']))}</p>
        
        <h3>ASSESSMENT</h3>
        <p>Primary: {patient.get('condition')}</p>
        
        <h3>PLAN</h3>
        <p>Continue current medication regimen. Follow up as scheduled.</p>
        """
        
        return jsonify({
            "patient_name": patient.get('name'),
            "patient_id": patient.get('id'),
            "content": content,
            "generated_at": "2025-12-18"
        })


@main.route('/api/generate-progress-note', methods=['POST'])
def generate_progress_note():
    """Generate a progress note"""
    from .clinical_note_assistant import note_assistant
    data = request.json
    patient_id = data.get('patient_id')
    chief_complaint = data.get('chief_complaint', '')
    hpi = data.get('hpi', '')
    
    patient = generator.get_patient(patient_id) if patient_id else None
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    encounter_data = {
        'chief_complaint': chief_complaint,
        'hpi': hpi
    }
    
    result = note_assistant.generate_progress_note(patient, encounter_data)
    return jsonify(result)


@main.route('/api/generate-discharge-summary', methods=['POST'])
def generate_discharge_summary():
    """Generate a discharge summary"""
    from .clinical_note_assistant import note_assistant
    data = request.json
    patient_id = data.get('patient_id')
    admission_date = data.get('admission_date', '')
    hospital_course = data.get('hospital_course', '')
    
    patient = generator.get_patient(patient_id) if patient_id else None
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    
    admission_data = {
        'admission_date': admission_date,
        'discharge_date': '2025-12-18',
        'hospital_course': hospital_course
    }
    
    result = note_assistant.generate_discharge_summary(patient, admission_data)
    return jsonify(result)


# ============ NEW EVIDENCE INTELLIGENCE APIS ============

@main.route('/api/drug-interactions', methods=['POST'])
def check_drug_interactions():
    """Check drug-drug interactions using AI"""
    data = request.json
    drugs = data.get('drugs', [])
    
    if len(drugs) < 2:
        return jsonify({"error": "At least 2 drugs required"}), 400
    
    import google.generativeai as genai
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""You are a clinical pharmacist analyzing drug interactions.

Analyze interactions between: {', '.join(drugs)}

For each interaction pair, provide:
1. Drug pair name
2. Severity (Major/Moderate/Minor)
3. Description of interaction
4. Mechanism
5. Clinical recommendation

Return JSON format:
{{
    "interactions": [
        {{
            "drug_pair": "Drug A + Drug B",
            "severity": "Major|Moderate|Minor",
            "description": "Clinical description of interaction",
            "mechanism": "Pharmacological mechanism",
            "recommendation": "How to manage"
        }}
    ]
}}

If no significant interactions, return empty interactions array.
Be specific and clinically accurate."""

        response = model.generate_content(prompt)
        
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            result = json.loads(json_match.group())
            return jsonify(result)
        
        return jsonify({"interactions": []})
        
    except Exception as e:
        print(f"Drug interaction error: {e}")
        return jsonify({"error": str(e)}), 500


@main.route('/api/quick-differential', methods=['POST'])
def quick_differential():
    """Generate quick differential diagnosis from symptoms"""
    data = request.json
    symptoms = data.get('symptoms', [])
    age = data.get('age', '')
    sex = data.get('sex', '')
    
    if not symptoms:
        return jsonify({"error": "Symptoms required"}), 400
    
    import google.generativeai as genai
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""You are an expert diagnostician generating a differential diagnosis.

Patient: {age} year old {sex}
Symptoms: {', '.join(symptoms) if isinstance(symptoms, list) else symptoms}

Generate top 5 differential diagnoses with:
1. Condition name
2. Probability (0-1)
3. Red flags to watch for

Return JSON:
{{
    "diagnoses": [
        {{
            "condition": "Diagnosis name",
            "probability": 0.35,
            "red_flags": ["flag1", "flag2"]
        }}
    ]
}}

Order by probability descending. Be clinically accurate."""

        response = model.generate_content(prompt)
        
        import json
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            result = json.loads(json_match.group())
            return jsonify(result)
        
        return jsonify({"diagnoses": []})
        
    except Exception as e:
        print(f"Quick differential error: {e}")
        return jsonify({"error": str(e)}), 500
