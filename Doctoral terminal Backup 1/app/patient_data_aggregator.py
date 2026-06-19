"""
Patient Data Aggregator

Central utility to extract and aggregate patient data from multiple sources:
- Analyzed lab reports
- Radiology findings
- Timeline medical events
- Abnormal values detection

Used by all clinical engines for comprehensive patient context.
"""

import re
from datetime import datetime, timedelta


class PatientDataAggregator:
    """Aggregates patient data from lab reports, timeline, and radiology"""
    
    def __init__(self):
        # Common lab test name variations
        self.lab_name_mapping = {
            'creatinine': ['creatinine', 'cr', 'serum creatinine'],
            'egfr': ['egfr', 'gfr', 'estimated gfr'],
            'glucose': ['glucose', 'blood glucose', 'fasting glucose', 'random glucose'],
            'hba1c': ['hba1c', 'a1c', 'hemoglobin a1c', 'glycated hemoglobin'],
            'ast': ['ast', 'aspartate aminotransferase', 'sgot'],
            'alt': ['alt', 'alanine aminotransferase', 'sgpt'],
            'bilirubin': ['bilirubin', 'total bilirubin'],
            'wbc': ['wbc', 'white blood cell', 'leukocyte'],
            'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
            'platelet': ['platelet', 'plt'],
            'inr': ['inr', 'international normalized ratio'],
            'potassium': ['potassium', 'k', 'k+'],
            'sodium': ['sodium', 'na', 'na+'],
            'ldl': ['ldl', 'ldl cholesterol', 'low density lipoprotein'],
            'hdl': ['hdl', 'hdl cholesterol', 'high density lipoprotein'],
            'triglycerides': ['triglycerides', 'tg']
        }
    
    def get_latest_lab_values(self, patient):
        """
        Extract latest lab values from analyzed reports
        
        Returns:
            dict: {test_name: {'value': X, 'date': 'YYYY-MM-DD', 'status': 'normal/high/low'}}
        """
        lab_values = {}
        lab_reports = patient.get('lab_reports', [])
        
        for report in lab_reports:
            analysis = report.get('analysis', {})
            if not analysis:
                continue
            
            # Extract from key findings
            key_findings = analysis.get('key_findings', [])
            for finding in key_findings:
                # Try to parse "Test: Value" patterns
                lab_info = self._parse_lab_finding(finding)
                if lab_info:
                    test_name = lab_info['test']
                    # Only update if this report is newer or test doesn't exist
                    if test_name not in lab_values or self._is_newer(report.get('date'), lab_values[test_name].get('date')):
                        lab_values[test_name] = {
                            'value': lab_info['value'],
                            'date': report.get('date', ''),
                            'status': 'normal',
                            'source': 'lab_report'
                        }
            
            # Check abnormal values
            abnormal_values = analysis.get('abnormal_values', [])
            for abnormal in abnormal_values:
                lab_info = self._parse_lab_finding(abnormal)
                if lab_info:
                    test_name = lab_info['test']
                    if test_name not in lab_values or self._is_newer(report.get('date'), lab_values[test_name].get('date')):
                        lab_values[test_name] = {
                            'value': lab_info['value'],
                            'date': report.get('date', ''),
                            'status': 'abnormal',
                            'source': 'lab_report'
                        }
        
        # Also check timeline attachments
        timeline_labs = self._get_labs_from_timeline(patient)
        for test_name, value_info in timeline_labs.items():
            if test_name not in lab_values or self._is_newer(value_info.get('date'), lab_values.get(test_name, {}).get('date')):
                lab_values[test_name] = value_info
        
        return lab_values
    
    def _get_labs_from_timeline(self, patient):
        """Extract lab values from timeline event attachments"""
        lab_values = {}
        history = patient.get('history', [])
        
        for event in history:
            attachments = event.get('attachments', [])
            for attachment in attachments:
                analysis = attachment.get('analysis', {})
                if not analysis:
                    continue
                
                # Only process if it's a lab report
                doc_type = analysis.get('document_type', '').lower()
                if 'lab' in doc_type or 'blood' in doc_type:
                    key_findings = analysis.get('key_findings', [])
                    for finding in key_findings:
                        lab_info = self._parse_lab_finding(finding)
                        if lab_info:
                            test_name = lab_info['test']
                            lab_values[test_name] = {
                                'value': lab_info['value'],
                                'date': event.get('date', ''),
                                'status': 'normal',
                                'source': 'timeline'
                            }
                    
                    # Check abnormal values
                    abnormal_values = analysis.get('abnormal_values', [])
                    for abnormal in abnormal_values:
                        lab_info = self._parse_lab_finding(abnormal)
                        if lab_info:
                            test_name = lab_info['test']
                            lab_values[test_name] = {
                                'value': lab_info['value'],
                                'date': event.get('date', ''),
                                'status': 'abnormal',
                                'source': 'timeline'
                            }
        
        return lab_values
    
    def _parse_lab_finding(self, finding_text):
        """
        Parse lab finding text to extract test name and value
        Handles formats like:
        - "Creatinine: 1.8 mg/dL"
        - "WBC: 12,500 cells/mcL (elevated)"
        - "Glucose 180 mg/dL"
        """
        if not isinstance(finding_text, str):
            return None
        
        # Pattern: TestName: Value units
        match = re.search(r'([A-Za-z0-9\s]+)[:\s]+([0-9.,]+)\s*([a-zA-Z/%]*)', finding_text)
        if not match:
            return None
        
        test_name = match.group(1).strip().lower()
        value_str = match.group(2).replace(',', '')
        
        # Normalize test name
        normalized_name = self._normalize_lab_name(test_name)
        if not normalized_name:
            normalized_name = test_name  # Keep original if no mapping
        
        try:
            value = float(value_str)
            return {
                'test': normalized_name,
                'value': value,
                'raw_text': finding_text
            }
        except ValueError:
            return None
    
    def _normalize_lab_name(self, test_name):
        """Normalize lab test names to standard format"""
        test_name_lower = test_name.lower().strip()
        
        for standard_name, variations in self.lab_name_mapping.items():
            if any(var in test_name_lower for var in variations):
                return standard_name
        
        return None
    
    def _is_newer(self, date1, date2):
        """Check if date1 is newer than date2"""
        if not date1 or not date2:
            return True
        
        try:
            d1 = datetime.strptime(date1.split()[0], '%Y-%m-%d')
            d2 = datetime.strptime(date2.split()[0], '%Y-%m-%d')
            return d1 > d2
        except:
            return True
    
    def get_radiology_findings(self, patient):
        """
        Extract radiology findings from reports
        
        Returns:
            list: [{finding: str, date: str, modality: str, source: str}]
        """
        findings = []
        
        # From radiology_studies
        radiology_studies = patient.get('radiology_studies', [])
        for study in radiology_studies:
            findings.append({
                'finding': study.get('findings', ''),
                'date': study.get('date', ''),
                'modality': study.get('modality', ''),
                'source': 'radiology_studies'
            })
        
        # From timeline attachments
        history = patient.get('history', [])
        for event in history:
            attachments = event.get('attachments', [])
            for attachment in attachments:
                analysis = attachment.get('analysis', {})
                if not analysis:
                    continue
                
                doc_type = analysis.get('document_type', '').lower()
                if any(term in doc_type for term in ['imaging', 'x-ray', 'ct', 'mri', 'ultrasound', 'radiology']):
                    key_findings = analysis.get('key_findings', [])
                    summary = analysis.get('summary', '')
                    
                    findings.append({
                        'finding': summary if summary else ', '.join(key_findings),
                        'date': event.get('date', ''),
                        'modality': doc_type,
                        'source': 'timeline'
                    })
        
        return findings
    
    def get_timeline_medical_events(self, patient):
        """
        Extract structured medical events from timeline
        
        Returns:
            list: [{event: str, date: str, notes: str, category: str}]
        """
        events = []
        history = patient.get('history', [])
        
        for event in history:
            event_type = event.get('event', '').lower()
            
            # Categorize events
            category = 'other'
            if any(term in event_type for term in ['surgery', 'operation', 'procedure']):
                category = 'surgical'
            elif any(term in event_type for term in ['infection', 'uti', 'pneumonia', 'sepsis']):
                category = 'infectious'
            elif any(term in event_type for term in ['medication', 'drug', 'started', 'prescribed']):
                category = 'medication'
            elif any(term in event_type for term in ['diagnosis', 'diagnosed']):
                category = 'diagnosis'
            elif any(term in event_type for term in ['hospitalization', 'admitted', 'er', 'emergency']):
                category = 'hospitalization'
            
            events.append({
                'event': event.get('event', ''),
                'date': event.get('date', ''),
                'notes': event.get('notes', ''),
                'category': category,
                'attachments_count': len(event.get('attachments', []))
            })
        
        return events
    
    def get_abnormal_labs(self, patient):
        """
        Get all abnormal lab values
        
        Returns:
            list: [{test: str, value: float, date: str, reason: str}]
        """
        all_labs = self.get_latest_lab_values(patient)
        abnormal = []
        
        for test_name, lab_info in all_labs.items():
            if lab_info.get('status') == 'abnormal':
                abnormal.append({
                    'test': test_name,
                    'value': lab_info.get('value'),
                    'date': lab_info.get('date'),
                    'reason': f"{test_name} is abnormal"
                })
        
        return abnormal
    
    def has_existing_test(self, patient, test_name):
        """
        Check if a test already exists in patient records (within last 30 days)
        
        Args:
            patient: patient dict
            test_name: test name to check (e.g., 'CBC', 'Creatinine')
        
        Returns:
            dict: {exists: bool, date: str, age_days: int} or {exists: False}
        """
        test_name_lower = test_name.lower()
        
        # Check lab reports
        lab_reports = patient.get('lab_reports', [])
        for report in lab_reports:
            report_text = report.get('report_text', '').lower()
            doc_type = report.get('analysis', {}).get('document_type', '').lower()
            
            if test_name_lower in report_text or test_name_lower in doc_type:
                date = report.get('date', '')
                if date:
                    age_days = self._days_since(date)
                    if age_days is not None and age_days <= 30:
                        return {
                            'exists': True,
                            'date': date,
                            'age_days': age_days,
                            'source': 'lab_reports'
                        }
        
        # Check timeline attachments
        history = patient.get('history', [])
        for event in history:
            attachments = event.get('attachments', [])
            for attachment in attachments:
                doc_type = attachment.get('analysis', {}).get('document_type', '').lower()
                summary = attachment.get('analysis', {}).get('summary', '').lower()
                
                if test_name_lower in doc_type or test_name_lower in summary:
                    date = event.get('date', '')
                    if date:
                        age_days = self._days_since(date)
                        if age_days is not None and age_days <= 30:
                            return {
                                'exists': True,
                                'date': date,
                                'age_days': age_days,
                                'source': 'timeline'
                            }
        
        return {'exists': False}
    
    def _days_since(self, date_str):
        """Calculate days since a date"""
        try:
            date = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
            delta = datetime.now() - date
            return delta.days
        except:
            return None
    
    def get_comprehensive_summary(self, patient):
        """
        Get comprehensive patient summary including all extracted data
        
        Returns:
            dict with labs, radiology, events, abnormalities
        """
        return {
            'lab_values': self.get_latest_lab_values(patient),
            'radiology_findings': self.get_radiology_findings(patient),
            'timeline_events': self.get_timeline_medical_events(patient),
            'abnormal_labs': self.get_abnormal_labs(patient),
            'basic_info': {
                'age': patient.get('age'),
                'sex': patient.get('sex'),
                'allergies': patient.get('allergies', []),
                'current_medications': [m.get('name') if isinstance(m, dict) else m for m in patient.get('current_medications', [])]
            }
        }


# Singleton instance
patient_data_aggregator = PatientDataAggregator()
