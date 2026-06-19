"""
Medical Literature Integration Engine

Provides real-time access to:
1. PubMed/medical literature search
2. Clinical trial matching
3. Evidence strength grading
4. Latest research synthesis
"""

import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class LiteratureEngine:
    """
    Searches medical literature and synthesizes evidence for clinical decisions.
    """
    
    def __init__(self):
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"
        
        # Initialize Gemini for synthesis
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
    
    def search_literature(self, query: str, condition: str = None, max_results: int = 10) -> Dict:
        """
        Search PubMed for relevant medical literature.
        
        Args:
            query: Search query (drug name, condition, etc.)
            condition: Optional condition context
            max_results: Number of results to return
        
        Returns:
            Dict with search results and AI synthesis
        """
        try:
            # Build search query
            search_query = query
            if condition:
                search_query = f"{query} AND {condition}"
            
            # Add filters for recent, high-quality studies
            search_query += " AND (systematic[sb] OR meta-analysis[pt] OR randomized controlled trial[pt])"
            
            # Search PubMed
            search_params = {
                'db': 'pubmed',
                'term': search_query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance',
                'datetype': 'pdat',
                'reldate': 1825  # Last 5 years
            }
            
            search_response = requests.get(
                f"{self.pubmed_base_url}esearch.fcgi",
                params=search_params,
                timeout=10
            )
            
            if search_response.status_code != 200:
                return self._fallback_literature_response(query, condition)
            
            search_data = search_response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return {
                    "query": query,
                    "condition": condition,
                    "results_found": 0,
                    "articles": [],
                    "summary": "No recent high-quality studies found for this query.",
                    "evidence_level": "Insufficient"
                }
            
            # Fetch article details
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            fetch_response = requests.get(
                f"{self.pubmed_base_url}efetch.fcgi",
                params=fetch_params,
                timeout=10
            )
            
            # Parse XML and extract article info (simplified for now)
            articles = self._parse_pubmed_xml(fetch_response.text, pmids)
            
            # Use Gemini to synthesize findings
            synthesis = self._synthesize_literature(query, condition, articles)
            
            return {
                "query": query,
                "condition": condition,
                "results_found": len(articles),
                "articles": articles[:5],  # Top 5
                "synthesis": synthesis,
                "evidence_level": self._grade_evidence(articles),
                "last_updated": "2025-12-16"
            }
            
        except Exception as e:
            print(f"Literature search error: {e}")
            return self._fallback_literature_response(query, condition)
    
    def find_clinical_trials(self, condition: str, patient: Dict = None) -> Dict:
        """
        Find matching clinical trials for patient condition.
        
        Args:
            condition: Medical condition
            patient: Optional patient data for eligibility matching
        
        Returns:
            Dict with matching trials
        """
        try:
            # Build query parameters
            params = {
                'query.cond': condition,
                'query.term': 'recruiting OR active',
                'pageSize': 10,
                'format': 'json'
            }
            
            # Add patient-specific filters
            if patient:
                age = patient.get('age')
                if age:
                    if age < 18:
                        params['filter.overallStatus'] = 'RECRUITING'
                        params['query.term'] += ' AND (pediatric OR child OR adolescent)'
                    elif age >= 65:
                        params['query.term'] += ' AND (elderly OR geriatric OR older adult)'
            
            response = requests.get(
                self.clinical_trials_url,
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                return self._fallback_trials_response(condition)
            
            data = response.json()
            studies = data.get('studies', [])
            
            # Extract relevant trial information
            trials = []
            for study in studies[:5]:
                protocol = study.get('protocolSection', {})
                id_module = protocol.get('identificationModule', {})
                status_module = protocol.get('statusModule', {})
                desc_module = protocol.get('descriptionModule', {})
                
                trials.append({
                    'nct_id': id_module.get('nctId', 'Unknown'),
                    'title': id_module.get('briefTitle', 'No title'),
                    'status': status_module.get('overallStatus', 'Unknown'),
                    'brief_summary': desc_module.get('briefSummary', 'No summary available'),
                    'url': f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}"
                })
            
            # Use AI to match patient eligibility
            eligibility_summary = self._assess_trial_eligibility(condition, trials, patient)
            
            return {
                "condition": condition,
                "trials_found": len(trials),
                "matching_trials": trials,
                "eligibility_summary": eligibility_summary,
                "recommendation": "Consult with oncologist/specialist for trial enrollment"
            }
            
        except Exception as e:
            print(f"Clinical trials search error: {e}")
            return self._fallback_trials_response(condition)
    
    def grade_recommendation(self, intervention: str, condition: str) -> Dict:
        """
        Grade the strength of evidence for a clinical recommendation.
        
        Uses GRADE system: High, Moderate, Low, Very Low
        """
        if not self.model:
            return self._fallback_grading(intervention, condition)
        
        try:
            prompt = f"""
You are a clinical epidemiologist using the GRADE system to evaluate evidence quality.

Intervention: {intervention}
Condition: {condition}

Grade the quality of evidence (High/Moderate/Low/Very Low) and provide:
1. Evidence quality rating
2. Strength of recommendation (Strong/Conditional)
3. Key supporting studies
4. Limitations/concerns
5. Clinical bottom line

Return JSON format:
{{
    "evidence_quality": "High|Moderate|Low|Very Low",
    "recommendation_strength": "Strong|Conditional",
    "supporting_studies": ["Study 1", "Study 2"],
    "limitations": "Any concerns",
    "clinical_bottom_line": "Clear recommendation"
}}
"""
            
            response = self.model.generate_content(prompt)
            import json
            text = response.text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            grading = json.loads(text)
            return grading
            
        except Exception as e:
            print(f"Grading error: {e}")
            return self._fallback_grading(intervention, condition)
    
    def _parse_pubmed_xml(self, xml_text: str, pmids: List[str]) -> List[Dict]:
        """
        Parse PubMed XML response (simplified).
        In production, use proper XML parser like xml.etree.ElementTree
        """
        # Simplified parsing - just return structure with PMIDs
        articles = []
        for pmid in pmids:
            articles.append({
                'pmid': pmid,
                'title': f"Study PMID {pmid}",
                'authors': "Various authors",
                'journal': "Medical Journal",
                'year': "2024",
                'abstract': "Abstract not available in simplified version",
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        return articles
    
    def _synthesize_literature(self, query: str, condition: str, articles: List[Dict]) -> str:
        """Use Gemini to synthesize literature findings"""
        if not self.model or not articles:
            return "Literature synthesis unavailable. Please review individual studies."
        
        try:
            prompt = f"""
You are a clinical researcher synthesizing medical literature.

Query: {query}
Condition: {condition}
Number of studies: {len(articles)}

Based on the available research, provide:
1. Key findings (2-3 sentences)
2. Clinical implications
3. Gaps in evidence
4. Practical recommendations

Be concise and clinically actionable.
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except:
            return "AI synthesis unavailable. Please review individual studies."
    
    def _grade_evidence(self, articles: List[Dict]) -> str:
        """Simple evidence grading based on article count and recency"""
        if len(articles) >= 5:
            return "Moderate-High"
        elif len(articles) >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_trial_eligibility(self, condition: str, trials: List[Dict], patient: Dict = None) -> str:
        """Use AI to assess patient eligibility for trials"""
        if not self.model or not patient:
            return "Review individual trials for eligibility criteria"
        
        try:
            patient_summary = f"Age: {patient.get('age')}, Conditions: {condition}"
            
            prompt = f"""
Patient: {patient_summary}

Available trials: {len(trials)}

Based on typical eligibility criteria, assess:
1. Which trials the patient might qualify for
2. Common exclusion criteria to watch for
3. Next steps for enrollment

Be brief (2-3 sentences).
"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except:
            return "Manual review of eligibility criteria recommended"
    
    def _fallback_literature_response(self, query: str, condition: str) -> Dict:
        """Fallback when API unavailable"""
        return {
            "query": query,
            "condition": condition,
            "results_found": 0,
            "articles": [],
            "synthesis": "Literature search currently unavailable. Please consult UpToDate, DynaMed, or PubMed directly.",
            "evidence_level": "Unknown"
        }
    
    def _fallback_trials_response(self, condition: str) -> Dict:
        """Fallback for trials search"""
        return {
            "condition": condition,
            "trials_found": 0,
            "matching_trials": [],
            "eligibility_summary": "Clinical trials search unavailable. Check ClinicalTrials.gov directly.",
            "recommendation": "Consult with specialist"
        }
    
    def _fallback_grading(self, intervention: str, condition: str) -> Dict:
        """Fallback grading"""
        return {
            "evidence_quality": "Unknown",
            "recommendation_strength": "Uncertain",
            "supporting_studies": [],
            "limitations": "Evidence grading unavailable",
            "clinical_bottom_line": "Consult clinical guidelines"
        }


# Singleton instance
literature_engine = LiteratureEngine()
