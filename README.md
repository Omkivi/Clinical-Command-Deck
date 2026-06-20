# Doctoral Terminal — Clinical Command Deck

> **2nd Prize — SparkX, IIT Bombay Techfest 2024**  
> India's largest national AI competition · Backed by MeitY, Digital India & INDIAai · 10,000+ students · 20+ cities

*Solo build · December 2024 · Zero budget*

---

## What This Is

A production-grade clinical AI system designed to assist in life-critical medical decisions. Unlike single-model AI tools, this system makes four independent AI models assess, debate, and vote on every clinical case — ensuring minority opinions are never silenced in decisions that affect human lives.

---

## Scale

| Metric | Value |
|--------|-------|
| Lines of Python | ~22,000 |
| Clinical engines | 13 specialised modules |
| REST API endpoints | 75+ |
| AI models integrated | 4 (GPT-4o, Gemini 2.0, Mistral Large, Llama 3.1) |
| Drug pairs screened | 5,000+ |
| API accounts rotated | 12+ (zero-budget resilience) |

---

## Core Innovation — 4-LLM Consensus Architecture

Single-model AI systems carry systematic bias. In healthcare, that bias can cost lives.

This system runs four AI models in parallel. Each independently assesses the clinical case. They then debate and vote. Minority opinions are always surfaced — overconfidence is penalised. No single model has the final word.

```
Patient Case
     │
     ├──▶ GPT-4o      (Academic validation)
     ├──▶ Gemini 2.0  (Lead diagnostician)
     ├──▶ Mistral     (Clinical protocols)
     └──▶ Llama 3.1   (Safety officer)
          │
          ▼
     Independent Assessment
          │
          ▼
     Structured Debate
          │
          ▼
     Consensus Verdict
     (minority opinions always shown)
```

---

## 13 Clinical Engines

| Engine | Function |
|--------|----------|
| AI Council | 4-LLM deliberation & consensus |
| Diagnosis Engine | Bayesian differential diagnosis (50+ diseases) |
| Drug Simulator | Drug safety & outcome simulation |
| Therapeutic Optimizer | Optimal multi-drug combinations |
| Contraindication Oracle | Drug-drug & drug-disease interaction screening |
| Pharmacogenomics Engine | CYP450-aware personalised dosing |
| Imaging Intelligence | 5-layer medical image analysis |
| Lab Analyser | Lab report interpretation & critical flagging |
| Sepsis Detector | Real-time qSOFA / SOFA / SIRS scoring |
| Surgical Risk Engine | ASA / RCRI / NSQIP calculators |
| Literature Engine | Real-time PubMed evidence synthesis |
| Unified Medical Analyser | Universal document & image processing |
| Patient Data Aggregator | Cross-engine patient context integration |

---

## Budget Engineering

Sustaining 15–20 parallel LLM calls per query required rotating API keys across 12+ accounts. What started as a constraint became a production resilience feature.

---

## Tech Stack

Python · Flask · GPT-4o · Gemini 2.0 Flash · Mistral Large · Llama 3.1 (Groq) · PubMed API · Bootstrap 5

---

## Recognition

**2nd Prize · SparkX · IIT Bombay Techfest · December 2024**  
Organised by IIT Bombay · IIIT Allahabad · STEMLearn.AI  
Backed by MeitY · Digital India · INDIAai  
10,000+ students · 20+ cities · 100+ finalists

---

## Safety Framework

This system is built as **decision support, never a decision maker.**

- All outputs include confidence intervals
- Life-threatening conditions flagged immediately
- Full AI council deliberation logged for audit
- Dissenting AI opinions always shown to the clinician
- Human-in-the-loop design is non-negotiable

---

## Setup

```bash
git clone https://github.com/Omkivi/Clinical-Command-Deck
cd Clinical-Command-Deck
pip install -r requirements.txt
cp .env.example .env   # Add your API keys
python run.py
```

---

## Contact

**Om Vivek Mahajan**  
ommahajankivi@gmail.com

---
