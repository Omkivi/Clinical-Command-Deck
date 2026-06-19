import json
import os

def load_knowledge(json_path: str) -> dict:
    """Load the medical knowledge base from a JSON file.

    The JSON should have the following top‑level structure:
    {
        "diseases": {
            "Disease Name": {
                "base_rate": float,
                "symptoms": {"Symptom": probability, ...},
                "clinical_note": "optional note"
            },
            ...
        }
    }
    """
    if not os.path.isabs(json_path):
        json_path = os.path.abspath(json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Knowledge file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Basic validation
    if "diseases" not in data or not isinstance(data["diseases"], dict):
        raise ValueError("Invalid knowledge format – missing 'diseases' dict.")
    return data["diseases"]
