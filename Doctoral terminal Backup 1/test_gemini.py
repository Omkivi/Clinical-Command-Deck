
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

try:
    genai.configure(api_key=api_key)

    # Try 2.0 Flash
    print("Testing gemini-2.0-flash...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Hello")
    print(f"2.0 Flash Response: {response.text}")
    print("SUCCESS: gemini-2.0-flash works.")
    
except Exception as e:
    print(f"ERROR: {e}")
