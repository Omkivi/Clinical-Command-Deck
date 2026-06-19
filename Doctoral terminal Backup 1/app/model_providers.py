"""
Model Providers - Unified interface for multiple LLM APIs with Key Rotation
Supports: Gemini, GPT-4, Mistral, Llama (via Groq)
All providers support automatic key rotation on quota errors.
"""
import os
import json
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# BASE MODEL INTERFACE
# ============================================================

class BaseModel(ABC):
    """Abstract base class for all LLM providers with key rotation support"""
    
    def __init__(self, name: str, provider: str):
        self.name = name
        self.provider = provider
        self.is_available = False
        self.keys = []
        self.current_key_index = 0
        self.key_cooldowns = {}
    
    def _load_keys(self, env_prefix: str):
        """Load all API keys for this provider"""
        keys = []
        
        # Primary key
        primary = os.getenv(env_prefix)
        if primary and primary.strip():
            keys.append(primary.strip())
        
        # Backup keys
        for i in range(2, 10):
            key = os.getenv(f'{env_prefix}_{i}')
            if key and key.strip():
                keys.append(key.strip())
        
        self.keys = keys
        self.key_cooldowns = {k: 0 for k in keys}
        
        if len(keys) > 1:
            print(f"       └─ {len(keys)} API keys loaded for rotation")
    
    def _rotate_key(self) -> bool:
        """Rotate to the next available key"""
        if len(self.keys) <= 1:
            return False
        
        # Mark current key in cooldown
        current_key = self.keys[self.current_key_index]
        self.key_cooldowns[current_key] = time.time() + 60
        
        # Find next available key
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            next_key = self.keys[self.current_key_index]
            
            if time.time() > self.key_cooldowns.get(next_key, 0):
                self._reinitialize_client(next_key)
                print(f"[INFO] {self.name}: Rotated to key #{self.current_key_index + 1}")
                return True
            
            attempts += 1
        
        self.current_key_index = original_index
        return False
    
    def _is_quota_error(self, error) -> bool:
        """Check if error is a quota/rate limit error"""
        error_str = str(error).lower()
        return any(x in error_str for x in ['429', 'quota', 'rate limit', 'too many requests', 'resource exhausted'])
    
    @abstractmethod
    def _reinitialize_client(self, new_key: str):
        """Reinitialize the client with a new API key"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> dict:
        """Generate a response from the model
        
        Returns:
            dict with keys: content, model, success, error (if any)
        """
        pass
    
    def get_info(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "available": self.is_available,
            "keys_count": len(self.keys)
        }


# ============================================================
# GEMINI MODEL (Google) - WITH KEY ROTATION
# ============================================================

class GeminiModel(BaseModel):
    """Google Gemini 2.0 Flash with automatic key rotation"""
    
    def __init__(self):
        super().__init__("Gemini 2.0 Flash", "Google")
        self._load_keys('GOOGLE_API_KEY')
        
        if self.keys:
            try:
                import google.generativeai as genai
                self.genai = genai
                genai.configure(api_key=self.keys[0])
                self.client = genai.GenerativeModel('gemini-2.0-flash')
                self.is_available = True
            except Exception as e:
                print(f"Gemini init error: {e}")
                self.is_available = False
    
    def _reinitialize_client(self, new_key: str):
        self.genai.configure(api_key=new_key)
        self.client = self.genai.GenerativeModel('gemini-2.0-flash')
    
    def generate(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> dict:
        if not self.is_available:
            return {"success": False, "error": "Gemini not available", "model": self.name}
        
        retries = 0
        while retries <= max_retries:
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.client.generate_content(full_prompt)
                return {
                    "success": True,
                    "content": response.text,
                    "model": self.name,
                    "provider": self.provider
                }
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] {self.name}: Quota exceeded, attempting rotation...")
                    if self._rotate_key():
                        retries += 1
                        continue
                return {"success": False, "error": str(e), "model": self.name}
        
        return {"success": False, "error": "All API keys exhausted", "model": self.name}


# ============================================================
# GPT-4 MODEL (OpenAI) - WITH KEY ROTATION
# ============================================================

class GPT4Model(BaseModel):
    """OpenAI GPT-4o with automatic key rotation"""
    
    def __init__(self):
        super().__init__("GPT-4o", "OpenAI")
        self._load_keys('OPENAI_API_KEY')
        
        if self.keys:
            try:
                from openai import OpenAI
                self.OpenAI = OpenAI
                self.client = OpenAI(api_key=self.keys[0])
                self.is_available = True
            except ImportError:
                print("OpenAI package not installed. Run: pip install openai")
                self.is_available = False
            except Exception as e:
                print(f"GPT-4 init error: {e}")
                self.is_available = False
    
    def _reinitialize_client(self, new_key: str):
        self.client = self.OpenAI(api_key=new_key)
    
    def generate(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> dict:
        if not self.is_available:
            return {"success": False, "error": "GPT-4 not available", "model": self.name}
        
        retries = 0
        while retries <= max_retries:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7
                )
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model": self.name,
                    "provider": self.provider
                }
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] {self.name}: Quota exceeded, attempting rotation...")
                    if self._rotate_key():
                        retries += 1
                        continue
                return {"success": False, "error": str(e), "model": self.name}
        
        return {"success": False, "error": "All API keys exhausted", "model": self.name}


# ============================================================
# MISTRAL MODEL (Mistral AI) - WITH KEY ROTATION
# ============================================================

class MistralModel(BaseModel):
    """Mistral Large with automatic key rotation"""
    
    def __init__(self):
        super().__init__("Mistral Large", "Mistral AI")
        self._load_keys('MISTRAL_API_KEY')
        
        if self.keys:
            try:
                from mistralai import Mistral
                self.Mistral = Mistral
                self.client = Mistral(api_key=self.keys[0])
                self.is_available = True
            except ImportError:
                print("Mistral package not installed. Run: pip install mistralai")
                self.is_available = False
            except Exception as e:
                print(f"Mistral init error: {e}")
                self.is_available = False
    
    def _reinitialize_client(self, new_key: str):
        self.client = self.Mistral(api_key=new_key)
    
    def generate(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> dict:
        if not self.is_available:
            return {"success": False, "error": "Mistral not available", "model": self.name}
        
        retries = 0
        while retries <= max_retries:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages
                )
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model": self.name,
                    "provider": self.provider
                }
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] {self.name}: Quota exceeded, attempting rotation...")
                    if self._rotate_key():
                        retries += 1
                        continue
                return {"success": False, "error": str(e), "model": self.name}
        
        return {"success": False, "error": "All API keys exhausted", "model": self.name}


# ============================================================
# LLAMA MODEL (via Groq - FREE & FAST) - WITH KEY ROTATION
# ============================================================

class LlamaModel(BaseModel):
    """Llama 3.1 70B via Groq with automatic key rotation"""
    
    def __init__(self):
        super().__init__("Llama 3.1 70B", "Groq")
        self._load_keys('GROQ_API_KEY')
        
        if self.keys:
            try:
                from groq import Groq
                self.Groq = Groq
                self.client = Groq(api_key=self.keys[0])
                self.is_available = True
            except ImportError:
                print("Groq package not installed. Run: pip install groq")
                self.is_available = False
            except Exception as e:
                print(f"Groq/Llama init error: {e}")
                self.is_available = False
    
    def _reinitialize_client(self, new_key: str):
        self.client = self.Groq(api_key=new_key)
    
    def generate(self, prompt: str, system_prompt: str = None, max_retries: int = 3) -> dict:
        if not self.is_available:
            return {"success": False, "error": "Llama/Groq not available", "model": self.name}
        
        retries = 0
        while retries <= max_retries:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7
                )
                return {
                    "success": True,
                    "content": response.choices[0].message.content,
                    "model": self.name,
                    "provider": self.provider
                }
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] {self.name}: Quota exceeded, attempting rotation...")
                    if self._rotate_key():
                        retries += 1
                        continue
                return {"success": False, "error": str(e), "model": self.name}
        
        return {"success": False, "error": "All API keys exhausted", "model": self.name}


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:
    """Registry of all available models with key rotation support"""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all model providers"""
        model_classes = [
            GeminiModel,
            GPT4Model,
            MistralModel,
            LlamaModel
        ]
        
        for ModelClass in model_classes:
            try:
                model = ModelClass()
                self.models[model.name] = model
                status = "✓ Available" if model.is_available else "✗ Not available"
                keys_info = f" ({len(model.keys)} keys)" if model.keys else ""
                print(f"  {model.name} ({model.provider}): {status}{keys_info}")
            except Exception as e:
                print(f"  Failed to init {ModelClass.__name__}: {e}")
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return [m for m in self.models.values() if m.is_available]
    
    def get_model(self, name: str) -> BaseModel:
        """Get a specific model by name"""
        return self.models.get(name)
    
    def get_all_models(self) -> dict:
        """Get all models (available or not)"""
        return self.models
    
    def get_key_status(self) -> dict:
        """Get API key status for all providers"""
        status = {}
        for name, model in self.models.items():
            status[name] = {
                "total_keys": len(model.keys),
                "current_key": model.current_key_index + 1 if model.keys else 0,
                "available": model.is_available
            }
        return status


# Initialize registry on import
print("[INFO] Initializing AI Model Providers with Key Rotation...")
model_registry = ModelRegistry()
print(f"[INFO] {len(model_registry.get_available_models())} models available for council\n")
