"""
Multi-Provider API Key Rotation Manager
Automatically rotates between multiple API keys for all providers when quota is exhausted.
Supports: Google Gemini, OpenAI, Mistral, Groq
"""

import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from functools import wraps

load_dotenv()


class MultiProviderKeyManager:
    """
    Manages multiple API keys for all AI providers with automatic rotation.
    
    Supported Providers:
        - Google Gemini
        - OpenAI
        - Mistral
        - Groq
    
    Usage in .env:
        GOOGLE_API_KEY=key1
        GOOGLE_API_KEY_2=key2
        OPENAI_API_KEY=key1
        OPENAI_API_KEY_2=key2
        etc.
    """
    
    def __init__(self):
        self.providers = {
            'google': {
                'keys': [],
                'current_index': 0,
                'cooldowns': {},
                'model': None
            },
            'openai': {
                'keys': [],
                'current_index': 0,
                'cooldowns': {},
                'client': None
            },
            'mistral': {
                'keys': [],
                'current_index': 0,
                'cooldowns': {},
                'client': None
            },
            'groq': {
                'keys': [],
                'current_index': 0,
                'cooldowns': {},
                'client': None
            }
        }
        
        self._load_all_keys()
        self._initialize_providers()
    
    def _load_all_keys(self):
        """Load all API keys for all providers"""
        
        # Google keys
        self._load_provider_keys('google', 'GOOGLE_API_KEY')
        
        # OpenAI keys
        self._load_provider_keys('openai', 'OPENAI_API_KEY')
        
        # Mistral keys
        self._load_provider_keys('mistral', 'MISTRAL_API_KEY')
        
        # Groq keys
        self._load_provider_keys('groq', 'GROQ_API_KEY')
        
        # Summary
        total_keys = sum(len(p['keys']) for p in self.providers.values())
        print(f"[INFO] Multi-Provider Key Manager: Loaded {total_keys} total API keys")
        for name, provider in self.providers.items():
            if provider['keys']:
                print(f"       - {name.upper()}: {len(provider['keys'])} key(s)")
    
    def _load_provider_keys(self, provider_name, env_prefix):
        """Load keys for a specific provider"""
        keys = []
        
        # Primary key
        primary = os.getenv(env_prefix)
        if primary and primary.strip():
            keys.append(primary.strip())
        
        # Backup keys (PREFIX_2, PREFIX_3, etc.)
        for i in range(2, 10):
            key = os.getenv(f'{env_prefix}_{i}')
            if key and key.strip():
                keys.append(key.strip())
        
        self.providers[provider_name]['keys'] = keys
        self.providers[provider_name]['cooldowns'] = {k: 0 for k in keys}
    
    def _initialize_providers(self):
        """Initialize all provider clients with their current keys"""
        
        # Google Gemini
        google_keys = self.providers['google']['keys']
        if google_keys:
            current_key = google_keys[0]
            genai.configure(api_key=current_key)
            self.providers['google']['model'] = genai.GenerativeModel('gemini-2.0-flash')
            print(f"[INFO] Google Gemini: Using key #1")
        
        # OpenAI
        openai_keys = self.providers['openai']['keys']
        if openai_keys:
            try:
                from openai import OpenAI
                self.providers['openai']['client'] = OpenAI(api_key=openai_keys[0])
                print(f"[INFO] OpenAI: Using key #1")
            except ImportError:
                print("[WARNING] OpenAI package not installed")
        
        # Mistral
        mistral_keys = self.providers['mistral']['keys']
        if mistral_keys:
            try:
                from mistralai import Mistral
                self.providers['mistral']['client'] = Mistral(api_key=mistral_keys[0])
                print(f"[INFO] Mistral: Using key #1")
            except ImportError:
                print("[WARNING] Mistral package not installed")
        
        # Groq
        groq_keys = self.providers['groq']['keys']
        if groq_keys:
            try:
                from groq import Groq
                self.providers['groq']['client'] = Groq(api_key=groq_keys[0])
                print(f"[INFO] Groq: Using key #1")
            except ImportError:
                print("[WARNING] Groq package not installed")
    
    def _rotate_key(self, provider_name):
        """Rotate to the next available key for a provider"""
        provider = self.providers[provider_name]
        keys = provider['keys']
        
        if len(keys) <= 1:
            print(f"[WARNING] {provider_name.upper()}: No backup keys to rotate to!")
            return False
        
        # Mark current key as in cooldown
        current_key = keys[provider['current_index']]
        provider['cooldowns'][current_key] = time.time() + 60
        
        # Find next available key
        original_index = provider['current_index']
        attempts = 0
        
        while attempts < len(keys):
            provider['current_index'] = (provider['current_index'] + 1) % len(keys)
            next_key = keys[provider['current_index']]
            
            if time.time() > provider['cooldowns'].get(next_key, 0):
                # Re-initialize the client with new key
                self._reinitialize_provider(provider_name, next_key)
                print(f"[INFO] {provider_name.upper()}: Rotated to key #{provider['current_index'] + 1}")
                return True
            
            attempts += 1
        
        print(f"[ERROR] {provider_name.upper()}: All keys in cooldown!")
        provider['current_index'] = original_index
        return False
    
    def _reinitialize_provider(self, provider_name, new_key):
        """Reinitialize a provider's client with a new key"""
        provider = self.providers[provider_name]
        
        if provider_name == 'google':
            genai.configure(api_key=new_key)
            provider['model'] = genai.GenerativeModel('gemini-2.0-flash')
        
        elif provider_name == 'openai':
            try:
                from openai import OpenAI
                provider['client'] = OpenAI(api_key=new_key)
            except:
                pass
        
        elif provider_name == 'mistral':
            try:
                from mistralai import Mistral
                provider['client'] = Mistral(api_key=new_key)
            except:
                pass
        
        elif provider_name == 'groq':
            try:
                from groq import Groq
                provider['client'] = Groq(api_key=new_key)
            except:
                pass
    
    # ========== GOOGLE GEMINI METHODS ==========
    
    def get_gemini_model(self):
        """Get the current Gemini model"""
        return self.providers['google']['model']
    
    def gemini_generate(self, prompt, max_retries=3):
        """Generate content with Gemini, auto-rotating on quota errors"""
        model = self.providers['google']['model']
        if not model:
            raise Exception("No Google API keys configured")
        
        retries = 0
        while retries <= max_retries:
            try:
                response = model.generate_content(prompt)
                return response
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] Gemini quota exceeded")
                    if self._rotate_key('google'):
                        model = self.providers['google']['model']
                        retries += 1
                        continue
                raise e
        
        raise Exception("All Gemini API keys exhausted")
    
    # ========== OPENAI METHODS ==========
    
    def get_openai_client(self):
        """Get the current OpenAI client"""
        return self.providers['openai']['client']
    
    def openai_generate(self, messages, model="gpt-4o", max_retries=3):
        """Generate with OpenAI, auto-rotating on quota errors"""
        client = self.providers['openai']['client']
        if not client:
            raise Exception("No OpenAI API keys configured")
        
        retries = 0
        while retries <= max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] OpenAI quota exceeded")
                    if self._rotate_key('openai'):
                        client = self.providers['openai']['client']
                        retries += 1
                        continue
                raise e
        
        raise Exception("All OpenAI API keys exhausted")
    
    # ========== MISTRAL METHODS ==========
    
    def get_mistral_client(self):
        """Get the current Mistral client"""
        return self.providers['mistral']['client']
    
    def mistral_generate(self, messages, model="mistral-large-latest", max_retries=3):
        """Generate with Mistral, auto-rotating on quota errors"""
        client = self.providers['mistral']['client']
        if not client:
            raise Exception("No Mistral API keys configured")
        
        retries = 0
        while retries <= max_retries:
            try:
                response = client.chat.complete(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] Mistral quota exceeded")
                    if self._rotate_key('mistral'):
                        client = self.providers['mistral']['client']
                        retries += 1
                        continue
                raise e
        
        raise Exception("All Mistral API keys exhausted")
    
    # ========== GROQ METHODS ==========
    
    def get_groq_client(self):
        """Get the current Groq client"""
        return self.providers['groq']['client']
    
    def groq_generate(self, messages, model="llama-3.1-70b-versatile", max_retries=3):
        """Generate with Groq, auto-rotating on quota errors"""
        client = self.providers['groq']['client']
        if not client:
            raise Exception("No Groq API keys configured")
        
        retries = 0
        while retries <= max_retries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                if self._is_quota_error(e):
                    print(f"[WARNING] Groq quota exceeded")
                    if self._rotate_key('groq'):
                        client = self.providers['groq']['client']
                        retries += 1
                        continue
                raise e
        
        raise Exception("All Groq API keys exhausted")
    
    # ========== UTILITY METHODS ==========
    
    def _is_quota_error(self, error):
        """Check if an error is a quota/rate limit error"""
        error_str = str(error).lower()
        return any(x in error_str for x in ['429', 'quota', 'rate limit', 'too many requests'])
    
    def get_status(self):
        """Get status of all API keys across all providers"""
        status = {}
        current_time = time.time()
        
        for provider_name, provider in self.providers.items():
            provider_status = []
            for i, key in enumerate(provider['keys']):
                cooldown = provider['cooldowns'].get(key, 0)
                provider_status.append({
                    'key_number': i + 1,
                    'is_current': i == provider['current_index'],
                    'available': current_time > cooldown,
                    'cooldown_remaining': max(0, int(cooldown - current_time))
                })
            status[provider_name] = provider_status
        
        return status


# Global instance
api_key_manager = MultiProviderKeyManager()


# Backward compatibility - single-provider interface for Gemini
class APIKeyManager:
    """Backward-compatible wrapper for Gemini-only operations"""
    
    def __init__(self):
        self.multi_manager = api_key_manager
    
    def get_model(self):
        return self.multi_manager.get_gemini_model()
    
    def generate_content(self, prompt, retry_on_quota=True, max_retries=3):
        return self.multi_manager.gemini_generate(prompt, max_retries)
    
    def upload_file(self, file_path):
        return genai.upload_file(file_path)
    
    def _rotate_key(self):
        return self.multi_manager._rotate_key('google')
    
    def get_status(self):
        return self.multi_manager.get_status()['google']
