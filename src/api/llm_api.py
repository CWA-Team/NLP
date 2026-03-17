"""
LLM API Integration Module
Handles API calls to OpenAI and Google Gemini
"""

import os
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
import json

# Try to import OpenAI and Google Generative AI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMAPI:
    """Unified interface for LLM API calls."""
    
    def __init__(self, provider: str = "openai", config: Dict = None):
        """
        Initialize LLM API client.
        
        Args:
            provider: 'openai' or 'gemini'
            config: Configuration dictionary for the provider
        """
        self.provider = provider
        self.config = config or {}
        self.last_response = None
        self.response_time = 0
        
        # Initialize based on provider
        if provider == "openai":
            self._init_openai()
        elif provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = api_key
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
    
    def generate(self, prompt: str, system_prompt: str = None, 
                 temperature: float = None, max_tokens: int = None,
                 top_p: float = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary containing response, metadata, and timing
        """
        start_time = time.time()
        
        # Override config with provided parameters
        temperature = temperature or self.config.get("temperature", 0.7)
        max_tokens = max_tokens or self.config.get("max_tokens", 500)
        top_p = top_p or self.config.get("top_p", 1.0)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, system_prompt, temperature, max_tokens, top_p)
            elif self.provider == "gemini":
                response = self._call_gemini(prompt, system_prompt, temperature, max_tokens, top_p)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            self.last_response = response
            self.response_time = time.time() - start_time
            
            return {
                "success": True,
                "response": response["content"],
                "model": response.get("model", "unknown"),
                "finish_reason": response.get("finish_reason", "unknown"),
                "tokens_used": response.get("tokens_used", 0),
                "response_time": self.response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.response_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "response_time": self.response_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _call_openai(self, prompt: str, system_prompt: str, temperature: float,
                     max_tokens: int, top_p: float) -> Dict:
        """Make OpenAI API call."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.config.get("model", "gpt-3.5-turbo"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=self.config.get("frequency_penalty", 0.0),
            presence_penalty=self.config.get("presence_penalty", 0.0)
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
    
    def _call_gemini(self, prompt: str, system_prompt: str, temperature: float,
                     max_tokens: int, top_p: float) -> Dict:
        """Make Gemini API call."""
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "top_k": self.config.get("top_k", 40)
        }
        
        model = genai.GenerativeModel(
            model_name=self.config.get("model", "gemini-pro"),
            generation_config=generation_config
        )
        
        # Combine system prompt with user prompt if provided
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = model.generate_content(full_prompt)
        
        return {
            "content": response.text,
            "model": self.config.get("model", "gemini-pro"),
            "finish_reason": "stop",
            "tokens_used": 0  # Gemini doesn't provide token count
        }
    
    def generate_with_context(self, context: str, question: str, 
                              system_prompt: str = None) -> Dict[str, Any]:
        """Generate response with context."""
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(prompt, system_prompt)
    
    def batch_generate(self, prompts: List[str], system_prompt: str = None) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, system_prompt)
            results.append(result)
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
        return results


class MultiProviderLLM:
    """Manages multiple LLM providers for comparison."""
    
    def __init__(self, providers: List[str] = None):
        """
        Initialize with multiple providers.
        
        Args:
            providers: List of provider names ['openai', 'gemini']
        """
        self.providers = {}
        self.provider_configs = {}
        
        if providers is None:
            providers = ["openai", "gemini"]
        
        for provider in providers:
            try:
                if provider == "openai" and OPENAI_AVAILABLE:
                    from utils.config import LLM_CONFIG
                    self.providers[provider] = LLMAPI("openai", LLM_CONFIG["openai"])
                    self.provider_configs[provider] = LLM_CONFIG["openai"]
                elif provider == "gemini" and GEMINI_AVAILABLE:
                    from utils.config import LLM_CONFIG
                    self.providers[provider] = LLMAPI("gemini", LLM_CONFIG["gemini"])
                    self.provider_configs[provider] = LLM_CONFIG["gemini"]
            except Exception as e:
                print(f"Warning: Could not initialize {provider}: {e}")
    
    def generate(self, prompt: str, provider: str = None, **kwargs) -> Dict[str, Any]:
        """Generate using specified provider or all providers."""
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not available")
            return self.providers[provider].generate(prompt, **kwargs)
        
        # Generate with all providers
        results = {}
        for name, api in self.providers.items():
            results[name] = api.generate(prompt, **kwargs)
        
        return results
    
    def compare_responses(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Compare responses from all providers."""
        results = self.generate(prompt, **kwargs)
        
        comparison = {
            "prompt": prompt,
            "providers": {},
            "agreements": [],
            "differences": []
        }
        
        responses = []
        for provider, result in results.items():
            comparison["providers"][provider] = {
                "response": result.get("response", ""),
                "success": result.get("success", False),
                "response_time": result.get("response_time", 0)
            }
            if result.get("success"):
                responses.append(result["response"])
        
        # Check for agreements/differences
        if len(responses) > 1:
            if len(set(responses)) == 1:
                comparison["agreements"] = list(self.providers.keys())
            else:
                comparison["differences"] = list(self.providers.keys())
        
        return comparison


# Factory function to get LLM instance
def get_llm(provider: str = "openai", config: Dict = None) -> LLMAPI:
    """Get LLM API instance."""
    return LLMAPI(provider, config)


# Test function
def test_api_connections():
    """Test API connections."""
    results = {}
    
    # Test OpenAI
    if OPENAI_AVAILABLE:
        try:
            llm = get_llm("openai")
            test_result = llm.generate("Say 'test successful' in one sentence")
            results["openai"] = test_result["success"]
        except Exception as e:
            results["openai"] = f"Error: {str(e)}"
    else:
        results["openai"] = "Not available"
    
    # Test Gemini
    if GEMINI_AVAILABLE:
        try:
            llm = get_llm("gemini")
            test_result = llm.generate("Say 'test successful' in one sentence")
            results["gemini"] = test_result["success"]
        except Exception as e:
            results["gemini"] = f"Error: {str(e)}"
    else:
        results["gemini"] = "Not available"
    
    return results
