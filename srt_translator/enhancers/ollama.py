"""
Ollama Enhancement Provider

This module provides subtitle enhancement using Ollama models.
It improves translation quality by refining subtitles with context-aware AI.
Uses structured JSON for both requests and responses to ensure reliable parsing.
"""

import logging
import json
import srt
from typing import List, Optional, Dict, Any, Union

try:
    import ollama
    from ollama import Client, ChatResponse
    import httpx
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .base import EnhancementProvider


logger = logging.getLogger("srt_translator")


class OllamaEnhancer(EnhancementProvider):
    """
    Enhance subtitles using Ollama models with structured JSON I/O.
    
    This enhancer uses locally running Ollama models or a custom Ollama API endpoint
    to improve translation quality, ensuring better context, natural language flow,
    and cultural appropriateness. Uses JSON format for reliable parsing of model outputs.
    """
    
    def __init__(
        self, 
        source_lang: str, 
        target_lang: str, 
        model: str = "llama3", 
        host: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        context_window: int = 4,
        **kwargs
    ):
        """
        Initialize the Ollama enhancer.
        
        Args:
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            model: Ollama model name to use
            host: Optional custom Ollama API endpoint (e.g., "http://localhost:11434")
            temperature: Model temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            context_window: Number of surrounding subtitles to include for context
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(source_lang, target_lang, **kwargs)
        
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama package is not installed. Please install it with 'pip install ollama'"
            )
        
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.context_window = context_window
        self.host = host
        
        # Initialize Ollama client
        self.client = Client(host=host)
        logger.info(f"Initialized Ollama enhancer with model: {model}")
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self) -> None:
        """
        Test connection to Ollama and check if the model is available.
        
        This method attempts to list available models and checks if the
        specified model is available. If not, it logs a warning but doesn't
        raise an error to allow for graceful fallback.
        """
        try:
            # Use direct HTTP request to ensure consistent JSON response
            base_url = self.host or "http://127.0.0.1:11434"
            if not base_url.startswith(("http://", "https://")):
                base_url = f"http://{base_url}"
            
            # Ensure the URL ends with the correct API path
            api_url = f"{base_url.rstrip('/')}/api/tags"
            
            # Make the request
            logger.info(f"Testing connection to Ollama API at: {api_url}")
            response = httpx.get(api_url, timeout=10.0)
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            logger.debug(f"Ollama API response: {json.dumps(data, indent=2)}")
            
            # Extract model names
            model_names = []
            if "models" in data:
                model_names = [model["name"] for model in data["models"]]
            elif isinstance(data, dict) and any(isinstance(data.get(k), list) for k in data):
                # Find any list in the response that might contain models
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and "name" in item:
                                model_names.append(item["name"])
            
            # If we still don't have models, check if the response itself is a list
            if not model_names and isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        model_names.append(item["name"])
            
            logger.info(f"Available Ollama models: {model_names}")
            
            if model_names and self.model not in model_names:
                logger.warning(f"Model '{self.model}' not found in available models: {model_names}")
                logger.info(f"You may need to pull the model first with 'ollama pull {self.model}'")
                # Don't raise an error, just warn and continue
            elif not model_names:
                logger.warning("No models found in Ollama response. Continuing anyway.")
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            # Don't raise an error, just log and continue
            logger.warning(f"Continuing without enhancement due to Ollama connection error: {e}")
    
    def _format_json_prompt(self, subtitle: srt.Subtitle, context: Optional[List[srt.Subtitle]] = None) -> str:
        """
        Format the prompt for the Ollama model with explicit JSON output instructions.
        
        Args:
            subtitle: The subtitle to enhance
            context: Optional list of surrounding subtitles for context
            
        Returns:
            Formatted prompt string for the model with JSON instructions
        """
        # Context information
        context_subtitles = []
        if context:
            for ctx_sub in context:
                # Include timing information to help understand scene flow
                context_subtitles.append({
                    "index": ctx_sub.index,
                    "content": ctx_sub.content,
                    "start_time": str(ctx_sub.start).split('.')[0],
                    "end_time": str(ctx_sub.end).split('.')[0],
                    "is_current": ctx_sub.index == subtitle.index
                })
        
        # Construct the prompt with explicit JSON output instructions
        prompt = f"""You are a professional subtitle translator and editor with expertise in {self.target_lang}.

I will provide you with a translated subtitle that needs improvement, along with surrounding context.
Your task is to enhance the translation to make it more natural, fluent, and contextually accurate.

IMPORTANT GUIDELINES:
1. Maintain narrative coherence with previous and next subtitles
2. Preserve the original meaning and important details
3. Fix any grammar, spelling, or punctuation errors
4. Use appropriate language register and cultural nuances for {self.target_lang}
5. Keep the subtitle concise and easy to read
6. Ensure the enhanced subtitle flows naturally with the surrounding dialogue

Original language: {self.source_lang}
Target language: {self.target_lang}

"""
        
        # Add comprehensive context if available
        if context_subtitles:
            prompt += "CONTEXT (SURROUNDING SUBTITLES):\n"
            prompt += "These subtitles form a continuous dialogue. Use this context to ensure your enhancement maintains narrative coherence:\n\n"
            
            for ctx in context_subtitles:
                if ctx["is_current"]:
                    prompt += f"[CURRENT SUBTITLE TO ENHANCE] #{ctx['index']} ({ctx['start_time']} → {ctx['end_time']}): {ctx['content']}\n"
                else:
                    # Indicate if subtitle comes before or after the current one
                    position = "BEFORE" if ctx["index"] < subtitle.index else "AFTER"
                    prompt += f"[{position}] #{ctx['index']} ({ctx['start_time']} → {ctx['end_time']}): {ctx['content']}\n"
            prompt += "\n"
            
            # Add specific instructions about maintaining coherence
            prompt += "Note how these subtitles connect with each other. Your enhancement should maintain this narrative flow.\n\n"
        
        # Add the subtitle to enhance with detailed JSON instructions
        prompt += f"""Please enhance the following subtitle translation:

"{subtitle.content}"

RESPOND USING ONLY THE FOLLOWING JSON FORMAT:
{{
  "enhanced_subtitle": "your improved subtitle text here",
  "grammar_fixes": ["list of grammar issues fixed"],
  "coherence_notes": "brief note on how this connects to surrounding subtitles"
}}

IMPORTANT GUIDELINES:
1. Provide the enhanced subtitle text in the JSON response as the primary output
2. Do not include any explanations, notes, or metadata outside the JSON structure
3. Preserve the original meaning while making it sound natural in {self.target_lang}
4. Keep the length similar to the original to maintain proper timing
5. Do not add any commentary or meta-text in the subtitle itself
6. Fix all grammar, spelling, and punctuation errors
7. Ensure the subtitle flows naturally with previous and next subtitles
8. Maintain consistent terminology, pronouns, and references across subtitles
9. Use culturally appropriate expressions and idioms for {self.target_lang}
10. The 'grammar_fixes' and 'coherence_notes' fields are for internal processing only and won't be shown to users
"""
        
        return prompt
    
    def enhance_subtitle(self, subtitle: srt.Subtitle, context: Optional[List[srt.Subtitle]] = None) -> srt.Subtitle:
        """
        Enhance a single subtitle using Ollama with JSON format.
        
        Args:
            subtitle: The subtitle to enhance
            context: Optional list of surrounding subtitles for context
            
        Returns:
            Enhanced subtitle with improved translation
        """
        try:
            # Format the JSON prompt
            prompt = self._format_json_prompt(subtitle, context)
            
            # Call Ollama API with JSON format
            try:
                # Use the format parameter if available in this version of Ollama
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    format="json",  # Request JSON format
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                    stream=False  # Ensure we get a complete response
                )
            except TypeError:
                # Fall back to standard generate if format parameter is not supported
                logger.warning("JSON format parameter not supported in this Ollama version, using standard generate")
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                )
            
            # Extract the enhanced text from the JSON response
            enhanced_text = subtitle.content  # Default to original content
            
            try:
                # Handle different response formats
                if isinstance(response, dict) and "response" in response:
                    # Parse the JSON string from the response
                    json_str = response["response"].strip()
                    logger.debug(f"Raw JSON response: {json_str}")
                    
                    try:
                        # Try to parse the JSON response
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict) and "enhanced_subtitle" in parsed_json:
                            enhanced_text = parsed_json["enhanced_subtitle"].strip()
                            
                            # Log grammar fixes and coherence notes if available
                            if "grammar_fixes" in parsed_json and isinstance(parsed_json["grammar_fixes"], list):
                                grammar_fixes = parsed_json["grammar_fixes"]
                                if grammar_fixes:
                                    logger.debug(f"Grammar fixes: {', '.join(grammar_fixes)}")
                            
                            if "coherence_notes" in parsed_json and isinstance(parsed_json["coherence_notes"], str):
                                coherence_notes = parsed_json["coherence_notes"]
                                if coherence_notes:
                                    logger.debug(f"Coherence notes: {coherence_notes}")
                                    
                            logger.debug(f"Successfully extracted enhanced subtitle from JSON")
                        else:
                            # If the response is valid JSON but doesn't have our expected field,
                            # try to find any field that might contain the subtitle
                            for key, value in parsed_json.items():
                                if isinstance(value, str) and len(value) > 5:
                                    enhanced_text = value.strip()
                                    logger.debug(f"Using field '{key}' as enhanced subtitle")
                                    break
                            else:
                                logger.warning(f"JSON response missing 'enhanced_subtitle' field: {json_str}")
                    except json.JSONDecodeError as e:
                        # If it's not valid JSON, try to extract text between curly braces
                        logger.warning(f"Failed to parse JSON response: {e}. Response: {json_str}")
                        
                        # Try to extract any JSON-like structure from the response
                        import re
                        json_matches = re.findall(r'\{.*?\}', json_str, re.DOTALL)
                        for json_match in json_matches:
                            try:
                                parsed_json = json.loads(json_match)
                                if isinstance(parsed_json, dict) and "enhanced_subtitle" in parsed_json:
                                    enhanced_text = parsed_json["enhanced_subtitle"].strip()
                                    logger.debug(f"Extracted enhanced subtitle from JSON fragment")
                                    break
                            except json.JSONDecodeError:
                                continue
                        
                        # If we still couldn't find JSON, use the whole response as the enhanced text
                        # but remove any markdown code block markers and obvious JSON syntax
                        if enhanced_text == subtitle.content:
                            cleaned_text = json_str.replace("```json", "").replace("```", "").strip()
                            cleaned_text = re.sub(r'^\s*\{\s*"enhanced_subtitle"\s*:\s*"(.+?)"\s*\}\s*$', r'\1', cleaned_text, flags=re.DOTALL)
                            if cleaned_text and len(cleaned_text) > 5:
                                enhanced_text = cleaned_text
                                logger.debug(f"Using cleaned response as enhanced subtitle")
                
                elif hasattr(response, "response"):
                    # Handle object with response attribute
                    json_str = response.response.strip()
                    logger.debug(f"Raw JSON response from attribute: {json_str}")
                    
                    try:
                        parsed_json = json.loads(json_str)
                        if isinstance(parsed_json, dict) and "enhanced_subtitle" in parsed_json:
                            enhanced_text = parsed_json["enhanced_subtitle"].strip()
                        else:
                            logger.warning(f"JSON response missing 'enhanced_subtitle' field: {json_str}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON response from attribute: {e}. Response: {json_str}")
                else:
                    logger.warning(f"Unexpected response format: {type(response)}")
                    
                # Fallback: Try to extract JSON from any string in the response
                if enhanced_text == subtitle.content and isinstance(response, dict):
                    for key, value in response.items():
                        if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                            try:
                                parsed_json = json.loads(value.strip())
                                if isinstance(parsed_json, dict) and "enhanced_subtitle" in parsed_json:
                                    enhanced_text = parsed_json["enhanced_subtitle"].strip()
                                    break
                            except json.JSONDecodeError:
                                continue
            
            except Exception as e:
                logger.warning(f"Error extracting enhanced text from response: {e}")
            
            # Remove any quotes that might be in the response
            enhanced_text = enhanced_text.strip('"\'')
            
            # Create a new subtitle with the enhanced text
            enhanced_subtitle = srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=enhanced_text
            )
            
            return enhanced_subtitle
        
        except Exception as e:
            logger.error(f"Error enhancing subtitle {subtitle.index}: {e}")
            # Return the original subtitle if enhancement fails
            return subtitle
    
    def enhance_batch(self, subtitles: List[srt.Subtitle], batch_size: int = 5, progress_callback=None) -> List[srt.Subtitle]:
        """
        Enhance a batch of subtitles with comprehensive context for improved narrative coherence.
        
        This method processes each subtitle with awareness of its surrounding context,
        ensuring that the enhanced translations maintain narrative flow and consistency
        across the entire subtitle sequence.
        
        Args:
            subtitles: List of subtitles to enhance
            batch_size: Number of subtitles to process at once (not used in this implementation)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of enhanced subtitles with improved grammar and narrative coherence
        """
        enhanced_subtitles = []
        total = len(subtitles)
        completed = 0
        
        # Create a progress lock for thread safety
        import threading
        progress_lock = threading.RLock()
        
        # Start timing
        import time
        start_time = time.time()
        
        # Determine optimal context window based on subtitle density
        # For dense dialogue, we want more context
        if total > 0:
            avg_duration = sum((s.end - s.start).total_seconds() for s in subtitles) / total
            # Adjust context window based on subtitle density
            if avg_duration < 2.0:  # Very dense dialogue
                actual_context_window = min(8, self.context_window * 2)
                logger.info(f"Dense dialogue detected (avg {avg_duration:.2f}s per subtitle). Increasing context window to {actual_context_window}")
            else:
                actual_context_window = self.context_window
        else:
            actual_context_window = self.context_window
        
        for i, subtitle in enumerate(subtitles):
            # Get context (previous and next subtitles)
            start_idx = max(0, i - actual_context_window // 2)
            end_idx = min(len(subtitles), i + (actual_context_window // 2) + 1)
            
            # Include the current subtitle in the context
            context = subtitles[start_idx:end_idx]
            
            # Check for scene breaks by analyzing timing gaps
            if i > 0 and i < len(subtitles) - 1:
                prev_gap = (subtitle.start - subtitles[i-1].end).total_seconds()
                next_gap = (subtitles[i+1].start - subtitle.end).total_seconds()
                
                # If there's a significant gap (potential scene change), log it
                if prev_gap > 5.0 or next_gap > 5.0:
                    logger.info(f"Potential scene break detected at subtitle #{subtitle.index} (gaps: prev={prev_gap:.2f}s, next={next_gap:.2f}s)")
            
            # Enhance subtitle with context
            enhanced_subtitle = self.enhance_subtitle(subtitle, context)
            enhanced_subtitles.append(enhanced_subtitle)
            
            # Update progress
            with progress_lock:
                completed += 1
                # Update progress callback if provided
                if progress_callback:
                    progress_callback(completed, total)
                
                # Log progress periodically
                if completed % 5 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    per_subtitle = elapsed / completed if completed > 0 else 0
                    logger.info(f"Enhanced {completed}/{total} subtitles {per_subtitle:.1f}s")
        
        return enhanced_subtitles
    
    def get_options(self) -> Dict[str, Any]:
        """
        Get the provider-specific options.
        
        Returns:
            Dictionary of provider options
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window
        }