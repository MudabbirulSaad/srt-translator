#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Ollama integration with SRT Translator.

This script tests the connection to Ollama, lists available models,
and tests text generation with a specified model.
"""

import sys
import logging
from typing import List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ollama_test")

def check_ollama_installed() -> bool:
    """Check if Ollama Python package is installed."""
    try:
        import ollama
        logger.info("✓ Ollama Python package is installed")
        return True
    except ImportError:
        logger.error("✗ Ollama Python package is not installed")
        logger.info("Install it with: pip install ollama")
        return False

def check_ollama_running(host: Optional[str] = None) -> bool:
    """Check if Ollama server is running."""
    try:
        import httpx
        
        # Construct the API URL based on the host
        base_url = host or "http://127.0.0.1:11434"
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        
        # Ensure the URL ends with the correct API path
        api_url = f"{base_url.rstrip('/')}/api/tags"
        
        # Make the request
        logger.info(f"Testing connection to Ollama API at: {api_url}")
        response = httpx.get(api_url, timeout=10.0)
        response.raise_for_status()
        
        logger.info(f"✓ Connected to Ollama server at {host or 'default endpoint'}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to Ollama server: {e}")
        logger.info("Make sure Ollama is running with: ollama serve")
        return False

def list_available_models(host: Optional[str] = None) -> List[str]:
    """List available Ollama models."""
    try:
        import httpx
        
        # Construct the API URL based on the host
        base_url = host or "http://127.0.0.1:11434"
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        
        # Ensure the URL ends with the correct API path
        api_url = f"{base_url.rstrip('/')}/api/tags"
        
        # Make the request
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
        
        logger.info(f"Available models: {', '.join(model_names) if model_names else 'None found'}")
        return model_names
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []

def test_ollama_generation(model: str, host: Optional[str] = None) -> bool:
    """Test Ollama text generation with JSON format."""
    try:
        import ollama
        client = ollama.Client(host=host)
        
        # Simple test prompt with JSON format instructions
        prompt = """Translate this to Spanish: Hello, how are you?

RESPOND USING ONLY THE FOLLOWING JSON FORMAT:
{
  "translation": "your translation here"
}

IMPORTANT: Provide ONLY the translation in the JSON response. Do not include any explanations or notes."""
        
        logger.info(f"Testing generation with model: {model}")
        logger.info(f"Prompt: {prompt}")
        
        try:
            # Try with JSON format parameter if available
            response = client.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0.7,
                },
                stream=False
            )
        except TypeError:
            # Fall back to standard generate if format parameter is not supported
            logger.warning("JSON format parameter not supported in this Ollama version, using standard generate")
            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                }
            )
        
        # Extract the response text
        if isinstance(response, dict) and "response" in response:
            response_text = response["response"].strip()
        elif hasattr(response, "response"):
            response_text = response.response.strip()
        else:
            logger.error(f"Unexpected response format: {type(response)}")
            return False
        
        logger.info(f"Raw response: {response_text}")
        
        # Try to parse as JSON
        try:
            parsed_json = json.loads(response_text)
            if isinstance(parsed_json, dict) and "translation" in parsed_json:
                translation = parsed_json["translation"].strip()
                logger.info(f"Parsed JSON response: {json.dumps(parsed_json, indent=2)}")
                logger.info(f"Translation: {translation}")
                logger.info("✓ JSON parsing successful")
            else:
                logger.warning(f"JSON response missing 'translation' field: {response_text}")
                # Try to extract any text field
                for key, value in parsed_json.items():
                    if isinstance(value, str):
                        logger.info(f"Found text field '{key}': {value}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.info(f"Using raw response as translation: {response_text}")
        
        logger.info("✓ Generation test completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to test generation: {e}")
        return False

def main() -> int:
    """Main entry point for the test script."""
    logger.info("=== Ollama Integration Test ===")
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        return 1
    
    # Check if Ollama is running
    if not check_ollama_running():
        return 1
    
    # List available models
    available_models = list_available_models()
    if not available_models:
        logger.error("No models available. Please pull at least one model.")
        logger.info("Example: ollama pull llama3")
        return 1
    
    # Test model specified by user or use first available
    model_to_test = None
    if len(sys.argv) > 1:
        model_to_test = sys.argv[1]
        if model_to_test not in available_models:
            logger.warning(f"Model '{model_to_test}' not found in available models")
            logger.info(f"You can pull it with: ollama pull {model_to_test}")
            
            # Ask if user wants to continue with the test anyway
            try:
                response = input(f"Model '{model_to_test}' not found. Test it anyway? (y/n): ")
                if response.lower() != 'y':
                    return 1
            except:
                # In non-interactive environments, just continue
                pass
    else:
        model_to_test = available_models[0]
        logger.info(f"Using first available model: {model_to_test}")
    
    # Test generation
    if not test_ollama_generation(model_to_test):
        return 1
    
    logger.info("=== All tests passed! ===")
    logger.info(f"You can now use the SRT Translator with Ollama enhancement:")
    logger.info(f"srt-translator input.srt -t es --enhance --enhance-model {model_to_test}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())