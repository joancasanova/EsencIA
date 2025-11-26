# src/utils/json_utils.py

import json
import re
from typing import Optional, Any

def extract_json_from_response(text: str) -> Optional[Any]:
    """
    Extracts JSON object from a given text response.
    Handles direct JSON, JSON within Markdown code blocks, and raw JSON structures.
    
    Args:
        text: Input text potentially containing JSON data.
    
    Returns:
        Parsed JSON object if extraction and parsing succeed, else None.
    """
    if not text:
        return None

    # 1. Direct attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Search for JSON within Markdown code blocks (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Search for raw JSON structure ({ ... }) by looking for the first open brace and the last close brace
    try:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
        
    return None