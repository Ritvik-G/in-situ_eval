
import requests
from typing import Dict, Any
import json


# LLM API Call
def generate_response(model_config: Dict[str, Any], prompt: str) -> str:
    """
    Generates a response using the Groq API based on the provided model configuration and prompt.

    Args:
        model_config (Dict[str, Any]): Configuration for the model, loaded from a JSON file.
                                       Includes keys like 'api_key', 'model', 'temperature', etc.
        prompt (str): The input prompt to generate a response for.

    Returns:
        str: The generated response or an error message if the request fails.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_config['api_key']}",
    }
    
    data = {
        "model": model_config['model'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(model_config['temperature']),
        "max_tokens": int(model_config['max_tokens']),
        "top_p":float(model_config['top_p']),
        "stream": model_config['stream'],
        "stop": model_config['stop'],
    }
    
    try:
        with open ("config.json","r") as c:
            access_configurator = json.load(c)
        access_config = access_configurator["ACCESS_CONFIG"]

        print("Type of top P is - ",type(model_config['top_p']))
        print(data)
        
        response = requests.post(access_config[model_config['type']], headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
