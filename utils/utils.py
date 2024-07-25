import os
import re
import json

FUNCTIONS_METADATA = [
    {
        "type": "function",
        "function": {
            "name": "book_taxi",
            "description": "book a taxi",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_location": {
                        "type": "string",
                        "description": "location for picking up"
                    },
                    "destination": {
                        "type": "string",
                        "description": "destination"
                    },
                },
                "required": [
                    "start_point", "end_point"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "generate image using Stable Diffusion",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "tell the AI what do you want to generate"
                    }
                },
                "required": [
                    "prompt"
                ]
            }
        }
    },
]

SYSTEM_PROMPT = f"""You are the virtual assistant of TonAI with access to the following functions:
{str(FUNCTIONS_METADATA)}\n\nTo use these functions respond with:
<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall>
Edge cases you must handle:
- If there are no functions that match the user request, you will respond user's question directly.
"""


def get_function_info(answer: str = ""):
    match = re.search(r'<functioncall>(.*?)</functioncall>', answer)
    if match:
        functioncall_string = match.group(1).strip()
        functioncall_string = functioncall_string.replace("'", '')
        dictionary = json.loads(functioncall_string)
        return dictionary
    else:
        return


def save_image(image, base_dir="stuffs", base_filename="generated.png"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    base_path = os.path.join(base_dir, base_filename)
    if not os.path.exists(base_path):
        image.save(base_path)
        return base_path
    else:
        base, ext = os.path.splitext(base_filename)
        counter = 1
        new_path = os.path.join(base_dir, f"{base}{counter}{ext}")
        while os.path.exists(new_path):
            counter += 1
            new_path = os.path.join(base_dir, f"{base}{counter}{ext}")
        image.save(new_path)
        return new_path