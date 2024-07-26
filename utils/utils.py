import os
import re
import ast
import json
from datetime import datetime

current_date_time = datetime.now()
current_year = current_date_time.year


HRM_TOOL = {
    "type": "function",
    "function": {
        "name": "process_absence_request",
        "description": "process absence request of employees",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "absence start time"
                },
                "end_time": {
                    "type": "string",
                    "description": "absence end time"
                },
                "manager_name": {
                    "type": "string",
                    "description": "name of the manager"
                },
                "alt_employee_name": {
                    "type": "string",
                    "description": "name of the employee will take charge during your absence"
                },
                "address": {
                    "type": "string",
                    "description": "Address for absence"
                }
            },
            "required": [
                "start_time", "end_time", "manager_name", "alt_employee_name"
            ]
        }
    }
}
CREATIVE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "generate image with Stable Diffusion",
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
}
FUNCTIONS_METADATA_MASTER = [HRM_TOOL, CREATIVE_TOOL]

TOOL_DICT_PARAMS = {
    "generate_image": ["prompt"],
    "process_absence_request": ["start_time", "end_time", "manager_name", "alt_employee_name", "address"]
}

SYSTEM_PROMPT = f"""
You are the virtual assistant of TonAI with access to the following functions:
{str(FUNCTIONS_METADATA_MASTER)}\n\nTo use these functions respond with:
<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall>
Edge cases you must handle:
- If no clear time information, today is {current_date_time.strftime("%d/%m/%y")} (dd/mm.yy)
- If there are no functions that match the user request, you will try to understand the question and respond user's question directly.
"""

HRM_CHECKER_SYSTEM_PROMPT = f"""
You are the virtual assistant with access to the following functions: f{HRM_TOOL}\n
The functions must have the following parameters: {str(TOOL_DICT_PARAMS["process_absence_request"])}\n
Now is {current_date_time.strftime("%d/%m/%y")} (dd/mm/yy)
Check the user input, if there are not enough required parameters, politely ask the user to provide more information.
If user provided enough parameters, respond '<accepted>'
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


def find_dict_in_string(input_string):
    input_string = remove_markdown_code_blocks(input_string)
    # Regular expression to match the dictionary within the string
    dict_regex = r'\{\s*\'start_time\':\s*\'\d{2}/\d{2}/\d{4}\',\s*\'end_time\':\s*\'\d{2}/\d{2}/\d{4}\',\s*\'remaining time\':\s*\d+\s*\}'

    # Search for the dictionary pattern in the input string
    match = re.search(dict_regex, input_string, re.DOTALL)

    if match:
        return match.group(0)
    else:
        return ""
    

def remove_markdown_code_blocks(input_string):
    # Regular expression to match Markdown code blocks (triple backticks)
    code_block_pattern = r'```[\s\S]*?```'

    # Remove all matches of the code block pattern from the input string
    cleaned_string = re.sub(code_block_pattern, '', input_string)

    return cleaned_string.strip()
