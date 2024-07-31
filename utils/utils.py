import os
import re
import cv2
import json
import random
from PIL import Image
from datetime import datetime

current_date_time = datetime.now()
current_year = current_date_time.year

RGN_SEED = random.randint(0, 999999)
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
OBJECT_DETECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "detect_object",
        "description": "Detect object in the given image",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "path to the image"
                }
            },
            "required": [
                "image_path"
            ]
        }
    }
}
FUNCTIONS_METADATA_MASTER = [HRM_TOOL, CREATIVE_TOOL]

TOOL_DICT_PARAMS = {
    "generate_image": ["prompt"],
    "process_absence_request": [
        "start_time",
        "end_time",
        "manager_name",
        "alt_employee_name",
        "address"],
    "detect_object": ["image_path"]}

SYSTEM_PROMPT = f"""
You are a virtual cute assistant named Claire developed by TonAI Lab. You can access to the following functions:
{str(FUNCTIONS_METADATA_MASTER)}\n\nTo use these functions respond with:
<functioncall> {{ "name": "function_name", "arguments": {{ "arg_1": "value_1", "arg_1": "value_1", ... }} }} </functioncall>
Edge cases you must handle:
- If there are no functions that match the user request, you will try to understand the question and respond user's question directly.
- If no clear time information, today is {current_date_time.strftime("%d/%m/%y")} (dd/mm.yy). 
- Convert datetime to dd/mm/yyyy by yourself
"""

HRM_CHECKER_SYSTEM_PROMPT = f"""
You are the virtual assistant with access to the following functions: f{HRM_TOOL}\n
The functions must have the following required parameters: {str(TOOL_DICT_PARAMS["process_absence_request"])}\n
Current year is {current_year}.
If you don't know current datetime, now is {current_date_time.strftime("%d/%m/%y")} (dd/mm/yyyy)
If the user don't provid correct datetime format, convert it to (dd/mm/yyyy) by yourself
Check the user input, if there are not enough information for required parameters, politely ask the user about missed information.
If user provided enough information, respond "<accepted>" only
"""

SD_CHECKER_SYSTEM_PROMPT = f"""
The user want you using Stable Diffusion to generate image.
You must ask the user confirm what did he/she told you to generate.
If the user agree, return "<accepted>"
If the user disagree, politely say that you can help them other tasks
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


def draw_bbox(image_path, bboxes, label):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Convert relative bbox coordinates to absolute coordinates
    for bbox in bboxes:
        x_min = int(bbox[0] * width)
        y_min = int(bbox[1] * height)
        x_max = int(bbox[2] * width)
        y_max = int(bbox[3] * height)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 255)  # Red color
        background_color = (255, 255, 255)  # White color
        label_size, _ = cv2.getTextSize(
            label, font, font_scale, font_thickness)
        label_x, label_y = x_min, y_min - 10

        cv2.rectangle(
            image,
            (label_x,
             label_y -
             label_size[1]),
            (label_x +
             label_size[0],
             label_y),
            background_color,
            cv2.FILLED)
        # Put the label text on the image
        cv2.putText(image, label, (label_x, label_y), font,
                    font_scale, text_color, font_thickness)

    cv2_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)

    return pil_image
