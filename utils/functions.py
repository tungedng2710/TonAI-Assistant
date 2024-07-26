import os
import gc
import random
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline
from utils.utils import *
from utils.assistant import VirtualAssistant

response_types = ["text", "image", "audio", "video"]


def book_taxi(start_location: str = "",
              destination: str = "",
              **kwargs):
    """
    Dummy function
    """
    message = f"I haved booked a car from {start_location} to {destination}"
    message += "\nDriver will pick you soon"
    print(message)
    return message, response_types[0]


def process_absence_request(start_time: str = "",
                            end_time: str = "",
                            remaining_day: int = 12,
                            manager_name: str = "",
                            alt_employee_name: str = "",
                            **kwargs):
    current_date_time = datetime.now()
    current_year = current_date_time.year

    model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
    hrm_assistant = VirtualAssistant(llm_model_id=model_id,
                                     llm_quantization=True)
    instruction = f"""
    You will fill the form of absence request from an employee.
    if no clear start time information, today is {current_date_time.strftime("%d/%m/%y")} (dd/mm/yy)
    manager name is {manager_name}
    person who will take charge your work is {alt_employee_name}
    If no year infomation, current year is {current_year}
    Don't show the detailed solution, only return the a string dictionary with format: {{
        'start time': 'start_time in dd/mm/yy',
        'end time': 'end_time in dd/mm/yy',
        'manager name': 'name of manager',
        'alt employee name': alt_employee_name,
        'address': provided address
    }}
    """
    messages = [
        {"role": "system", "content": instruction},
    ]
    prompt = f"Currently, an employee has {remaining_day} available days for absence, and will be absent from {start_time} to {end_time}"
    messages.append({"role": "user", "content": prompt})
    answer = hrm_assistant.complete(messages)
    hrm_assistant.release_gpu_memory()
    torch.cuda.empty_cache()
    gc.collect()
    del hrm_assistant
    dict_info = find_dict_in_string(answer)
    if len(str(dict_info)) > 0:
        answer = str(dict_info)
    return answer, response_types[0]


def generate_image(prompt: str = "", **kwargs):
    seed = random.randint(0, 999999)
    device = torch.device(f"cuda:0")
    generator = torch.Generator(device).manual_seed(seed)
    if len(prompt) == 0:
        prompt = input(
            "TonAI Assistant: Tell me what do you want to generate: ")
    model_path = "../checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
    try:
        StableDiffusionPipeline.safety_checker = None
        pipeline = StableDiffusionPipeline.from_single_file(
            model_path).to(device)
        # pipeline.enable_sequential_cpu_offload()
        image = pipeline(prompt=prompt,
                         negative_prompt="ugly, blurred",
                         width=512,
                         height=512,
                         num_inference_steps=20,
                         generator=generator,
                         guidance_scale=2).images[0]
        image_path = save_image(image)
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        return image, image_path, response_types[1]
    except BaseException:
        message = "Server is overload, I can't draw picture now"
        return message, response_types[0]
