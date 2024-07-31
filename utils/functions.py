import gc
import random
import torch
from PIL import Image
from datetime import datetime

from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils.utils import *
from utils.assistant import VirtualAssistant
from assistant_info import assistant

response_types = ["text", "image", "audio", "video"]


def detect_object(image_path: str = "", **kwargs):
    """
    Object detection using Vision Language Model
    """
    try:
        image = Image.open(image_path)
        model_id = "microsoft/kosmos-2-patch14-224"
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto")
        processor = AutoProcessor.from_pretrained(model_id)
        prompt = "<grounding>An image of"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt").to(
            model.device)
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        processed_text, entities = processor.post_process_generation(
            generated_text)
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()
        for entity in entities:
            processed_image = draw_bbox(image_path, entity[2], entity[0])
        return processed_image, image_path, processed_text, response_types[1]
    except BaseException:
        message = "I can't find the image"
        return message, response_types[0]


def process_absence_request(start_time: str = "",
                            end_time: str = "",
                            manager_name: str = "",
                            alt_employee_name: str = "",
                            address: str = "",
                            **kwargs):
    """
    Process absence request of employees
    """
    print("process_absence_request is being called")
    current_date_time = datetime.now()
    current_year = current_date_time.year

    # model_id = "hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # hrm_assistant = VirtualAssistant(llm_model_id=model_id,
    #                                  llm_quantization=True,
    #                                  llm_use_bitsandbytes=True)
    hrm_assistant = assistant
    instruction = f"""
    You will fill the form of absence request from an employee.
    Try to convert the time to format (dd/mm/yy)
    If no clear start time information, today is {current_date_time.strftime("%d/%m/%y")} (dd/mm/yy)
    manager name is {manager_name}
    person who will take charge your work is {alt_employee_name}
    If no year infomation, current year is {current_year}
    Address where the employee will be is {address}
    Don't show the detailed solution, only return the a json dictionary with format: {{
        'start_time': 'start_time in dd/mm/yy',
        'end_time': 'end_time in dd/mm/yy',
        'manager_name': {manager_name},
        'alt_employee_name': {alt_employee_name},
        'address': where you will be during absence
    }}
    """
    messages = [
        {"role": "system", "content": instruction},
    ]
    # prompt = f"Currently, an employee has {remaining_day} available days for absence, and will be absent from {start_time} to {end_time}"
    prompt = f"Process the absence request of an employee from {start_time} to {end_time}"
    messages.append({"role": "user", "content": prompt})
    answer = hrm_assistant.complete(messages)
    # hrm_assistant.release_gpu_memory()
    # del hrm_assistant
    dict_info = find_dict_in_string(answer)
    if len(str(dict_info)) > 0:
        answer = str(dict_info)
    return answer, response_types[0]


def generate_image(prompt: str = "", **kwargs):
    seed = RGN_SEED
    device = torch.device(f"cuda:0")
    generator = torch.Generator(device).manual_seed(seed)
    if len(prompt) == 0:
        prompt = "A random image"
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
        image_info = prompt
        return image, image_path, image_info, response_types[1]
    except BaseException:
        message = "Server is overload, I can't draw picture now"
        return message, response_types[0]
