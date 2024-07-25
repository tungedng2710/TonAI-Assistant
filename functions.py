import os
import random
import torch
from diffusers import StableDiffusionPipeline
from utils.utils import save_image

response_types = ["text", "image", "audio", "video"]


def book_taxi(start_location: str = "",
              destination: str = ""):
    """
    Dummy function
    """
    message = f"I haved booked a car from {start_location} to {destination}"
    message += "\nDriver will pick you soon"
    print(message)
    return message, response_types[0]


def book_flight_ticket(start_location: str = "",
                       destination: str = "",
                       date_time: str = ""):
    """
    Dummy function
    """
    message = f"Your flight info: {start_location} to {destination}"
    message += "\nDriver will pick you soon"
    print(message)
    return message, response_types[0]


def generate_image(prompt: str = ""):
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
        image = pipeline(prompt=prompt,
                         negative_prompt="ugly, blurred",
                         width=512,
                         height=512,
                         num_inference_steps=20,
                         generator=generator,
                         guidance_scale=2).images[0]
        image_path = save_image(image)
        return image, image_path, response_types[1]
    except BaseException:
        message = "Server is overload, I can't draw picture now"
        return message, response_types[0]
