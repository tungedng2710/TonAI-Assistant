import json
import torch
from diffusers import StableDiffusionPipeline


def book_taxi(start_location: str = "",
              destination: str = ""):
    print(
        f"I haved booked a car from {start_location} to {destination}, wait for few minutes!")


def generate_image(prompt: str = ""):
    device = torch.device(f"cuda:0")
    if len(prompt) == 0:
        prompt = input("TonAI Assistant: Tell me what do you want to generate: ")
    model_path = "../checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
    try:
        pipeline = StableDiffusionPipeline.from_single_file(model_path).to(device)
        image = pipeline(prompt=prompt,
                        negative_prompt="ugly, blurred",
                        width=512,
                        height=512,
                        num_inference_steps=20,
                        guidance_scale=7).images[0]
        image.save("generated.png")
    except:
        print("Server is overload")


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
