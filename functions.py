import torch
from diffusers import StableDiffusionPipeline


def book_taxi(start_location: str = "",
              destination: str = ""):
    print(
        f"I haved booked a car from {start_location} to {destination}, wait for few minutes!")


def generate_image(prompt: str = ""):
    device = torch.device(f"cuda:0")
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
                         guidance_scale=7).images[0]
        image.save("generated.png")
    except BaseException:
        print("Server is overload")
