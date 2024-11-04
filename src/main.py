import torch
import uuid
from pathlib import Path
from diffusers import StableDiffusionPipeline

def main():
  cuda_available, device =  check_cuda()
  if cuda_available and device:
    calculate_tensor(device)
    model_dir, output_dir = check_needed_folders()
    generate_image(device, model_dir, output_dir)
  else:
      print('CUDA not available')



def check_cuda():
    print(torch.__version__)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. Running on GPU.")
        device = torch.device("cuda:0")
        return (cuda_available, device)
    else:
        return (False, None)

def calculate_tensor(device):
    # Perform a simple tensor operation
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y

    print(f"Result of tensor addition: {z}")


def check_needed_folders():
    model_dir = Path("/app/models")
    output_dir = Path("/app/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, output_dir


def generate_image(device, cache_path, output_path):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=cache_path,
        torch_dtype=torch.float16
    ).to(device)
    pipe.enable_attention_slicing()


    # Define the prompt
    prompt = "A futuristic cityscape at sunset"

    # Generate the image
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    # Save the image
    output_path = output_path / f"generated_image_{uuid.uuid4()}.png"
    image.save(output_path)
    print(f"Image saved to {output_path}")



if __name__ == "__main__":
    main()
