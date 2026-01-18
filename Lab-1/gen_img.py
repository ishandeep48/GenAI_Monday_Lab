from diffusers import StableDiffusionPipeline
import torch
import os

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Test to reduce VRAM usage
pipe.enable_attention_slicing()

# Dataset config
output_dir = "synthetic_dog_dataset"
os.makedirs(output_dir, exist_ok=True)

dog_breeds = [
    "golden retriever",
    "german shepherd",
    "labrador retriever",
    "bulldog",
    "beagle",
    "poodle",
    "rottweiler",
    "yorkshire terrier",
    "boxer",
    "dachshund",
    "siberian husky",
    "great dane",
    "doberman pinscher",
    "shih tzu",
    "australian shepherd",
    "border collie",
    "cocker spaniel",
    "chihuahua",
    "pomeranian",
    "bernese mountain dog"
]


images_per_breed = 20

# usign low res cause my gpu takes too much time for 512x512
# low res didnt worked :'(
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Generate images
for breed in dog_breeds:
    breed_dir = os.path.join(output_dir, breed.replace(" ", "_"))
    os.makedirs(breed_dir, exist_ok=True)

    for i in range(images_per_breed):
        # prompt = f"A realistic photo of a {breed} dog" this was makign weird pics
        prompt = f"a high quality photo of a {breed} dog which his hyperrealistic and have sunglight on its face for best illumination"

        image = pipe(
            prompt,
            height=IMG_HEIGHT,
            width=IMG_WIDTH,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        image.save(f"{breed_dir}/img_{i+1}.png")
        print(f"Generated {breed} image {i+1}")

print("Low-resolution synthetic dog dataset generated!")
