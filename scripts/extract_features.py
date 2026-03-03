from diffusers import AutoencoderKL
import numpy as np
import torch
import cv2
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Get VAE embeddings of an input image or directory of images."
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to scene directory (containing images_4/ folder).",
)

parser.add_argument(
    "--output",
    type=str,
    default="features_sd_turbo",
    help="Output directory name for features (default: features_sd_turbo).",
)

parser.add_argument(
    "--res",
    type=str,
    default="images_4",
    help="Input images directory name (default: images_4).",
)

parser.add_argument(
    "--device", 
    type=str, 
    default="cuda", 
    help="The device to run generation on."
)

def main(args: argparse.Namespace) -> None:
    print("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.to(device=args.device)
    vae.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Construct paths
    images_dir = os.path.join(args.input, args.res)
    output_dir = os.path.join(args.input, args.output)
    
    if not os.path.exists(images_dir):
        print(f"Error: {args.res}/ directory not found at {images_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.JPG', '.jpeg'))]
    print(f"Found {len(image_files)} images in {images_dir}")
    
    for img_file in tqdm(image_files, desc="Extracting features"):
        img_path = os.path.join(images_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Resize if dimensions not divisible by 8
            w, h = image.size
            new_w = (w // 8) * 8
            new_h = (h // 8) * 8
            if w != new_w or h != new_h:
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Preprocess
            image_tensor = transform(image).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                # Encode to latent space
                latent = vae.encode(image_tensor).latent_dist.sample()
                
            # Save embedding
            torch.save(latent.squeeze(0).cpu(), os.path.join(output_dir, f"{img_name}.pt"))
            
        except Exception as e:
            print(f"Could not process '{img_path}': {e}, skipping...")
            continue
    
    print(f"✓ Extracted {len(image_files)} features to {output_dir}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)