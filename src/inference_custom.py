"""Custom inference script for Splatent model."""
import os
import imageio
import argparse
import numpy as np
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm

from model import Splatent


class LatentLoader:
    """Utility class for loading latents or encoding images."""
    
    @staticmethod
    def load(path, vae):
        """Load latent from file or encode image to latent."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.pt', '.pth']:
            return torch.load(path).cuda()
        elif ext == '.npy':
            return torch.from_numpy(np.load(path)).cuda()
        else:
            return LatentLoader._encode_image(path, vae)
    
    @staticmethod
    def _encode_image(path, vae):
        """Encode image to latent representation."""
        img = Image.open(path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            return vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
    
    @staticmethod
    def normalize_dims(latent):
        """Ensure latent has shape (B, V, C, H, W)."""
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        if latent.dim() == 4:
            latent = latent.unsqueeze(1)
        return latent


def decode_to_image(vae, latent):
    """Decode latent to PIL image."""
    img = (vae.decode(latent / vae.config.scaling_factor).sample.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
    return Image.fromarray((img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def get_file_paths(path):
    """Get sorted list of file paths from directory or single file."""
    return sorted(glob(os.path.join(path, "*"))) if os.path.isdir(path) else [path]


def save_outputs(output_images, output_dir, input_paths, as_video=False):
    """Save outputs as video or individual images."""
    if as_video:
        video_path = os.path.join(output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for img in tqdm(output_images, desc="Saving video"):
            writer.append_data(np.array(img))
        writer.close()
        print(f"Video saved to {video_path}")
    else:
        for i, img in enumerate(tqdm(output_images, desc="Saving images")):
            output_name = os.path.splitext(os.path.basename(input_paths[i]))[0] + '.png'
            img.save(os.path.join(output_dir, output_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image/feature or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to reference image/feature or directory')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='Save as video')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    model = Splatent(pretrained_path=args.model_path, timestep=args.timestep)
    model.set_eval()
    model.to('cuda')
    vae = model.vae
    
    input_paths = get_file_paths(args.input_image)
    ref_paths = None
    single_ref = False
    
    if args.ref_image:
        ref_paths = get_file_paths(args.ref_image)
        single_ref = len(ref_paths) == 1
        if not single_ref and len(input_paths) != len(ref_paths):
            raise ValueError(f"Mismatch: {len(input_paths)} inputs vs {len(ref_paths)} references")
    
    prompt_tokens = model.tokenizer(
        args.prompt,
        max_length=model.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.cuda()
    
    print(f"Processing {len(input_paths)} images...")
    
    loader = LatentLoader()
    output_images = []
    
    for i, input_path in enumerate(tqdm(input_paths, desc="Processing")):
        x_src = loader.normalize_dims(loader.load(input_path, vae))
        
        if ref_paths:
            ref_idx = 0 if single_ref else i
            x_ref = loader.normalize_dims(loader.load(ref_paths[ref_idx], vae))
            x_src = torch.cat([x_src, x_ref], dim=1)
        
        with torch.no_grad():
            x_pred = model(x_src, prompt_tokens=prompt_tokens)[:, 0]
            output_images.append(decode_to_image(vae, x_pred))
    
    save_outputs(output_images, args.output_dir, input_paths, args.video)