"""Evaluation metrics for image quality assessment."""
import os
import json
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser


class ImageMetrics:
    """Collection of image quality metrics."""
    
    @staticmethod
    def psnr(img1, img2):
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    @staticmethod
    def ssim(img1, img2):
        """Calculate Structural Similarity Index."""
        from pytorch_msssim import ssim as ssim_func
        return ssim_func(img1, img2, data_range=1.0, size_average=True)
    
    @staticmethod
    def lpips(img1, img2, lpips_fn):
        """Calculate Learned Perceptual Image Patch Similarity."""
        return lpips_fn(img1 * 2 - 1, img2 * 2 - 1).item()


class ImageLoader:
    """Utility for loading and preprocessing images."""
    
    @staticmethod
    def read_images(renders_dir, gt_dir):
        """Read rendered and ground truth images from directories."""
        renders, gts, image_names = [], [], []
        for fname in sorted(os.listdir(renders_dir)):
            if fname.startswith(".") or not fname.endswith('.png'):
                continue
            render = Image.open(os.path.join(renders_dir, fname))
            gt = Image.open(os.path.join(gt_dir, fname))
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
        return renders, gts, image_names


class Evaluator:
    """Evaluation pipeline for image quality metrics."""
    
    def __init__(self, output_dir, method_name):
        self.output_dir = output_dir
        self.method_name = method_name
        self.method_dir = os.path.join(output_dir, method_name)
        self.metrics = ImageMetrics()
        self.loader = ImageLoader()
        
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').cuda()
    
    def _validate_directories(self):
        """Check if required directories exist."""
        required_dirs = ["gt_images", "render_latent_reconstruction", "input_images"]
        for dir_name in required_dirs:
            if not os.path.exists(os.path.join(self.method_dir, dir_name)):
                print(f"Error: Missing directory {dir_name} in {self.method_dir}")
                return False
        return True
    
    def _compute_metrics(self, renders, gts, inputs):
        """Compute metrics for all images."""
        ssims_out, psnrs_out, lpipss_out = [], [], []
        ssims_in, psnrs_in, lpipss_in = [], [], []
        
        for idx in tqdm(range(len(renders)), desc="Computing metrics"):
            ssims_out.append(self.metrics.ssim(renders[idx], gts[idx]).item())
            psnrs_out.append(self.metrics.psnr(renders[idx], gts[idx]).item())
            lpipss_out.append(self.metrics.lpips(renders[idx], gts[idx], self.lpips_fn))
            
            ssims_in.append(self.metrics.ssim(inputs[idx], gts[idx]).item())
            psnrs_in.append(self.metrics.psnr(inputs[idx], gts[idx]).item())
            lpipss_in.append(self.metrics.lpips(inputs[idx], gts[idx], self.lpips_fn))
        
        return {
            'output': {'ssim': ssims_out, 'psnr': psnrs_out, 'lpips': lpipss_out},
            'input': {'ssim': ssims_in, 'psnr': psnrs_in, 'lpips': lpipss_in}
        }
    
    def _print_results(self, metrics):
        """Print evaluation results."""
        print(f"\n{self.method_name} Results:")
        for category, label in [('output', 'Output vs GT'), ('input', 'Input vs GT')]:
            print(f"\n{label}:")
            print(f"  SSIM : {torch.tensor(metrics[category]['ssim']).mean():.7f}")
            print(f"  PSNR : {torch.tensor(metrics[category]['psnr']).mean():.7f}")
            print(f"  LPIPS: {torch.tensor(metrics[category]['lpips']).mean():.7f}")
        print()
    
    def _save_results(self, metrics, image_names):
        """Save results to JSON files."""
        full_dict = {self.method_name: {}}
        per_view_dict = {self.method_name: {}}
        
        for category, label in [('output', 'output_vs_gt'), ('input', 'input_vs_gt')]:
            full_dict[self.method_name][label] = {
                "SSIM": torch.tensor(metrics[category]['ssim']).mean().item(),
                "PSNR": torch.tensor(metrics[category]['psnr']).mean().item(),
                "LPIPS": torch.tensor(metrics[category]['lpips']).mean().item()
            }
            per_view_dict[self.method_name][label] = {
                "SSIM": {name: s for name, s in zip(image_names, metrics[category]['ssim'])},
                "PSNR": {name: p for name, p in zip(image_names, metrics[category]['psnr'])},
                "LPIPS": {name: l for name, l in zip(image_names, metrics[category]['lpips'])}
            }
        
        with open(os.path.join(self.output_dir, "results.json"), 'w') as fp:
            json.dump(full_dict, fp, indent=2)
        with open(os.path.join(self.output_dir, "per_view.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        if not self._validate_directories():
            return
        
        print(f"Evaluating {self.method_name}...")
        
        gt_dir = os.path.join(self.method_dir, "gt_images")
        renders_dir = os.path.join(self.method_dir, "render_latent_reconstruction")
        input_dir = os.path.join(self.method_dir, "input_images")
        
        renders, gts, image_names = self.loader.read_images(renders_dir, gt_dir)
        inputs, _, _ = self.loader.read_images(input_dir, gt_dir)
        
        metrics = self._compute_metrics(renders, gts, inputs)
        self._print_results(metrics)
        self._save_results(metrics, image_names)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    parser = ArgumentParser(description="Evaluation script")
    parser.add_argument('--output_dir', required=True, type=str, help="Directory containing method subdirectories")
    parser.add_argument('--method_name', default="splatent", type=str, help="Method name to evaluate")
    args = parser.parse_args()
    
    evaluator = Evaluator(args.output_dir, args.method_name)
    evaluator.evaluate()
