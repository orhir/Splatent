"""Dataset-based inference script for Splatent model."""
import os
import json
import argparse
from math import exp
import torch
import torch.nn.functional as F
import torch.distributed as dist
import lpips
from tqdm import tqdm
from torchvision.utils import save_image

from model import Splatent
from dataset import PairedDataset


class MetricsCalculator:
    """Calculate image quality metrics."""
    
    @staticmethod
    def gaussian(window_size, sigma):
        """Generate Gaussian kernel."""
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    @staticmethod
    def create_window(window_size, channel):
        """Create 2D Gaussian window for SSIM."""
        _1D_window = MetricsCalculator.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    @staticmethod
    def psnr(img1, img2):
        """Calculate PSNR."""
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    @staticmethod
    def ssim(img1, img2, window_size=11):
        """Calculate SSIM."""
        channel = img1.size(-3)
        window = MetricsCalculator.create_window(window_size, channel).type_as(img1)
        
        padding = window_size // 2
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class DistributedManager:
    """Manage distributed training setup."""
    
    @staticmethod
    def setup():
        """Initialize distributed process group."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
        return local_rank, world_size
    
    @staticmethod
    def gather_results(world_size, local_rank, metrics_lists):
        """Gather results from all processes."""
        if world_size == 1:
            return metrics_lists
        
        gathered = {}
        for key, values in metrics_lists.items():
            all_values = [None] * world_size
            dist.all_gather_object(all_values, values)
            if local_rank == 0:
                gathered[key] = [item for sublist in all_values for item in sublist]
        return gathered if local_rank == 0 else {}
    
    @staticmethod
    def cleanup(world_size):
        """Clean up distributed process group."""
        if world_size > 1:
            dist.destroy_process_group()


class OutputSaver:
    """Handle saving of output images and latents."""
    
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
    
    def save(self, feature_id, pred_i, input_i, gt_i, x_pred_i, x_input_i, x_gt_i):
        """Save outputs based on configuration."""
        feature_path = self.dataset.data[feature_id]["target_feature"]
        scene_name = feature_path.split('/')[-3]
        frame_name = os.path.basename(feature_path).replace('.pt', '')
        
        if self.args.save_images:
            self._save_images(scene_name, frame_name, pred_i, input_i, gt_i)
        
        if self.args.save_latent:
            self._save_latents(scene_name, frame_name, x_pred_i, x_input_i, x_gt_i)
    
    def _save_images(self, scene_name, frame_name, pred_i, input_i, gt_i):
        """Save image outputs."""
        for img_type, img in [("input", input_i), ("output", pred_i), ("gt", gt_i)]:
            if img is None:
                continue
            img_dir = os.path.join(self.args.output_dir, "vis", scene_name, img_type)
            os.makedirs(img_dir, exist_ok=True)
            save_image(img, os.path.join(img_dir, f"{frame_name}.png"), padding=0)
    
    def _save_latents(self, scene_name, frame_name, x_pred_i, x_input_i, x_gt_i):
        """Save latent features."""
        for feat_type, feat in [("input", x_input_i), ("output", x_pred_i), ("gt", x_gt_i)]:
            if feat is None:
                continue
            feat_dir = os.path.join(self.args.output_dir, "features", scene_name, feat_type)
            os.makedirs(feat_dir, exist_ok=True)
            torch.save(feat.cpu(), os.path.join(feat_dir, f"{frame_name}.pt"))


class InferenceEngine:
    """Main inference engine for dataset evaluation."""
    
    def __init__(self, args):
        self.args = args
        self.local_rank, self.world_size = DistributedManager.setup()
        
        self.model = self._load_model()
        self.vae = self.model.vae
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.local_rank)
        self.metrics_calc = MetricsCalculator()
        
        self.dataloader, self.dataset = self._create_dataloader()
        self.output_saver = OutputSaver(args, self.dataset)
    
    def _load_model(self):
        """Load and prepare model."""
        if self.local_rank == 0:
            print(f"Loading model from {self.args.model_path}")
        
        model = Splatent(pretrained_path=self.args.model_path, timestep=self.args.timestep)
        model.set_eval()
        model.to(self.local_rank)
        return model
    
    def _create_dataloader(self):
        """Create dataloader with optional distributed sampling."""
        dataset = PairedDataset(
            dataset_path=self.args.dataset_path,
            split="test",
            tokenizer=self.model.tokenizer,
            top_k=self.args.top_k,
            min_k=self.args.min_k,
            scale=self.args.scale,
            rgb_root=self.args.rgb_root,
            no_metrics=self.args.no_metrics
        )
        
        if self.args.scene_id:
            dataset.data = {k: v for k, v in dataset.data.items() 
                          if self.args.scene_id in v.get("feature", "")}
            if not dataset.data:
                raise ValueError(f"Scene ID '{self.args.scene_id}' not found in dataset")
            dataset.feature_ids = list(dataset.data.keys())
        
        persistent = self.args.num_workers > 0
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            return torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, sampler=sampler,
                num_workers=self.args.num_workers, persistent_workers=persistent
            ), dataset
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.num_workers, persistent_workers=persistent
        ), dataset
    
    def _decode_latents(self, latents):
        """Decode latents to images."""
        return (self.vae.decode(latents / self.vae.config.scaling_factor).sample.clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
    
    def _compute_batch_metrics(self, pred_i, gt_i, input_i):
        """Compute metrics for a single batch item."""
        return {
            'ssim_out': self.metrics_calc.ssim(pred_i, gt_i).item(),
            'psnr_out': self.metrics_calc.psnr(pred_i, gt_i).squeeze().item(),
            'lpips_out': self.lpips_fn(pred_i * 2 - 1, gt_i * 2 - 1).item(),
            'ssim_in': self.metrics_calc.ssim(input_i, gt_i).item(),
            'psnr_in': self.metrics_calc.psnr(input_i, gt_i).squeeze().item(),
            'lpips_in': self.lpips_fn(input_i * 2 - 1, gt_i * 2 - 1).item()
        }
    
    def _process_batch(self, batch, metrics_lists):
        """Process a single batch."""
        x_src = batch["conditioning_pixel_values"].to(self.local_rank)
        x_tgt = batch["output_pixel_values"].to(self.local_rank)
        prompt_tokens = batch["input_ids"].to(self.local_rank)
        gt_img = batch["gt_image"].to(self.local_rank)
        feature_ids = batch["feature_id"]
        
        with torch.no_grad():
            x_pred = self.model(x_src, prompt_tokens=prompt_tokens)[:, 0]
            x_input = x_src[:, 0]
            
            pred_img = self._decode_latents(x_pred)
            input_img = self._decode_latents(x_input)
            
            gt_img = (gt_img * 0.5 + 0.5).clamp(0, 1)
            if gt_img.shape[2:] != pred_img.shape[2:]:
                gt_img = F.interpolate(gt_img, size=pred_img.shape[2:], mode='area')
            
            x_gt = self.vae.encode(gt_img * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor
            
            for i in range(pred_img.shape[0]):
                pred_i, input_i = pred_img[i:i+1], input_img[i:i+1]
                gt_i = gt_img[i:i+1] if not self.args.no_metrics else None
                
                if self.args.save_images or self.args.save_latent:
                    self.output_saver.save(
                        feature_ids[i], pred_i, input_i, gt_i,
                        x_pred[i], x_input[i], x_gt[i] if not self.args.no_metrics else None
                    )
                
                if not self.args.no_metrics:
                    batch_metrics = self._compute_batch_metrics(pred_i, gt_i, input_i)
                    for key, value in batch_metrics.items():
                        metrics_lists[key].append(value)
        
        return metrics_lists
    
    def _print_and_save_results(self, metrics):
        """Print and save final results."""
        results = {
            "output_vs_gt": {
                "SSIM": torch.tensor(metrics['ssims_out']).mean().item(),
                "PSNR": torch.tensor(metrics['psnrs_out']).mean().item(),
                "LPIPS": torch.tensor(metrics['lpipss_out']).mean().item()
            },
            "input_vs_gt": {
                "SSIM": torch.tensor(metrics['ssims_in']).mean().item(),
                "PSNR": torch.tensor(metrics['psnrs_in']).mean().item(),
                "LPIPS": torch.tensor(metrics['lpipss_in']).mean().item()
            }
        }
        
        print(f"\n{'='*50}")
        print(f"Results for {self.args.method_name}:")
        print(f"{'='*50}")
        for category in ["output_vs_gt", "input_vs_gt"]:
            print(f"\n{category.replace('_', ' ').title()}:")
            for metric, value in results[category].items():
                print(f"  {metric:6s}: {value:.7f}")
        print(f"\n{'='*50}\n")
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        results_path = os.path.join(self.args.output_dir, f"{self.args.method_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    def run(self):
        """Run inference on dataset."""
        if self.local_rank == 0:
            print(f"Evaluating {len(self.dataset)} test samples...")
        
        metrics_lists = {
            'ssims_out': [], 'psnrs_out': [], 'lpipss_out': [],
            'ssims_in': [], 'psnrs_in': [], 'lpipss_in': []
        }
        
        pbar = tqdm(self.dataloader, desc="Computing metrics") if self.local_rank == 0 else self.dataloader
        
        for batch in pbar:
            metrics_lists = self._process_batch(batch, metrics_lists)
            
            if self.local_rank == 0 and not self.args.no_metrics:
                pbar.set_postfix({
                    'SSIM': f"{torch.tensor(metrics_lists['ssims_out']).mean():.4f}",
                    'PSNR': f"{torch.tensor(metrics_lists['psnrs_out']).mean():.2f}",
                    'LPIPS': f"{torch.tensor(metrics_lists['lpipss_out']).mean():.4f}"
                })
        
        if not self.args.no_metrics:
            metrics_lists = DistributedManager.gather_results(
                self.world_size, self.local_rank, metrics_lists
            )
            
            if self.local_rank == 0:
                self._print_and_save_results(metrics_lists)
        
        DistributedManager.cleanup(self.world_size)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--method_name", default="splatent", type=str)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--top_k", default=3, type=int, help="Number of reference views to include")
    parser.add_argument("--min_k", default=0, type=int, help="Minimum reference image proximity")
    parser.add_argument("--scale", action="store_true", help="Apply scaling factor (0.18215)")
    parser.add_argument("--rgb_root", default=None, type=str, help="Root directory for RGB images")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--scene_id", default=None, type=str, help="Specific scene ID to evaluate")
    parser.add_argument("--save_images", action="store_true", help="Save predicted images")
    parser.add_argument("--save_latent", action="store_true", help="Save predicted latents")
    parser.add_argument("--no_metrics", action="store_true", help="Skip metric computation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = InferenceEngine(args)
    engine.run()
