"""Training script for Splatent model."""
import os
import gc
import lpips
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange
from sklearn.decomposition import PCA

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

try:
    from allegroai import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("[WARNING] ClearML not available. Install with: pip install clearml")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not available. Install with: pip install tensorboard")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='clearml')
import logging

from model import Splatent, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from inference_dataset import MetricsCalculator


class VisualizationUtils:
    """Utilities for visualization during training."""
    
    @staticmethod
    def make_2x2_grid(top_left, top_right, bottom_left, bottom_right):
        """Create 2x2 grid from 4 images (HWC format)."""
        top = np.concatenate([top_left, top_right], axis=1)
        bottom = np.concatenate([bottom_left, bottom_right], axis=1)
        return np.concatenate([top, bottom], axis=0)
    
    @staticmethod
    def features_grid_to_pca_rgb(feat_list):
        """Convert list of 4 feature maps to RGB using shared PCA.
        
        Args:
            feat_list: List of 4 tensors with shape (C, H, W)
        
        Returns:
            List of 4 numpy arrays with shape (H, W, 3)
        """
        all_features = []
        shapes = []
        for feat in feat_list:
            C, H, W = feat.shape
            shapes.append((H, W))
            all_features.append(feat.reshape(C, -1).T.float().cpu().numpy())
        
        all_features_concat = np.concatenate(all_features, axis=0)
        
        pca = PCA(n_components=3)
        pca_all = pca.fit_transform(all_features_concat)
        
        pca_min, pca_max = pca_all.min(), pca_all.max()
        pca_all = (pca_all - pca_min) / (pca_max - pca_min + 1e-8)
        
        result = []
        offset = 0
        for H, W in shapes:
            n_pixels = H * W
            pca_img = pca_all[offset:offset+n_pixels].reshape(H, W, 3)
            result.append((pca_img * 255).astype(np.uint8))
            offset += n_pixels
        
        return result


class UnifiedLogger:
    """Unified logger interface for ClearML and TensorBoard."""
    
    def __init__(self, logger_type, task=None, clearml_logger=None, tb_writer=None):
        self.logger_type = logger_type
        self.task = task
        self.clearml_logger = clearml_logger
        self.tb_writer = tb_writer
    
    def report_scalar(self, title, series, value, iteration):
        """Log scalar value."""
        if self.logger_type == "clearml" and self.clearml_logger:
            self.clearml_logger.report_scalar(title=title, series=series, value=value, iteration=iteration)
        elif self.logger_type == "tensorboard" and self.tb_writer:
            tag = f"{title}/{series}" if title else series
            self.tb_writer.add_scalar(tag, value, iteration)
    
    def report_image(self, title, series, image, iteration):
        """Log image."""
        if self.logger_type == "clearml" and self.clearml_logger:
            self.clearml_logger.report_image(title=title, series=series, image=image, iteration=iteration)
        elif self.logger_type == "tensorboard" and self.tb_writer:
            tag = f"{title}/{series}" if title else series
            # Convert numpy array to tensor format (HWC -> CHW)
            if len(image.shape) == 3:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                self.tb_writer.add_image(tag, image_tensor, iteration, dataformats='CHW')


class LoggingManager:
    """Manage logging and experiment tracking."""
    
    @staticmethod
    def setup_logging(accelerator):
        """Configure logging verbosity."""
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
    
    @staticmethod
    def init_logger(accelerator, args):
        """Initialize experiment tracking logger (ClearML or TensorBoard)."""
        task, logger = None, None
        
        if not accelerator.is_main_process:
            return task, logger
        
        if args.logger == "clearml":
            if not CLEARML_AVAILABLE:
                print("[ERROR] ClearML not available. Install with: pip install clearml")
                print("[INFO] Falling back to TensorBoard")
                args.logger = "tensorboard"
            else:
                logging.getLogger('clearml.metrics').setLevel(logging.ERROR)
                task = Task.init(
                    project_name=args.tracker_project_name,
                    task_name=args.tracker_run_name,
                    auto_connect_frameworks={'pytorch': False}
                )
                clearml_logger = task.get_logger()
                task.connect_configuration(vars(args))
                logger = UnifiedLogger("clearml", task=task, clearml_logger=clearml_logger)
                print(f"[INFO] Using ClearML logger - Project: {args.tracker_project_name}, Run: {args.tracker_run_name}")
        
        if args.logger == "tensorboard":
            if not TENSORBOARD_AVAILABLE:
                print("[ERROR] TensorBoard not available. Install with: pip install tensorboard")
                print("[WARNING] No logger available - training will continue without logging")
                return None, None
            else:
                log_dir = os.path.join(args.output_dir, "logs", args.tracker_run_name)
                os.makedirs(log_dir, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=log_dir)
                logger = UnifiedLogger("tensorboard", tb_writer=tb_writer)
                print(f"[INFO] Using TensorBoard logger - Log dir: {log_dir}")
                print(f"[INFO] To view logs, run: tensorboard --logdir {os.path.join(args.output_dir, 'logs')}")
        
        return task, logger


class ModelManager:
    """Manage model setup and checkpointing."""
    
    @staticmethod
    def setup_model(args):
        """Setup and configure the model."""
        network = Splatent(timestep=args.timestep)
        network.set_train()
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                network.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers not available, install with: pip install xformers")
        
        if args.gradient_checkpointing:
            network.unet.enable_gradient_checkpointing()
        
        return network
    
    @staticmethod
    def setup_loss_networks(args):
        """Setup perceptual loss networks."""
        net_lpips = lpips.LPIPS(net='vgg', spatial=args.lpips_spatial).cuda()
        net_lpips.requires_grad_(False)
        return net_lpips
    
    @staticmethod
    def load_checkpoint(args, net_splatent, optimizer):
        """Load checkpoint if resuming training."""
        global_step = 0
        if args.resume is not None:
            if os.path.isdir(args.resume):
                ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
                assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
                print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
                global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
                net_splatent, optimizer = load_ckpt_from_state_dict(net_splatent, optimizer, ckpt_files[-1])
            elif args.resume.endswith(".pkl"):
                print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
                global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
                net_splatent, optimizer = load_ckpt_from_state_dict(net_splatent, optimizer, args.resume)
            else:
                raise NotImplementedError(f"Invalid resume path: {args.resume}")
        else:
            print("="*50); print(f"Training from scratch"); print("="*50)
        return net_splatent, optimizer, global_step


class LossCalculator:
    """Calculate training losses."""
    
    def __init__(self, args, net_lpips):
        self.args = args
        self.net_lpips = net_lpips
        self.loss_fn = F.l1_loss if args.use_l1 else F.mse_loss
    
    def compute_losses(self, x_tgt_pred, x_tgt, gt_img, vae, weight_dtype):
        """Compute all training losses."""
        x_tgt_v0, x_tgt_pred_v0 = x_tgt[:, 0], x_tgt_pred[:, 0]
        loss_l2 = self.loss_fn(x_tgt_pred_v0.float(), x_tgt_v0.float(), reduction="mean") * self.args.lambda_l2
        
        x_tgt_pred_img_v0 = vae.decode((x_tgt_pred_v0 / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
        if gt_img.shape[2:] != x_tgt_pred_img_v0.shape[2:]:
            gt_img = F.interpolate(gt_img, size=x_tgt_pred_img_v0.shape[2:], mode='bilinear', align_corners=False)
        
        loss_lpips = self.net_lpips(x_tgt_pred_img_v0.float(), gt_img.float()).mean() * self.args.lambda_lpips
        loss_rgb = F.mse_loss(x_tgt_pred_img_v0.float(), gt_img.float(), reduction="mean") * self.args.lambda_rgb
        
        return loss_l2, loss_lpips, loss_rgb, x_tgt_pred_img_v0, gt_img





class TrainingVisualizer:
    """Handle training visualization."""
    
    def __init__(self):
        self.viz_utils = VisualizationUtils()
    
    def visualize_training(self, logger, global_step, B, x_src_clean, x_tgt_pred, x_tgt, gt_img, vae, weight_dtype):
        """Generate and log training visualizations."""
        with torch.no_grad():
            for idx in range(B):
                src_img = vae.decode((x_src_clean[idx] / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
                pred_img = vae.decode((x_tgt_pred[idx] / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
                gt_img_viz = gt_img[idx]
                
                if gt_img_viz.shape != pred_img[0].shape:
                    gt_img_viz = F.interpolate(gt_img_viz.unsqueeze(0), size=pred_img[0].shape[1:], mode='bilinear', align_corners=False).squeeze(0)
                
                src_v0 = ((src_img[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
                src_v1 = ((src_img[1].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
                pred_v0 = ((pred_img[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
                tgt_v0 = ((gt_img_viz.permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
                
                grid_img = self.viz_utils.make_2x2_grid(src_v0, src_v1, pred_v0, tgt_v0)
                logger.report_image(title=f"Step {global_step}", series=f"Train/grid_{idx}", image=grid_img, iteration=global_step)
                
                pca_imgs = self.viz_utils.features_grid_to_pca_rgb([x_src_clean[idx, 0], x_src_clean[idx, 1], x_tgt_pred[idx, 0], x_tgt[idx, 0]])
                grid_feat = self.viz_utils.make_2x2_grid(pca_imgs[0], pca_imgs[1], pca_imgs[2], pca_imgs[3])
                logger.report_image(title=f"Step {global_step}", series=f"Train/features_{idx}", image=grid_feat, iteration=global_step)


class ValidationRunner:
    """Handle validation during training."""
    
    def __init__(self, args):
        self.args = args
        self.viz_utils = VisualizationUtils()
        self.metrics_calc = MetricsCalculator()
    
    def run_validation(self, dl_val, net_splatent, vae, net_lpips, accelerator, weight_dtype, logger, global_step):
        """Run validation and return metrics."""
        l_l2, l_lpips, l_ssim, l_psnr = [], [], [], []
        
        for step, batch_val in enumerate(dl_val):
            if step >= self.args.num_samples_eval:
                break
            
            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
            gt_img_val = batch_val["gt_image"].to(accelerator.device, dtype=weight_dtype)
            
            with torch.no_grad():
                x_tgt_pred = accelerator.unwrap_model(net_splatent)(x_src, prompt_tokens=batch_val["input_ids"].cuda())
                
                if (step + 5) % 10 == 0 and logger:
                    self._visualize_validation(step, x_src, x_tgt_pred, x_tgt, gt_img_val, vae, weight_dtype, logger, global_step)
                
                x_tgt = x_tgt[:, 0]
                x_tgt_pred = x_tgt_pred[:, 0]
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                
                x_tgt_pred_img_val = vae.decode((x_tgt_pred / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
                if gt_img_val.shape[2:] != x_tgt_pred_img_val.shape[2:]:
                    gt_img_val = F.interpolate(gt_img_val, size=x_tgt_pred_img_val.shape[2:], mode='bilinear', align_corners=False)
                
                loss_lpips = net_lpips(x_tgt_pred_img_val.float(), gt_img_val.float()).mean()
                img_pred = (x_tgt_pred_img_val * 0.5 + 0.5).clamp(0, 1)
                img_tgt = (gt_img_val * 0.5 + 0.5).clamp(0, 1)
                ssim_val = self.metrics_calc.ssim(img_pred, img_tgt)
                psnr_val = self.metrics_calc.psnr(img_pred, img_tgt)
                
                l_l2.append(loss_l2.item())
                l_lpips.append(loss_lpips.item())
                l_ssim.append(ssim_val.item())
                l_psnr.append(psnr_val.squeeze().item())
        
        gc.collect()
        torch.cuda.empty_cache()
        return {"val/l2": np.mean(l_l2), "val/lpips": np.mean(l_lpips), "val/ssim": np.mean(l_ssim), "val/psnr": np.mean(l_psnr)}
    
    def _visualize_validation(self, step, x_src, x_tgt_pred, x_tgt, gt_img_val, vae, weight_dtype, logger, global_step):
        """Visualize validation samples."""
        src_img = vae.decode((x_src[0] / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
        pred_img = vae.decode((x_tgt_pred[0] / vae.config.scaling_factor).to(weight_dtype)).sample.clamp(-1, 1)
        gt_img_viz = gt_img_val[0]
        
        if gt_img_viz.shape != pred_img[0].shape:
            gt_img_viz = F.interpolate(gt_img_viz.unsqueeze(0), size=pred_img[0].shape[1:], mode='bilinear', align_corners=False).squeeze(0)
        
        src_v0 = ((src_img[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
        src_v1 = ((src_img[1].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
        pred_v0 = ((pred_img[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
        tgt_v0 = ((gt_img_viz.permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().cpu().numpy()
        
        grid_img = self.viz_utils.make_2x2_grid(src_v0, src_v1, pred_v0, tgt_v0)
        logger.report_image(title=f"Step {global_step}", series=f"Validation/grid_{step}", image=grid_img, iteration=global_step)
        
        pca_imgs = self.viz_utils.features_grid_to_pca_rgb([x_src[0, 0], x_src[0, 1], x_tgt_pred[0, 0], x_tgt[0, 0]])
        grid_feat = self.viz_utils.make_2x2_grid(pca_imgs[0], pca_imgs[1], pca_imgs[2], pca_imgs[3])
        logger.report_image(title=f"Step {global_step}", series=f"Validation/features_{step}", image=grid_feat, iteration=global_step)

    
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    logging_mgr = LoggingManager()
    task, logger = logging_mgr.init_logger(accelerator, args)
    logging_mgr.setup_logging(accelerator)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    model_mgr = ModelManager()
    net_splatent = model_mgr.setup_model(args)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    net_lpips = model_mgr.setup_loss_networks(args)
    
    loss_calc = LossCalculator(args, net_lpips)
    visualizer = TrainingVisualizer()
    validator = ValidationRunner(args)

    # make the optimizer
    layers_to_opt = list(net_splatent.unet.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    def collate_fn(batch):
        return {
            "output_pixel_values": torch.stack([item["output_pixel_values"] for item in batch]),
            "conditioning_pixel_values": torch.stack([item["conditioning_pixel_values"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "gt_image": torch.stack([item["gt_image"] for item in batch]),
        }
    
    dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_splatent.tokenizer, top_k=args.top_k, min_k=args.min_k, scale=args.scale)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, collate_fn=collate_fn)
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_splatent.tokenizer, top_k=args.top_k, min_k=args.min_k, scale=args.scale)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    net_splatent, optimizer, global_step = model_mgr.load_checkpoint(args, net_splatent, optimizer)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to device and cast to weight_dtype
    net_splatent.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    
    # Keep reference to VAE before wrapping
    vae = net_splatent.vae
    
    # Prepare everything with our `accelerator`.
    net_splatent, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_splatent, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)



    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_splatent]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                gt_img = batch["gt_image"].to(accelerator.device, dtype=weight_dtype)
                B, V, C, H, W = x_src.shape

                # Save clean input for visualization (before model modifies it)
                x_src_clean = x_src.clone()

                # forward pass
                x_tgt_pred = net_splatent(x_src, prompt_tokens=batch["input_ids"])
                loss_l2, loss_lpips, loss_rgb, x_tgt_pred_img_v0, gt_img = loss_calc.compute_losses(x_tgt_pred, x_tgt, gt_img, vae, weight_dtype)
                
                loss = loss_l2 + loss_lpips + loss_rgb                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_rgb": loss_rgb.detach().item()
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 0 and logger:
                        visualizer.visualize_training(logger, global_step, B, x_src_clean, x_tgt_pred, x_tgt, gt_img, vae, weight_dtype)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_splatent).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_splatent), optimizer, outf)

                    if args.eval_freq > 0 and global_step % args.eval_freq == 0 and global_step > 0:
                        val_metrics = validator.run_validation(dl_val, net_splatent, vae, net_lpips, accelerator, weight_dtype, logger, global_step)
                        logs.update(val_metrics)
                    
                    # Log to ClearML
                    if logger:
                        for k, v in logs.items():
                            if not k.startswith(("train/", "sample/")):
                                logger.report_scalar(title=f"Loss/{k}", series=k, value=v, iteration=global_step)
    
    # Close logger
    if task is not None:
        task.close()
        print("[INFO] ClearML task marked as completed")
    elif logger and logger.tb_writer:
        logger.tb_writer.close()
        print("[INFO] TensorBoard writer closed")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_rgb", default=1.0, type=float)
    parser.add_argument("--use_l1", action="store_true", help="Use L1 loss instead of L2 loss for reconstruction.")
    parser.add_argument("--lpips_spatial", action="store_true", help="Enable spatial mode for LPIPS loss.")

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["clearml", "tensorboard"], 
                        help="Logger to use for experiment tracking. Options: clearml, tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="Splatent", help="The name of the project to log to (ClearML only).")
    parser.add_argument("--tracker_run_name", type=str, required=True, help="The name of the run/experiment.")

    # details about the model architecture
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--top_k", default=5, type=int, help="Number of reference views to include (total views = 1 source + top_k refs)")
    parser.add_argument("--min_k", default=0, type=int, help="Minimum reference image proximity (0=best match, 1=second best, etc.)")
    parser.add_argument("--scale", action="store_true", help="Apply scaling factor (0.18215) to output features")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)
    
    args = parser.parse_args()
    main(args)
