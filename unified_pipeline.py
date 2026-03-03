#!/usr/bin/env python3
"""
Unified Pipeline for Splatent: Reference-based Diffusion Enhancement for 3D Gaussian Splatting

This script integrates all stages of the Splatent pipeline:
1. Extract features from images using SD-Turbo VAE
2. Train Feature-3DGS with extracted features
3. Render novel view features from trained 3DGS model
4. Generate dataset JSON with camera-aware reference selection
5. Run inference with the trained Splatent model

Usage:
    python unified_pipeline.py --scene_path /path/to/scene --output_dir ./results
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Print progress on same line, overwriting previous output
                if any(prog_indicator in output.lower() for prog_indicator in ['progress', 'iter', 'step', '%']):
                    print(f"\r{output.strip()}", end='', flush=True)
        print()  # New line after command completes
        process.wait()
        
        if process.returncode != 0 and check:
            raise subprocess.CalledProcessError(process.returncode, cmd, '\n'.join(output_lines))
        return subprocess.CompletedProcess(cmd, process.returncode, '\n'.join(output_lines), '')
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}")
        print(f"Output: {e.output}")
        if check:
            raise
        return e

def extract_features(scene_path, res="images_4", device="cuda", output="features_sd_turbo"):
    """Stage 1: Extract SD-Turbo VAE latent features from RGB images."""
    cmd = [
        "python", "scripts/extract_features.py",
        "--input", scene_path,
        "--output", output,
        "--res", res,
        "--device", device
    ]
    return run_command(cmd)

def train_feature_3dgs(scene_path, output_model_path, images_dir="images_4"):
    """Stage 2: Train 3D Gaussian Splatting with extracted features."""
    # Train the model
    cmd = [
        "python", "submodules/feature-3dgs/train.py",
        "-s", scene_path,
        "-m", output_model_path,
        "-i", images_dir
    ]
    result = run_command(cmd)
    
    # Render novel views
    render_cmd = [
        "python", "submodules/feature-3dgs/render.py",
        "-m", output_model_path
    ]
    return run_command(render_cmd)

def generate_dataset(rendered_features_path, original_dataset_path, output_json, top_k=3, scene_name=None):
    """Stage 3: Generate dataset JSON from rendered 3DGS features."""
    cmd = [
        "python", "scripts/generate_dataset.py",
        "--out_path", rendered_features_path,
        "--original_path", original_dataset_path,
        "--output", output_json,
        "--top_k", str(top_k)
    ]
    
    # Auto-detect COLMAP format - check in the scene subdirectory
    if scene_name:
        colmap_path = Path(original_dataset_path) / scene_name / "sparse" / "0" / "images.bin"
    else:
        colmap_path = Path(original_dataset_path) / "sparse" / "0" / "images.bin"
    
    if colmap_path.exists():
        cmd.append("--llff")
        print(f"Detected COLMAP format at {colmap_path}, using --llff flag")
    
    return run_command(cmd)

def run_inference(model_splatent_path, dataset_json_path, output_dir, top_k=3, save_images=False, compute_metrics=False, rgb_root=None):
    """Stage 4: Run inference using inference_dataset.py."""
    cmd = [
        "python", "src/inference_dataset.py",
        "--model_path", model_splatent_path,
        "--dataset_path", dataset_json_path,
        "--output_dir", output_dir,
        "--method_name", "mv_t300",
        "--timestep", "300",
        "--batch_size", "1",
        "--top_k", str(top_k),
        "--scale"
    ]
    
    if save_images:
        cmd.append("--save_images")
    
    if not compute_metrics:
        cmd.append("--no_metrics")
    elif rgb_root:
        cmd.extend(["--rgb_root", rgb_root])
    
    return run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="Unified Splatent Pipeline")
    
    # Main paths
    parser.add_argument("--scene_path", type=str, required=True,
                        help="Path to scene directory containing images")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for all results")
    parser.add_argument("--model_splatent_path", type=str, default="ckpt/ckpt.pkl",
                        help="Path to trained Splatent checkpoint (default: ckpt/ckpt.pkl)")
    
    # Feature extraction parameters
    parser.add_argument("--images_dir", type=str, default="images_4",
                        help="Input images directory name")
    parser.add_argument("--features_output", type=str, default="features_sd_turbo",
                        help="Output directory name for features")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for feature extraction")
    
    # 3DGS training parameters
    parser.add_argument("--model_3dgs_output", type=str, default="./output_3dgs",
                        help="Output directory for 3DGS model")
    
    # Dataset generation parameters
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of closest reference views")
    
    # Checkpoint arguments
    parser.add_argument("--skip_completed_stages", action="store_true",
                        help="Skip stages that have already been completed")
    parser.add_argument("--save_images", action="store_true",
                        help="Save output images during inference")
    parser.add_argument("--compute_metrics", action="store_true",
                        help="Compute metrics (requires GT RGB images)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine paths
    # generate_dataset.py expects scene directories, so we need to wrap the render output
    rendered_features_base = os.path.join(args.output_dir, "rendered_features")
    scene_name = os.path.basename(os.path.normpath(args.scene_path))
    rendered_features_path = rendered_features_base
    dataset_json_path = os.path.join(args.output_dir, "dataset.json")
    inference_output_dir = os.path.join(args.output_dir, "inference_results")
    
    # Prepare paths for symlink creation
    Path(rendered_features_base).mkdir(parents=True, exist_ok=True)
    scene_link = os.path.join(rendered_features_base, scene_name)
    render_dir = os.path.abspath(os.path.join(args.model_3dgs_output, "render"))
    
    # Stage 1: Extract features
    if not args.skip_completed_stages or not os.path.exists(os.path.join(args.scene_path, args.features_output)):
        print("=== Stage 1: Extracting Features ===")
        extract_features(
            scene_path=args.scene_path,
            res=args.images_dir,
            device=args.device,
            output=args.features_output
        )
    else:
        print("=== Skipping Stage 1: Extracting Features ===")
    
    # Stage 2: Train Feature-3DGS
    if not args.skip_completed_stages or not os.path.exists(os.path.join(args.model_3dgs_output, "render", "nvs")):
        print("=== Stage 2: Training Feature-3DGS ===")
        train_feature_3dgs(
            scene_path=args.scene_path,
            output_model_path=args.model_3dgs_output,
            images_dir=args.images_dir
        )
    else:
        print("=== Skipping Stage 2: Training Feature-3DGS ===")
    
    # Ensure symlink exists before stage 3
    if os.path.islink(scene_link) and not os.path.exists(scene_link):
        # Remove broken symlink
        os.unlink(scene_link)
    if not os.path.exists(scene_link) and os.path.exists(render_dir):
        os.symlink(render_dir, scene_link, target_is_directory=True)
    
    # Stage 3: Generate dataset
    if not args.skip_completed_stages or not os.path.exists(dataset_json_path):
        print("=== Stage 3: Generating Dataset ===")
        # generate_dataset.py expects parent dir containing scene folders
        original_dataset_parent = os.path.dirname(os.path.normpath(args.scene_path))
        generate_dataset(
            rendered_features_path=rendered_features_path,
            original_dataset_path=original_dataset_parent,
            output_json=dataset_json_path,
            top_k=args.top_k,
            scene_name=scene_name
        )
    else:
        print("=== Skipping Stage 3: Generating Dataset ===")
    
    # Stage 4: Run inference
    if not args.skip_completed_stages or not os.path.exists(inference_output_dir):
        print("=== Stage 4: Running Inference ===")
        run_inference(
            model_splatent_path=args.model_splatent_path,
            dataset_json_path=dataset_json_path,
            output_dir=inference_output_dir,
            top_k=args.top_k,
            save_images=args.save_images,
            compute_metrics=args.compute_metrics,
            rgb_root=original_dataset_parent if args.compute_metrics else None
        )
    else:
        print("=== Skipping Stage 4: Running Inference ===")
    
    print("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()