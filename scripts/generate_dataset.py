import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
import struct

def load_transforms(transforms_path: Path) -> Dict:
    """Load camera transforms from transforms.json"""
    with open(transforms_path) as f:
        return json.load(f)

def load_colmap_cameras(cameras_path: Path) -> Dict:
    """Load camera data from cameras.json (COLMAP format)"""
    with open(cameras_path) as f:
        cameras = json.load(f)
    
    frames = []
    for cam in cameras:
        # Convert position + rotation to transform_matrix
        pos = np.array(cam["position"])
        rot = np.array(cam["rotation"])
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot
        transform_matrix[:3, 3] = pos
        
        frames.append({
            "file_path": cam["img_name"],
            "transform_matrix": transform_matrix.tolist()
        })
    
    return {"frames": frames}

def load_colmap_images_bin(images_bin_path: Path) -> Dict:
    """Load all camera poses from COLMAP images.bin"""
    frames = []
    
    with open(images_bin_path, 'rb') as f:
        num_images = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_images):
            image_id = struct.unpack('I', f.read(4))[0]
            qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
            tx, ty, tz = struct.unpack('ddd', f.read(24))
            camera_id = struct.unpack('I', f.read(4))[0]
            
            name_len = 0
            name_bytes = b''
            while True:
                char = f.read(1)
                if char == b'\x00':
                    break
                name_bytes += char
            img_name = name_bytes.decode('utf-8')
            
            num_points2D = struct.unpack('Q', f.read(8))[0]
            f.read(24 * num_points2D)  # Skip point2D data
            
            # Convert quaternion to rotation matrix
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])
            
            # COLMAP uses world-to-camera, convert to camera-to-world
            t = np.array([tx, ty, tz])
            pos = -R.T @ t
            rot = R.T
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot
            transform_matrix[:3, 3] = pos
            
            frames.append({
                "file_path": img_name,
                "transform_matrix": transform_matrix.tolist()
            })
    
    return {"frames": frames}

def compute_camera_distance(cam1: Dict, cam2: Dict) -> float:
    """Compute distance between camera poses (position + orientation)"""
    T1 = np.array(cam1["transform_matrix"])
    T2 = np.array(cam2["transform_matrix"])
    
    # Position distance
    pos_dist = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    
    # Rotation distance (Frobenius norm of rotation difference)
    R1, R2 = T1[:3, :3], T2[:3, :3]
    rot_dist = np.linalg.norm(R1 - R2, 'fro')
    
    return pos_dist + 0.5 * rot_dist  # Weight rotation less than position

def find_top_k_closest_ref_views(target_frame: Dict, train_frames: List[Dict], k: int) -> List[int]:
    """Find top K closest training views to target frame, sorted from closest to furthest"""
    distances = [(i, compute_camera_distance(target_frame, train_frame)) for i, train_frame in enumerate(train_frames)]
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    # Return top K indices
    return [idx for idx, _ in distances[:k]]

def generate_dataset(out_path: Path, original_path: Path, output_path: Path, top_k: int = 3, benchmark_path: Path = None, is_benchmark: bool = False, llff: bool = False):
    """Generate dataset from feature files"""
    
    # Get all scene directories
    scene_dirs = [d for d in out_path.iterdir() if d.is_dir()]
    
    # Filter out benchmark scenes if benchmark_path is provided (only in non-benchmark mode)
    if benchmark_path and not is_benchmark:
        benchmark_scenes = {d.name for d in benchmark_path.iterdir() if d.is_dir()}
        skipped_scenes = [d.name for d in scene_dirs if d.name in benchmark_scenes]
        scene_dirs = [d for d in scene_dirs if d.name not in benchmark_scenes]
        print(f"Skipped {len(skipped_scenes)} benchmark scenes:")
        for scene_name in skipped_scenes:
            print(f"  - {scene_name}")
    
    random.seed(42)
    
    # Split scenes: benchmark mode = all test, normal mode = 90% train, 10% test
    if is_benchmark:
        train_scenes = []
        test_scenes = scene_dirs
    else:
        train_scenes = random.sample(scene_dirs, int(len(scene_dirs) * 0.9))
        test_scenes = [s for s in scene_dirs if s not in train_scenes]
    
    dataset = {"train": {}, "test": {}}
    data_id = 0
    
    def process_scenes(scenes, split_name):
        nonlocal data_id
        scene_pbar = tqdm(scenes, desc=f"{split_name.capitalize()} scenes")
        
        # Count total NVS features for overall progress
        total_nvs_features = 0
        for scene_dir in scenes:
            nvs_dir = scene_dir / "nvs"
            if nvs_dir.exists():
                total_nvs_features += len(list(nvs_dir.glob("*.pt")))
        
        feature_pbar = tqdm(total=total_nvs_features, desc=f"{split_name.capitalize()} features", position=1)
        
        for scene_dir in scene_pbar:
            scene_id = scene_dir.name
            if "_model" in scene_id:
                continue
            # Check required directories
            nvs_dir = scene_dir / "nvs"
            nvs_gt_dir = scene_dir / "nvs_gt"
            train_gt_dir = scene_dir / "train_gt"
            
            if not all(d.exists() for d in [nvs_dir, nvs_gt_dir, train_gt_dir]):
                # print missing dir
                print(f"Missing directory for scene {scene_id}")
                print(f"nvs_dir path: {nvs_dir} - {nvs_dir.exists()}")
                print(f"nvs_gt_dir path: {nvs_gt_dir} - {nvs_gt_dir.exists()}")
                continue
            
            if llff:
                # Load all cameras from COLMAP images.bin
                colmap_images_path = original_path / scene_id / "sparse" / "0" / "images.bin"
                if not colmap_images_path.exists():
                    print(f"COLMAP images.bin not found at {colmap_images_path} - skipping scene")
                    continue
                transforms = load_colmap_images_bin(colmap_images_path)
                frames = transforms["frames"]
            else:
                transforms_path = original_path / scene_id / "gaussian_splat" / "transforms.json"
                if not transforms_path.exists():
                    transforms_path = original_path / scene_id / "transforms.json"
                    if not transforms_path.exists():
                        continue
                transforms = load_transforms(transforms_path)
                frames = transforms["frames"]
            
            # Get feature files
            nvs_features = list(nvs_dir.glob("*.pt"))
            train_gt_features = list(train_gt_dir.glob("*.pt"))
            
            scene_pbar.set_postfix({"scene": scene_id[:8], "nvs_features": len(nvs_features)})
            
            for nvs_feature in nvs_features:
                nvs_gt_feature = nvs_gt_dir / nvs_feature.name
                if not nvs_gt_feature.exists():
                    continue
                
                # Find corresponding frame
                img_name = nvs_feature.stem
                nvs_frame = None
                for frame in frames:
                    if img_name in frame["file_path"]:
                        nvs_frame = frame
                        break
                
                if not nvs_frame:
                    continue
                
                # Find closest training frame
                train_frames = [f for f in frames if any(tf.stem in f["file_path"] for tf in train_gt_features)]
                if not train_frames:
                    continue
                
                # Find top K closest training frames
                ref_indices = find_top_k_closest_ref_views(nvs_frame, train_frames, top_k)
                
                # Find corresponding ref feature files
                ref_features = []
                for ref_idx in ref_indices:
                    ref_frame = train_frames[ref_idx]
                    for tf in train_gt_features:
                        if tf.stem in ref_frame["file_path"]:
                            ref_features.append(str(tf))
                            break
                
                if len(ref_features) == 0:
                    continue
                
                entry = {
                    "feature": str(nvs_feature),
                    "target_feature": str(nvs_gt_feature),
                    "ref_features": ref_features,  # Now a list of top K
                    "prompt": "remove degradation"
                }
                
                dataset[split_name][f"{data_id:06d}"] = entry
                data_id += 1
                feature_pbar.update(1)
        
        feature_pbar.close()
    
    if train_scenes:
        process_scenes(train_scenes, "train")
    process_scenes(test_scenes, "test")
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated dataset with {len(dataset['train'])} training pairs and {len(dataset['test'])} test pairs")
    print(f"Used {len(train_scenes)} train scenes and {len(test_scenes)} test scenes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True, help="Path to output features directory")
    parser.add_argument("--original_path", type=str, required=True, help="Path to original dataset with transforms.json")
    parser.add_argument("--output", type=str, default="data/dataset.json", help="Output JSON file path")
    parser.add_argument("--top_k", type=int, default=10, help="Number of closest reference views to include")
    parser.add_argument("--benchmark_path", type=str, default=None, help="Path to benchmark directory (scenes in this path will be skipped)")
    parser.add_argument("--benchmark", action="store_true", help="Generate benchmark dataset (all scenes as test, no train split)")
    parser.add_argument("--llff", action="store_true", help="Use LLFF/COLMAP camera format (cameras.json) instead of transforms.json")
    
    args = parser.parse_args()
    
    out_path = Path(args.out_path)
    original_path = Path(args.original_path)
    output_path = Path(args.output)
    benchmark_path = Path(args.benchmark_path) if args.benchmark_path else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_dataset(out_path, original_path, output_path, args.top_k, benchmark_path, args.benchmark, args.llff)