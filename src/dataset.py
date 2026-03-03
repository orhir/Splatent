import json
import os
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    """Dataset for paired feature-based training with reference views."""
    
    def __init__(self, dataset_path, split, tokenizer=None, top_k=1, min_k=0, 
                 scale=False, rgb_root=None, no_metrics=False):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.feature_ids = list(self.data.keys())
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.min_k = min_k
        self.scale = scale
        self.rgb_root = rgb_root
        self.no_metrics = no_metrics
    
    def _feature_path_to_image_path(self, feature_path):
        """Convert feature .pt path to corresponding image path."""
        if self.rgb_root:
            parts = feature_path.split('/')
            filename = os.path.basename(feature_path).replace('.pt', '.png')
            
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] and not parts[i].endswith('.pt'):
                    scene_name = parts[i - 1] if i > 0 else parts[i]
                    break
            
            return os.path.join(self.rgb_root, scene_name, 'images_4', filename)
        return feature_path.replace('.pt', '.png')

    def __len__(self):
        return len(self.feature_ids)

    def _load_features(self, input_path, output_path):
        """Load input and output features with error handling."""
        try:
            input_feature = torch.load(input_path, map_location='cpu')
            output_feature = torch.load(output_path, map_location='cpu')
            return input_feature, output_feature
        except Exception as e:
            raise RuntimeError(f"Error loading features: {e}")
    
    def _load_gt_image(self, feature_path):
        """Load ground truth image if metrics are enabled."""
        if self.no_metrics:
            return torch.zeros(3, 512, 512)
        
        gt_image_path = self._feature_path_to_image_path(feature_path)
        if not os.path.exists(gt_image_path):
            raise FileNotFoundError(f"GT image not found: {gt_image_path}")
        
        gt_image = Image.open(gt_image_path).convert("RGB")
        return F.to_tensor(gt_image) * 2.0 - 1.0
    
    def _prepare_features(self, input_feature, output_feature, ref_features):
        """Stack input and reference features."""
        if ref_features:
            if isinstance(ref_features, str):
                ref_features = [ref_features]
            
            ref_tensors = [torch.load(ref_f, map_location='cpu') 
                          for ref_f in ref_features[self.min_k:self.min_k + self.top_k]]
            feature_t = torch.stack([input_feature] + ref_tensors, dim=0)
            output_t = torch.stack([output_feature] + ref_tensors, dim=0)
        else:
            feature_t = input_feature.unsqueeze(0)
            output_t = output_feature.unsqueeze(0)
        
        if self.scale:
            feature_t *= 0.18215
            output_t *= 0.18215
        
        return feature_t, output_t
    
    def __getitem__(self, idx):
        feature_id = self.feature_ids[idx]
        data_item = self.data[feature_id]
        
        input_feature_path = data_item["feature"]
        output_feature_path = data_item["target_feature"]
        ref_features = data_item.get("ref_features", data_item.get("ref_feature"))
        caption = data_item["prompt"]
        
        try:
            input_feature, output_feature = self._load_features(input_feature_path, output_feature_path)
            gt_image = self._load_gt_image(output_feature_path)
            feature_t, output_t = self._prepare_features(input_feature, output_feature, ref_features)
        except Exception as e:
            print(f"Error processing {feature_id}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        out = {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": feature_t,
            "caption": caption,
            "gt_image": gt_image,
            "feature_id": feature_id,
        }
        
        if self.tokenizer:
            out["input_ids"] = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
        
        return out
