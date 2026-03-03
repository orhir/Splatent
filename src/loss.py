"""Loss utilities for feature extraction."""
import torch


STYLE_WEIGHTS = {
    'relu1_2': 1.0 / 2.6,
    'relu2_2': 1.0 / 4.8,
    'relu3_3': 1.0 / 3.7,
    'relu4_3': 1.0 / 5.6,
    'relu5_3': 10.0 / 1.5
}

DEFAULT_VGG_LAYERS = {
    '3': 'relu1_2',
    '8': 'relu2_2',
    '15': 'relu3_3',
    '22': 'relu4_3',
    '29': 'relu5_3'
}


def get_features(image, model, layers=None):
    """Extract features from specific layers of a model.
    
    Args:
        image: Input image tensor
        model: Pretrained model (e.g., VGG)
        layers: Mapping of layer indices to layer names
    
    Returns:
        Dictionary of features for the specified layers
    """
    if layers is None:
        layers = DEFAULT_VGG_LAYERS
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features