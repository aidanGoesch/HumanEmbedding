"""
resnet_embedder.py

Low-level, purely visual image embeddings via early layers of ResNet-50
(edges, textures, colors — minimal semantic content).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from models.base_embedder import BaseEmbedder


class ResNetEmbedder(BaseEmbedder):
    """
    Generates low-level visual embeddings using the early layers of a
    pre-trained ResNet-50, stopping before the network develops strong
    semantic representations.
    """

    # Standard ImageNet preprocessing
    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, num_layers: str = "layer1", device: str = None):
        """
        Args:
            num_layers: Which layer to stop at. Options:
                - 'conv1':  Just the first conv layer (very low-level: edges, colors)
                - 'layer1': First residual block (low-level: simple textures, patterns)
                - 'layer2': Second residual block (still mostly visual, minimal semantics)
            device: Torch device string. Defaults to cuda if available, else cpu.
        """
        self.device = device or self._select_device()

        resnet_model = models.resnet50(pretrained=True)
        resnet_model.eval()

        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        layers = [
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
        ]
        if num_layers in ("layer1", "layer2"):
            layers.append(resnet_model.layer1)
        if num_layers == "layer2":
            layers.append(resnet_model.layer2)

        self.features = nn.Sequential(*layers).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_embedding(self, stimulus):
        """
        Generate a low-level visual embedding for an image using early
        ResNet-50 layers.

        Args:
            stimulus: A PIL Image object or a path to an image file.

        Returns:
            A torch.Tensor representing the image's low-level visual embedding.
        """
        image = self._load_image(stimulus)
        processed_image = self._TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = self.features(processed_image)
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # Flatten
        return x