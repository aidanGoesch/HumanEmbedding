"""
clip_embedder.py

Semantic image embeddings via CLIP.
"""

import torch
from transformers import CLIPModel, CLIPProcessor

from models.base_embedder import BaseEmbedder


class CLIPEmbedder(BaseEmbedder):
    """
    Generates semantic vector embeddings for images using CLIP's vision
    encoder (`get_image_features`).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or self._select_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_embedding(self, stimulus):
        """
        Generate a vector embedding for an image using CLIP.

        Args:
            stimulus: A PIL Image object or a path to an image file.

        Returns:
            A torch.Tensor representing the image's vector embedding.
        """
        image = self._load_image(stimulus)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features