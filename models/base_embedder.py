"""
base_embedder.py

Defines the shared interface for all stimulus embedding models used in
the embedding analysis pipeline.
"""

from abc import ABC, abstractmethod
from PIL import Image
import torch


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.

    Subclasses (ResNetEmbedder, CLIPEmbedder, VLMEmbedder, ...) must implement
    `get_embedding`, which takes a stimulus (currently: an image) and returns
    its vector embedding.
    """

    @abstractmethod
    def get_embedding(self, stimulus):
        """
        Return a vector embedding for the given stimulus.

        Args:
            stimulus: A PIL Image, a path to an image file, or another
                stimulus type supported by the subclass.

        Returns:
            A vector embedding (torch.Tensor or np.ndarray, depending on
            the subclass).
        """
        raise NotImplementedError

    @staticmethod
    def _load_image(image_input):
        """
        Shared helper: accept either a PIL Image or a path string and
        return a PIL Image in RGB mode.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
        return image.convert("RGB")
    
    @staticmethod
    def _select_device():
        """Pick the best available device: cuda > mps > cpu."""
        if torch.cuda.is_available():
            ret =  "cuda"
        if torch.backends.mps.is_available():
            ret =  "mps"
        ret =  "cpu"

        print(f"using {ret}")
        return ret