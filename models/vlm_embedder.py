"""
vlm_embedder.py

Vision-language model (Qwen2-VL) embeddings, supporting two workflows:

  1. get_embedding(...)       — pure visual embedding, pooled directly from
                                 the model's hidden states over the image's
                                 own visual tokens. No text is generated.

  2. get_text_embedding(...)  — the VLM first generates a text description
                                 of the requested feature, and that text is
                                 then embedded with a sentence-transformer.
"""

import torch
import numpy as np
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from qwen_vl_utils import process_vision_info

from base_embedder import BaseEmbedder


class VLMEmbedder(BaseEmbedder):
    """
    Generates feature-conditioned embeddings for images using a
    vision-language model (default: Qwen2-VL-2B-Instruct).
    """

    SYSTEM_PROMPT = (
        "You are a precise visual analyst. "
        "Respond only about the specific aspect asked. Do not mention anything else."
    )

    DEFAULT_FEATURE_PROMPT = "Describe this image."

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        embed_model_id: str = "all-mpnet-base-v2",
        device_map: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Args:
            model_id: HF model id for the VLM.
            embed_model_id: sentence-transformers model used to embed VLM
                text descriptions (used only by `get_text_embedding`).
            device_map: device_map passed to `from_pretrained`.
            load_in_4bit: whether to load the VLM in 4-bit quantization.
        """
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        self.model.eval()

        # Lazily loaded — only needed for the describe-then-embed workflow.
        self._embed_model_id = embed_model_id
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embed_model_id)
        return self._embedder

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _build_inputs(self, image, text: str, add_generation_prompt: bool):
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        return vector / (np.linalg.norm(vector) + 1e-10)

    # ------------------------------------------------------------------
    # Workflow 1: pure visual embedding (no text generation)
    # ------------------------------------------------------------------
    def get_embedding(
        self,
        stimulus,
        feature_prompt: str = None,
        layer: int = -1,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Pure visual embedding: run a forward pass and pool the hidden
        states at `layer` over the image's own visual token positions
        only (text tokens are excluded). No text is generated.

        Args:
            stimulus: A PIL Image (or path) to embed.
            feature_prompt: Optional text accompanying the image in the
                forward pass (can bias which visual features the model
                attends to). Defaults to a minimal, neutral prompt.
            layer: Which hidden_states layer to pool (-1 = last layer).
            pooling: "mean" or "max" pooling over visual tokens.
            normalize: L2-normalize the output vector.

        Returns:
            np.ndarray of shape (hidden_dim,)
        """
        image = self._load_image(stimulus)
        prompt = feature_prompt or self.DEFAULT_FEATURE_PROMPT

        inputs = self._build_inputs(image, prompt, add_generation_prompt=False)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of (n_layers + 1) tensors, each (1, seq_len, d_model)
        hidden = outputs.hidden_states[layer].squeeze(0)  # (seq_len, d_model)

        # Isolate visual token positions via Qwen2-VL's image pad token.
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        input_ids = inputs["input_ids"].squeeze(0)
        visual_mask = input_ids == image_token_id

        if visual_mask.sum() == 0:
            # Fallback: use all non-special tokens if image_pad not found
            special_ids = set(self.processor.tokenizer.all_special_ids)
            visual_mask = torch.tensor(
                [tok.item() not in special_ids for tok in input_ids],
                dtype=torch.bool,
            )

        visual_hidden = hidden[visual_mask]  # (n_visual_tokens, d_model)

        if pooling == "mean":
            pooled = visual_hidden.mean(dim=0)
        elif pooling == "max":
            pooled = visual_hidden.max(dim=0).values
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        vector = pooled.cpu().float().numpy()
        if normalize:
            vector = self._normalize(vector)
        return vector

    # ------------------------------------------------------------------
    # Workflow 2: describe in text, then embed the description
    # ------------------------------------------------------------------
    def describe_feature(
        self,
        stimulus,
        feature_prompt: str,
        max_new_tokens: int = 150,
    ) -> str:
        """
        Run a VLM forward pass and return the decoded text response,
        focused on `feature_prompt`.
        """
        image = self._load_image(stimulus)
        inputs = self._build_inputs(image, feature_prompt, add_generation_prompt=True)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Strip the input tokens — keep only newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        description = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return description.strip()

    def get_text_embedding(
        self,
        stimulus,
        feature_prompt: str,
        normalize: bool = True,
        return_description: bool = False,
    ):
        """
        Describe `feature_prompt` for the given image via the VLM, then
        embed that text description with a sentence-transformer.

        Args:
            stimulus: A PIL Image (or path) to embed.
            feature_prompt: Plain-text description of the aspect to focus
                on, e.g. "size", "whether the object is living or
                non-living", "dominant color".
            normalize: L2-normalize the output vector.
            return_description: If True, return (description, vector)
                instead of just vector — useful for inspecting what the
                VLM said.

        Returns:
            np.ndarray of shape (embedding_dim,), or (str, np.ndarray) if
            return_description is True.
        """
        description = self.describe_feature(stimulus, feature_prompt)
        vector = self.embedder.encode(description, convert_to_numpy=True)

        if normalize:
            vector = self._normalize(vector)

        if return_description:
            return description, vector
        return vector

    def get_multi_feature_embedding(
        self,
        stimulus,
        features: list,
        mode: str = "concat",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate text-workflow embeddings for multiple features and combine them.

        Args:
            features: list of feature strings, e.g. ["size", "living", "color"]
            mode: "concat" -> vector of shape (n_features * embedding_dim,)
                  "mean"   -> vector of shape (embedding_dim,)
        """
        vectors = [
            self.get_text_embedding(stimulus, f, normalize=True)
            for f in features
        ]

        if mode == "concat":
            combined = np.concatenate(vectors)
        elif mode == "mean":
            combined = np.mean(vectors, axis=0)
        else:
            raise ValueError(f"mode must be 'concat' or 'mean', got '{mode}'")

        if normalize:
            combined = self._normalize(combined)
        return combined