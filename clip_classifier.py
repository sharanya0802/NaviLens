"""
clip_classifier.py — CLIP-based zero-shot object classification.

Replaces YOLO's unreliable 80-class labels with CLIP's open-vocabulary
classification into ~150 everyday object/product categories.
Runs entirely locally — no API calls needed.
"""

import threading
import cv2
import numpy as np
import torch
from PIL import Image

# ── Rich category list for zero-shot classification ─────────────────
OBJECT_CATEGORIES = [
    # Personal care / beauty
    "face serum", "face cream", "moisturizer", "sunscreen",
    "soap bar", "liquid soap", "hand sanitizer",
    "shampoo bottle", "conditioner bottle", "hair oil",
    "toothpaste", "toothbrush", "mouthwash",
    "deodorant", "perfume bottle", "lipstick", "makeup",
    "nail polish", "skincare product", "lotion bottle",
    # Food & drinks
    "water bottle", "soft drink can", "juice box", "milk carton",
    "snack packet", "chips bag", "chocolate bar", "biscuit packet",
    "cereal box", "instant noodles packet", "rice bag",
    "coffee jar", "tea box", "sugar packet",
    # Electronics
    "smartphone", "tablet", "laptop computer",
    "charger", "USB cable", "power bank",
    "headphones", "earbuds", "bluetooth speaker",
    "remote control", "mouse", "keyboard",
    "smartwatch", "calculator",
    # Stationery
    "notebook", "diary", "textbook", "novel",
    "pen", "pencil", "marker", "eraser", "ruler",
    "stapler", "scissors", "tape roll", "glue stick",
    # Clothing & accessories
    "t-shirt", "shirt", "jacket", "jeans",
    "shoe", "sandal", "slipper",
    "wallet", "handbag", "backpack",
    "sunglasses", "eyeglasses", "wristwatch",
    "belt", "cap", "scarf", "tie",
    # Household
    "curtain", "towel", "pillow", "blanket",
    "duster", "broom", "mop", "sponge",
    "tissue box", "paper roll", "napkin",
    "plate", "bowl", "cup", "mug", "glass",
    "spoon", "fork", "knife", "bottle opener",
    "candle", "flower vase", "photo frame",
    # Kitchen
    "microwave oven", "toaster", "blender",
    "cooking pot", "frying pan", "kettle",
    "food container", "lunch box", "thermos",
    # Medicine & health
    "medicine box", "pill bottle", "first aid kit",
    "face mask", "thermometer", "bandage",
    # Sports & outdoor
    "tennis ball", "cricket bat", "football",
    "yoga mat", "dumbbell", "skipping rope",
    "umbrella", "water flask",
    # Misc
    "key", "coin", "banknote",
    "toy", "stuffed animal", "action figure",
    "playing cards", "board game",
    "plant pot", "small plant",
    "box", "cardboard box", "plastic bag",
    "spray can", "cleaning spray",
]


class CLIPClassifier:
    """
    Zero-shot image classifier using OpenCLIP.
    Lazy-loads the model on first use (~350 MB download first time).
    Pre-computes text embeddings for all categories.
    """

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._text_features = None
        self._initialized = False
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            print("[CLIP] Loading CLIP model (first run downloads ~350 MB)...")
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(self._device).eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            # Pre-compute text embeddings for all categories
            prompts = [f"a photo of a {c}" for c in OBJECT_CATEGORIES]
            text_tokens = tokenizer(prompts).to(self._device)
            with torch.no_grad():
                text_feat = model.encode_text(text_tokens)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)

            self._model = model
            self._preprocess = preprocess
            self._text_features = text_feat
            self._initialized = True
            print(f"[CLIP] Ready ({len(OBJECT_CATEGORIES)} categories, device={self._device}).")

    def classify(self, image_crop: np.ndarray, top_k: int = 1) -> list:
        """
        Classify a cropped BGR image into the closest object categories.

        Returns:
            List of (category_name, confidence) tuples, sorted by confidence.
        """
        self._ensure_loaded()

        if image_crop.size == 0 or image_crop.shape[0] < 10 or image_crop.shape[1] < 10:
            return []

        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            img_feat = self._model.encode_image(image_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            similarity = (img_feat @ self._text_features.T).squeeze(0)
            probs = similarity.softmax(dim=0)
            values, indices = probs.topk(top_k)

        results = []
        for v, i in zip(values, indices):
            results.append((OBJECT_CATEGORIES[int(i)], float(v)))
        return results
