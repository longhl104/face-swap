"""Face mask generation via pretrained face-parsing ONNX (BiSeNet / CelebAMask-HQ)."""

from __future__ import annotations

import urllib.request
from functools import lru_cache

import cv2
import numpy as np

from src.config import PROJECT_ROOT

PARSING_MODEL_URL = (
    "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
)
PARSING_MODEL_PATH = PROJECT_ROOT / "models" / "face_parsing_resnet18.onnx"

# CelebAMask-HQ classes to include in swap mask (skin + facial features, no hair/neck/bg)
FACE_PARSE_CLASSES = frozenset(range(1, 14))


def _ensure_parsing_model() -> str:
    PARSING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PARSING_MODEL_PATH.exists():
        print(f"Downloading face parsing model to {PARSING_MODEL_PATH}...")
        urllib.request.urlretrieve(PARSING_MODEL_URL, PARSING_MODEL_PATH)
    return str(PARSING_MODEL_PATH)


class FaceParsingMaskGenerator:
    """BiSeNet face parsing (frozen ONNX) for accurate face-region masks."""

    def __init__(self) -> None:
        import onnxruntime as ort

        model_path = _ensure_parsing_model()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = (512, 512)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def _postprocess(self, output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        labels = output.squeeze(0).argmax(0).astype(np.uint8)
        return cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)

    def predict_labels(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return per-pixel class labels (HxW) for a BGR image."""
        h, w = image_bgr.shape[:2]
        outputs = self.session.run(
            self.output_names, {self.input_name: self._preprocess(image_bgr)}
        )
        return self._postprocess(outputs[0], (w, h))

    def face_mask(
        self, image_bgr: np.ndarray, feather_kernel: int = 15
    ) -> np.ndarray:
        """Soft float mask [0, 1] covering parsed face skin and features."""
        labels = self.predict_labels(image_bgr)
        mask = np.isin(labels, list(FACE_PARSE_CLASSES)).astype(np.float32)

        if mask.sum() < 1:
            raise RuntimeError("Face parsing produced an empty mask")

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        k = feather_kernel if feather_kernel % 2 == 1 else feather_kernel + 1
        return cv2.GaussianBlur(mask, (k, k), 0)


@lru_cache(maxsize=1)
def get_face_parser() -> FaceParsingMaskGenerator:
    return FaceParsingMaskGenerator()
