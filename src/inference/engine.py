"""Core face swap inference engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from models.face_swap_model import FaceSwapModel
from src.config import StoragePaths, load_config
from src.data.preprocess import FacePreprocessor
from src.inference.blending import blend_face_into_image
from src.inference.classical_swap import warp_and_blend


class FaceSwapEngine:
    """Production-ready face swap inference engine."""

    def __init__(self, model_path: Path | None = None, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.paths = StoragePaths(self.config)
        self.paths.ensure_dirs()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = self.config["image_size"]
        infer_cfg = self.config["inference"]

        self.preprocessor = FacePreprocessor(image_size=self.image_size)
        train_cfg = self.config["training"]
        self.model = FaceSwapModel(
            facenet_pretrained=train_cfg.get("facenet_pretrained", "vggface2"),
        ).to(self.device)

        weights = model_path or self.paths.best_model_path
        if weights.exists():
            self.model.load_trainable(weights, map_location=self.device)
            print(f"Loaded generator weights from {weights}")
        else:
            print(f"Warning: No weights found at {weights}. Using untrained model.")

        self.model.eval()
        self.blend_ratio = infer_cfg["blend_ratio"]
        self.feather_kernel = infer_cfg["feather_kernel"]
        self.mode = infer_cfg.get("mode", "classical")
        self.neural_blend = infer_cfg.get("neural_blend", 0.15)

    def _to_tensor(self, face: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor.unsqueeze(0).to(self.device)

    def _from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        face = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip((face + 1.0) * 127.5, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _neural_swap_face(
        self, source_face: np.ndarray, target_face: np.ndarray
    ) -> np.ndarray:
        source_tensor = self._to_tensor(source_face)
        target_tensor = self._to_tensor(target_face)
        swapped_tensor = self.model.swap(source_tensor, target_tensor)
        return self._from_tensor(swapped_tensor)

    @torch.no_grad()
    def swap_faces(
        self, source_image: np.ndarray, target_image: np.ndarray
    ) -> np.ndarray | None:
        """
        Swap source identity onto the target face.

        Modes (config inference.mode):
        - classical: landmark warp + alpha blend (reliable, visible swap)
        - neural:    trained generator only
        - hybrid:    classical base + light neural refinement
        """
        source_region = self.preprocessor.detect_face(source_image)
        target_region = self.preprocessor.detect_face(target_image)
        if source_region is None or target_region is None:
            return None

        if self.mode == "neural":
            return self._swap_neural_only(source_image, target_image, source_region, target_region)

        classical = warp_and_blend(
            source_image, target_image, source_region, target_region
        )
        if classical is None or self.mode == "classical":
            return classical

        # hybrid: optional light neural blend on top of classical warp
        source_face = self.preprocessor.crop_and_align(source_image, source_region)
        target_face = self.preprocessor.crop_and_align(target_image, target_region)
        neural_face = self._neural_swap_face(source_face, target_face)
        return blend_face_into_image(
            classical,
            neural_face,
            target_region,
            blend_ratio=self.neural_blend,
            feather_kernel=self.feather_kernel,
        )

    def _swap_neural_only(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_region,
        target_region,
    ) -> np.ndarray | None:
        source_face = self.preprocessor.crop_and_align(source_image, source_region)
        target_face = self.preprocessor.crop_and_align(target_image, target_region)
        swapped_face = self._neural_swap_face(source_face, target_face)
        return blend_face_into_image(
            target_image,
            swapped_face,
            target_region,
            blend_ratio=self.blend_ratio,
            feather_kernel=self.feather_kernel,
        )

    def swap_from_paths(
        self, source_path: Path, target_path: Path, output_path: Path | None = None
    ) -> Path | None:
        """Swap faces from file paths and save the result."""
        source = cv2.imread(str(source_path))
        target = cv2.imread(str(target_path))
        if source is None or target is None:
            raise FileNotFoundError("Could not read source or target image")

        result = self.swap_faces(source, target)
        if result is None:
            return None

        out = output_path or self.paths.inference_output / f"swap_{target_path.stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), result)
        return out

    def close(self) -> None:
        self.preprocessor.close()

    def __enter__(self) -> FaceSwapEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
