from cnocr import CnOcr
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple


class TextImgExtractor:
    def __init__(self,
                 det_model: str = "en_PP-OCRv3_det",
                 rec_model: str = "en_number_mobile_v2.0"):
        self.ocr = CnOcr(det_model_name=det_model, rec_model_name=rec_model)

    def _extract_scores(self, outputs: list, threshold: float = 0.5) -> Tuple[List[float], List[int]]:
        scores, indices = [], []
        for i, line in enumerate(outputs):
            if line.get("score", 0) > threshold:
                scores.append(line["score"])
                indices.append(i)
        return scores, indices

    def _extract_text(self, outputs: list, indices: List[int]) -> List[str]:
        return [outputs[i]["text"] for i in indices]

    def _extract_boxes(self, outputs: list, indices: List[int]) -> List:
        return [outputs[i].get("position", None) for i in indices]

    def extract(self,
                img: np.ndarray,
                return_boxes: bool = False,
                return_scores: bool = False,
                return_texts: bool = True,
                score_threshold: float = 0.5) -> Tuple[Optional[List[str]], Optional[List], Optional[List[float]]]:

        outputs = self.ocr.ocr(img)
        scores, indices = self._extract_scores(outputs, threshold=score_threshold)

        texts = self._extract_text(outputs, indices) if return_texts else None
        boxes = self._extract_boxes(outputs, indices) if return_boxes else None
        filtered_scores = scores if return_scores else None

        return texts, boxes, filtered_scores

    def extract_text_from_image(self, image_path: str, score_threshold: float = 0.5) -> str:
        image = Image.open(image_path)
        image_np = np.array(image)
        texts, _, _ = self.extract(image_np, return_texts=True, score_threshold=score_threshold)
        return " ".join(texts) if texts else ""
