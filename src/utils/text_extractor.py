
from cnocr import CnOcr
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np 
from typing import Optional

class TextImgExtractor():

    def __init__(self,
                 engine: str = "cnocr",  # 'cnocr' or 'paddleocr' engine
                 detection_model_name: str = "en_PP-OCRv3_det",
                 rec_model_name: str = "en_number_mobile_v2.0"):
        self.engine = engine.lower()
        self.detection_model_name = detection_model_name
        self.rec_model_name = rec_model_name
        self._setup_ocr()

    def _setup_ocr(self):
        if self.engine == "cnocr":
            self.ocr = CnOcr(det_model_name=self.detection_model_name,
                             rec_model_name=self.rec_model_name)
        elif self.engine == "paddleocr":
            self.ocr = PaddleOCR(lang='en')
        else:
            raise ValueError("Unsupported OCR engine. Choose 'cnocr' or 'paddleocr'.")
        

    def _extract_text(self, outputs:list, scores_indces:Optional[list])->list:
        txts = [line['text'] for line in outputs]
        
        txts = np.array(txts).astype(str)
        indeces = np.array(scores_indces)

        return txts[indeces]

    def _extract_boxes(self, outputs:list, scores_indces:Optional[list])-> list:
        boxes = [line['position'] for line in outputs]

        boxes = np.array(boxes)
        indeces = np.array(scores_indces)

        return boxes[indeces]

    def _extract_scores(self, outputs:list, threshold:float = 0.5)-> tuple[list, list]: 
        scores = [line['score'] for line in outputs if line['score'] > threshold]

        scores, indences = [],[]

        for index, line in enumerate(outputs): 

            if line['score']  > threshold: 
                indences.append(index)
                scores.append(line['score'])


        return scores, indences
        
    def extract(self, img: np.ndarray, 
                     return_boxes: bool =False, 
                     return_scors:bool = False, 
                     return_texts:bool = True, 
                     score_threshold:float = 0.0):
        
        texts, boxes, scores = None, None, None

        output = self.ocr.ocr(img)

        thresh_score, indces = self._extract_scores(outputs=output, threshold=score_threshold)

        if return_scors: 
            scores = thresh_score

        if return_boxes: 
            boxes = self._extract_boxes(outputs=output, scores_indces=indces)

        if return_texts: 
            texts = self._extract_text(output, scores_indces=indces)

        return texts, boxes, scores
    
    def extract_text_paddleocr(self, image_path):
        cropped_img = Image.open(image_path)
        cropped_img_np = np.array(cropped_img) 

        result = self.ocr.ocr(cropped_img_np) 
        results = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                results.append(line[1][0])

        return " ".join(results)

if __name__ == "__main__":
    import os
    from image_handler import ImageHandler  

    image_path = "/home/azooz/mydisk/ocr_invoices/imgs/254.jpg"
    img = ImageHandler.read_img(image_path, return_numpy=True)

    print("=== Using CnOCR ===")
    cnocr_extractor = TextImgExtractor(engine="cnocr")
    texts, _, _ = cnocr_extractor.extract(img, score_threshold=0.5)
    print("Extracted Texts (CnOCR):", texts)

    print("\n=== Using PaddleOCR ===")
    paddle_extractor = TextImgExtractor(engine="paddleocr")
    paddle_text = paddle_extractor.extract_text_paddleocr(image_path)
    print("Extracted Text (PaddleOCR):", paddle_text)
