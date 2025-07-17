
from enum import Enum
from typing import Union, get_args
import os 
import sys

from utils.paper_detection import PaperDetector, PaperDetectionMethodType
from utils.remove_background import BackgroundRemover
from utils.llm_api import llm_api
from utils.image_handler import ImageHandler
from utils.text_extractor import TextImgExtractor
from utils.text_handler import TextHandler

import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image

class ExtractorPipline(Enum): 
    DetectPage = 1
    RemoveBackground = 2
    OCR = 3
    GEMINIApi = 4

    @property
    def method_class(self):
        mapping = {
            ExtractorPipline.DetectPage: PaperDetector, 
            ExtractorPipline.RemoveBackground: BackgroundRemover,
            ExtractorPipline.GEMINIApi: llm_api,
            ExtractorPipline.OCR: TextImgExtractor
        }
        return mapping[self]
        
    

class TextExtractor():
    def __init__(self):
        pass
    
    def get_info(self,*args):
        return ','.join(args)
    
        
class EverythingExtractor():
    def __init__(self):
        pass

    def extract(self, image: Union[Image.Image, np.ndarray], 
                enhancement_rate: float = 0.5, 
                score_threshold: float = 0.5,
                wanted_information: str = 'vendor name, total amount')-> dict:
        # Step 1: Detect the page
        paper_detected = ExtractorPipline.DetectPage.method_class(PaperDetectionMethodType(2))(img=image)
        
        # Step 2: Remove background
        enhanced_img = ExtractorPipline.RemoveBackground.method_class().run(paper_detected, enhancement_rate)

        # Step 3: OCR
        text, _, _ = ExtractorPipline.OCR.method_class().extract(enhanced_img, score_threshold)

        text = ' '.join(text)
        
        # Step 4: Call GEMINIApi
        extract_json = ExtractorPipline.GEMINIApi.method_class(contents=text, wanted_information=wanted_information)
        
        return extract_json

if __name__ == "__main__": 

    image = "/home/azooz/mydisk/ocr_invoices/imgs/254.jpg"
    image = ImageHandler.read_img(image, return_numpy=True)

    extract_everything = EverythingExtractor()
    extract_json = extract_everything.extract(image, enhancement_rate=0.5, score_threshold=0.5, wanted_information='vendor name, total amount')

    print(TextHandler.parse_json2dict(extract_json))