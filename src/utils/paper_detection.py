
from utils.image_handler import ImageHandler

from typing import Union, get_args
from enum import Enum
import cv2
import os
import PIL
import numpy as np 
from PIL import Image

import torch
from torch import nn
from torchvision import transforms

import onnx
import onnxruntime as ort

from config.config import FilePaths

file_paths = FilePaths()

class PaperDetectionMethodType(Enum):
    tranditional = 1
    seg_mblv3 = 2


    @staticmethod
    def dictionarize(method_type): 
        methods = {
            PaperDetectionMethodType.tranditional: TranditionalMethod,
            PaperDetectionMethodType.seg_mblv3: SegMethod
        }
        return methods[method_type]

class TranditionalMethod():
    def __init__(self):
        pass
    
    def inference(self, img: Union[Image.Image, np.ndarray]):
        self.processed_img = ImageHandler.close_morphology(img)
        self.processed_img = ImageHandler.guassian_blur(self.processed_img)
        self.processed_img = ImageHandler.edge_detection(self.processed_img)
        self.processed_img = ImageHandler.dialation(self.processed_img)

        return self.processed_img
        

class SegMethod():

    
    def __init__(self, model_path: str = file_paths.paper_seg_model):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        
    def _preprocessing(self, img: PIL.Image):
        
        img_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return img_transform(img).unsqueeze(0).numpy()
    
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def inference(self, org_img: np.ndarray):

        org_img = Image.fromarray(org_img) 

        img = self._preprocessing(org_img)
        outputs = self.session.run(None, {self.input_name: img})[0]

        return self._process_output(torch.tensor(outputs)[0], org_img)
        # return  self.preprocess_mask(outputs, (org_img.width, org_img.height))
    
    def preprocess_mask(self,outputs, size):
        outputs = self._sigmoid(outputs)[0]
        outputs = (np.transpose(outputs, (2, 1,0))[:,:,1] * 255).astype(np.uint8)
        mask = cv2.resize(outputs, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
        # mask_3ch = np.stack([mask] * 3, axis=-1)

        return mask
    
    def _process_output(self, output, original_image):

        output = output.argmax(0).byte().numpy()  # Convert to binary mask
        mask = cv2.resize(output, (original_image.width, original_image.height), interpolation=cv2.INTER_NEAREST)
        
        original_np = np.array(original_image, dtype=np.uint8)

        # Find bounding box of foreground
        coords = np.column_stack(np.where(mask > 0))
        if coords.size == 0:
            return Image.fromarray(original_np)  # Return original if no foreground found
        
        return mask


class PostProcess():    

    def _process(self, bn_img:np.ndarray):
        self.contours = ImageHandler.contour(bn_img)
        self.page = ImageHandler.get_page(self.contours)
        corners = ImageHandler.get_corners(self.page)
        corners = ImageHandler.reorder_rectangle_pts(corners)
        distances = ImageHandler.get_max_dist(corners)
        return distances, corners

    def run(self,org_img, bn_img):
        distances, corners = self._process(bn_img)
        pre_img = ImageHandler.perspective(org_img, corners, distances)

        return pre_img
    
class PaperDetector(PostProcess):
    def __init__(self, method_type: PaperDetectionMethodType):
        super().__init__()
        self.method_type = method_type

    def __call__(self, img: Union[Image.Image, np.ndarray, str]):

        if self.method_type.value == PaperDetectionMethodType.tranditional.value:
            img = ImageHandler.read_img(img, return_numpy=True, space_color='L')
        else: 
            img = ImageHandler.read_img(img, return_numpy=True)

        inference_method = PaperDetectionMethodType.dictionarize(self.method_type)()

        bn_img = inference_method.inference(img)

        pre_img = self.run(img,bn_img)

        return pre_img  
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    segmentation = SegMethod()
    post_process = PostProcess()
    detection_paper = PaperDetector(PaperDetectionMethodType(1))
    image_types = detection_paper.__call__.__annotations__['img']

    img = PIL.Image.open("/home/azooz/mydisk/ocr_invoices/test3.jpeg")

    pre_img = detection_paper(img)
    # bn_img = segmentation.inference(img)
    
    # img = np.array(img)
    # pre_img = post_process.run(img,bn_img)


    plt.imshow(pre_img)
    plt.show()