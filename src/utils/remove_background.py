import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.image_handler import ImageHandler

from typing import Optional, Union

CONSTANT_SHARPING_RATIO = 10

class BackgroundRemover():
    def __init__(self):
        self.kernel = np.array([[0, -1, 0], 
                        [-1, 5,-1], 
                        [0, -1, 0]])
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        self.guassin_blur_kenel = (5,5)

    def run(self, img: Union[str, np.ndarray], enahncing_ratio: float = 0)-> np.ndarray:

        if enahncing_ratio > 0: 
            self.kernel[1,1] = int(enahncing_ratio*CONSTANT_SHARPING_RATIO)

        if isinstance(img, str):
            img = ImageHandler.read_img(img, return_numpy=True)

        img = cv2.GaussianBlur(img, self.guassin_blur_kenel, 0)
        _, img = ImageHandler.remove_shodow(img)

        img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, self.rectKernel)
        img = cv2.filter2D(img, -1, self.kernel)
        img = cv2.bitwise_not(img)
        # img = cv2.GaussianBlur(img, self.guassin_blur_kenel, 0)
        return img

    

if __name__ == "__main__":
    remover = BackgroundRemover()
    img = remover.run("/home/azooz/mydisk/ocr_invoices/imgs/test1.jpg")
    img1 = ImageHandler.read_img("/home/azooz/mydisk/ocr_invoices/imgs/test1.jpg", return_numpy=True)

    ImageHandler.plot_stack_imgs(img1, img)
    plt.imshow(img, cmap='gray')
    plt.show()

    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))

    # rectKerneld = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 7))
    # sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))


    # # Inpaint to remove spots
    
        
    #     # Apply the sharpening kernel to the binary image
    #     sharpened = cv2.filter2D(blackhat, -1, kernel)
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     # morphed = cv2.morphologyEx(blackhat, cv2.MORPH_CLOSE, kernel, iterations=4)

    #     # img = cv2.dilate(sharpened, (11,11), iterations=2)
    #     image = cv2.bitwise_not(sharpened)
    #     blurred = cv2.GaussianBlur(image, (5, 5), 0)    

    #     cv2.imwrite('output2.jpg', blurred)    
    #     # plot_imgs(blurred, blackhat, img, sharpened)

    # remove_background("/home/azooz/mydisk/ocr_invoices/imgs/254.jpg", 'output.jpg')