

from typing import Union

from PIL import Image
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

from typing import Union

KERNEL = np.ones((5,5),np.uint8)
BLURING_KERNEL = (11,11)
DIALATION_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
EDGE_DETECTION_THRESHOLD = (0,200)

class ImageHandler():
    
    @staticmethod
    def read_img(image:Union[Image.Image,str] ,return_numpy:bool = False, space_color='RGB'):

        if isinstance(image, np.ndarray): 
            image = Image.fromarray(image).convert(space_color)
        elif isinstance(image, str):
            image = Image.open(image).convert(space_color)

        if return_numpy: 
            image = np.array(image)
        
        return image

    @staticmethod
    def close_morphology(img: np.ndarray, iterations=4): 
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, KERNEL, iterations= iterations)
        return img
    
    @staticmethod
    def guassian_blur(img: np.ndarray):
        return cv.GaussianBlur(img, BLURING_KERNEL , 0)

    @staticmethod 
    def edge_detection(img: np.ndarray): 
        return cv.Canny(img, EDGE_DETECTION_THRESHOLD[0], EDGE_DETECTION_THRESHOLD[1])
    
    @staticmethod
    def dialation(img:np.ndarray):
        return cv.dilate(img,DIALATION_KERNEL)

    @staticmethod
    def contour(img:np.ndarray):
        contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        return contours, hierarchy

    @staticmethod
    def get_page(contours: np.ndarray): 
        page = sorted(contours[0], key=cv.contourArea, reverse=True)[:5]

        if not page:
            raise ValueError("No page found")
        
        return page
        
    @staticmethod
    def get_corners(page: np.ndarray):
        for c in page:
            epsilon = 0.02 * cv.arcLength(c, True)
            corners = cv.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break
        return np.concatenate(corners).tolist()

    @staticmethod
    def reorder_rectangle_pts(pts):
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)

        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]
    
        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates.
        return rect.astype('int').tolist()

    @staticmethod
    def convert_xyxy2pts(xyxy):
        x_min, y_min, x_max, y_max = xyxy
        return np.array([
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_max, y_max],  # Bottom-right
            [x_min, y_max],  # Bottom-left
        ], dtype=np.float32)
    
    @staticmethod
    def get_img_conrner(img: Union[Image.Image, np.ndarray]):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        y,x = img.shape[:2]

        return np.array([
            [0,0],
            [x, 0],
            [0, y],
            [x, y]
        ],dtype=np.float32)

    @staticmethod
    def get_max_dist(pts):
        (tl, tr, br, bl) = pts
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

        return destination_corners
    
    @staticmethod
    def perspective(img, corners, destinace):
        M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destinace))
        perspective_img = cv.warpPerspective(img, M, (destinace[2][0], destinace[2][1]), flags=cv.INTER_LINEAR)
        return perspective_img
    
    @staticmethod
    def remove_shodow(image):
        rgb_planes = cv.split(image)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
            
        result = cv.merge(result_planes)
        result_norm = cv.merge(result_norm_planes)

        return result, result_norm
    
    @staticmethod
    def plot_stack_imgs(*imgs):
        stacked_imgs = np.hstack(imgs)
        plt.imshow(stacked_imgs)
        plt.show()

    @staticmethod
    def write_on_img(image, 
                        text, 
                        position=(50, 50), 
                        font_scale=1, 
                        color=(0, 255, 0), 
                        thickness=2):

        font = cv.FONT_HERSHEY_SIMPLEX

        cv.putText(image, text, position, font, font_scale, color, thickness)

        return image


if __name__ == "__main__":
    pass 
    # import matplotlib.pyplot as plt 
    
    # img = "../assests/images/Original-Form-1.jpg"
    # pipline = SeqPipline()
    # pipline.add_step(ImageHandler.read_img, return_numpy=True, space_color='L', is_return=True)
    # pipline.add_step(ImageHandler.close_morphology)
    # pipline.add_step(ImageHandler.guassian_blur)
    # pipline.add_step(ImageHandler.edge_detection)
    # pipline.add_step(ImageHandler.dialation)
    # pipline.add_step(ImageHandler.contour)
    # pipline.add_step(ImageHandler.get_page)
    # pipline.add_step(ImageHandler.get_corners, is_return=True)
    # pipline.add_step(ImageHandler.reorder_rectangle_pts)
    # pipline.add_step(ImageHandler.get_max_dist, is_return=True)
    # outoput = pipline.run(img)
    # org_img, corners, distances = outoput
    # pre_img = ImageHandler.perspective(org_img, corners, distances)

    # org_img = ImageHandler.read_img(img, return_numpy=True, space_color='L')
    # con = np.zeros_like(org_img)
    # img = ImageHandler.close_morphology(org_img)
    # img = ImageHandler.guassian_blur(img)
    # img = ImageHandler.edge_detection(img)
    # img = ImageHandler.dialation(img)
    # countors = ImageHandler.contour(img)
    # page = ImageHandler.get_page(countors)
    # corners = ImageHandler.get_corners(page)
    # rec_pts = ImageHandler.reorder_rectangle_pts(corners)
    # distances = ImageHandler.get_max_dist(rec_pts)

    # pre_img = cv.resize(pre_img, (org_img.shape[1], org_img.shape[0]))
    # stacked_img = np.hstack([org_img, pre_img])
    # plt.imshow(stacked_img)
    # plt.show()
    # print(img.shape)
