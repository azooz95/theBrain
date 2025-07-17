

from functools import wraps
from typing import Union,get_args

from PIL import Image
import numpy as np 

def validate_input(converted_type=np.ndarray, allowed_type=Union[Image.Image, np.ndarray, str]):
    def decorator(func):
        annotations = func.____annotations__
        image_argments 
        for arg_annotaion in annotations: 
            if 'img' not in arg_annotaion: 
                raise ValueError(f'Value Error, this function has no image')
            
        @wraps(func)
        def wrapper(*args, **kaw):
            if not isinstance(value, get_args(allowed_type)):
                raise TypeError(f"Expected {allowed_type}, got {type(value).__name__}")
            
            if isinstance(value, Image.Image):
                value
            return func(value)
        return wrapper
    return decorator
