from flask import request
import base64

import numpy as np
import cv2
# from PIL import Image
from io import BytesIO

class Minic(object):
    @staticmethod
    def receive():
        jpegrdata = request.form.get('img')
        return "success"
    
    @staticmethod
    def base64Decode():
        jpegrdata = request.form.get('img')
        jpegdata = base64.b64decode(jpegrdata)
        return "success"

    @staticmethod
    def dump():
        jpegrdata = request.form.get('img')
        jpegdata = base64.b64decode(jpegrdata)

        with open("../dump.jpg", 'wb') as f:
            f.write(jpegdata)

        return "success"
    
    @staticmethod
    def readByOpenCV():
        jpegrdata = request.form.get('img')
        jpegdata = base64.b64decode(jpegrdata)

        img_array = np.fromstring(jpegdata, np.uint8)
        src_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return str(src_img.shape)
    
    @staticmethod
    def readByPIL():
        jpegrdata = request.form.get('img')
        jpegdata = base64.b64decode(jpegrdata)

        jpegdata = BytesIO(jpegdata)
        img = Image.open(jpegdata)
        img_array = np.asarray(img)
        return str(img_array.shape)        
        
