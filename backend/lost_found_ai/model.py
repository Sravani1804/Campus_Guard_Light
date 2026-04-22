import cv2
import numpy as np
from PIL import Image

def extract_features(image: Image.Image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=1200)
    kp, des = orb.detectAndCompute(gray, None)

    return kp, des
