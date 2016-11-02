import numpy as np
import os
import urllib
import json
import cv2

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
            base_path=os.path.abspath(os.path.dirname(__file__)))

FACE_DETECTOR_PATH = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"


def grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        print("Loading " + path)
        image = cv2.imread(path)
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        # We are ignoring streaming right now.
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return image
    return image

def find_faces(image):
    print FACE_DETECTOR_PATH
    # convert the image to grayscale, load the face cascade detector,
    # and detect faces in the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                                 
    # construct a list of bounding boxes from the detection
    rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
                                                 
    # update the data dictionary with the faces detected
    data ={"num_faces": len(rects), "faces": rects, "success": True}
    print data

image = grab_image(path='/home/tkulish/life-gather/sageviewer/test-images/obama.jpg')
find_faces(image)
image = grab_image(path='/home/tkulish/life-gather/sageviewer/test-images/multiple-faces.jpg')
find_faces(image)

