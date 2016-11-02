# http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html
import numpy as np
import os
import urllib
import json
import cv2

# define the path to the face detector
FACE_DETECTOR_PATH = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH);
#recognizer = cv2.createLBPHFaceRecognizer()

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

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]

    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        # Get the label of the image
        #nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        nbr = os.path.split(image_path)[1].split(".")[0]
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            #cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

images, labels = get_images_and_labels('/home/tkulish/life-gather/sageviewer/train-images')
print labels

# Train the model using a dataset in 
