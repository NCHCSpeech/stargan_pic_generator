'''
Sources:
http://opencv.willowgarage.com/documentation/python/cookbook.html
http://www.lucaamore.com/?p=638
'''
import cv2 as cv
from PIL import Image
import glob
import os
import sys

def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    #variables    
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    # Turn image to gray scale
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # Equalize the histogram
    cv.equalizeHist(image, image)

    # Detect the faces
    faces = faceCascade.detectMultiScale(
            image, haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces is not None and returnImage is not None:
        for ((x, y, w, h)) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
    #cv.rectangle(image, pt1, pt2, (255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    cv_im = cv.cvtColor(pil_im, cv.COLOR_BGR2GRAY)
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box),PIL_box

def faceCrop(img,boxScale=1.5):
    print("img:",img)
    faceCascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

    cv_im = cv.imread(img)
    faces=DetectFace(cv_im,faceCascade)
    print(faces)
    pil_im = Image.open(img)
    result = []
    if faces is not None:
        n=1
        for face in faces:
            print(face)
            croppedImage, PIL_box=imgCrop(pil_im, face,boxScale=boxScale)
            fname,ext=os.path.splitext(img)
            croppedImage.save(fname+'_crop'+str(n)+ext)
            print("PIL_box:",PIL_box)
            result.append(([int(i) for i in PIL_box],croppedImage))
            n+=1
    else:
        print ('No faces found:', img)
    print("result:",result)
    return result


