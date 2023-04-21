#! /usr/local/bin/python3
import numpy as np
import os
import cv2

"""
Inspired by https://www.youtube.com/watch?v=oNjQpq8QuAo&t=588s
Paths to the files

I want to have a menu that shows the available images names in the stdoutput and make the user choose one of them
"""


""" Show the images in the directory """
images = os.listdir("images")
print(images)


""" Make the user select the image by its index of that list """

print("Accessing elements is done by thier index , starting from 0")
print(" ")
print("left to right , starting from 0")
image_index = int(input("Enter the index of the image: "))
image_path = "images/" + images[image_index]

prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"

image = cv2.imread(image_path)
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
pts = np.load(kernel_path)

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

norm_image = image.astype("float32") / 255.0
lab_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab_image, (224, 224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab_image)[0]

colorized_image = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)
colorized_image = (255 * colorized_image).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()

cv2.imwrite("output/colorized_image.jpg", colorized_image)
