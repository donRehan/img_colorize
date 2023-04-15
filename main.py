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
image_index = int(input("Enter the index of the image: "))
image_path = "images/" + images[image_index]

prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = "images/" + input("Enter the name of the image: ")

"""
# Load the image
#image = cv2.imread(image_path)

#
#net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
#points = np.load(kernel_path)
"""
