
import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('/efs/workspace/PiP-Planning-informed-Prediction/highD_Results/1_1/*.png'):
    img = cv2.imread(filename)

    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('highD.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()