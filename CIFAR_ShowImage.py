import numpy as np
import cv2
from matplotlib import pyplot as plt


CIFAR_DIR = "/users/hashu/Desktop/Computer_Vision/SVM_Image_Classification/cifar-10-batches-py/"



def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        cifar_dict = pickle.load(fo)
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4', 'data_batch_5', 'test_batch']

all_data = [0,1,2,3,4,5,6]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta =  all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

label_names = batch_meta["label_names"]
#print(label_names)
##getting an image from data_batch1

image1 = data_batch1["data"]
labels1 = data_batch1["labels"]

image1 = np.reshape(image1,(10000,3,32,32))
labels1 = np.array(labels1)
#print(image1.shape) #10000 pics of 3, 32, 32....but opencv needs 32x32x3
image_number = 10
tester = image1[image_number].transpose(1,2,0)
#array is RGB, cv2 requires BGR
tester = cv2.cvtColor(tester,cv2.COLOR_RGB2BGR)
#print(labels1.shape)
name = labels1[image_number]

print(label_names[name])
plt.imshow(tester, interpolation = 'nearest')
plt.show()
