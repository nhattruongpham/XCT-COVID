import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import splitfolders

def load_labels(label_file):
    """Loads image filenames, classes, and bounding boxes"""
    fnames, classes, bboxes = [], [], []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
            fnames.append(fname)
            classes.append(int(cls))
            bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
    return fnames, classes, bboxes

# Set paths
image_dir = '/home/nhattruongpham/CBBL_SKKU_Projs/Datasets/COVIDx-CT-3/3A_images'
label_file = '/home/nhattruongpham/CBBL_SKKU_Projs/Datasets/COVIDx-CT-3/val_COVIDx_CT-3A.txt'
data_dir = '/home/nhattruongpham/CBBL_SKKU_Projs/COVID-19-CT/data/COVIDx-CT-3'
splitfolders.ratio(data_dir, output= data_dir + '/', seed=1234, ratio=(0.70, 0.15, 0.15), group_prefix=None)

# # Load labels
# fnames, classes, bboxes = load_labels(label_file)
# class_names = ('Normal', 'Pneumonia', 'COVID-19')
# print(fnames[0])
# print(class_names[classes[0]])
# print(bboxes[0])

# # cnt = 0
# for i in range(len(fnames)):
#     # cnt = cnt + 1
#     iname = fnames[i]
#     print(iname)
#     ifile = os.path.join(image_dir, iname)
#     # print(ifile)
#     img = cv2.imread(ifile)
#     # plt.figure(1)
#     # plt.imshow(img)
#     xmin, ymin, xmax, ymax = bboxes[i]
#     cropped = img[ymin:ymax, xmin:xmax]
#     # plt.figure(2)
#     # plt.imshow(cropped)
#     label = class_names[classes[i]]
#     print(label)
#     if label == 'Normal' or label == 'Pneumonia':
#         cv2.imwrite(data_dir + '/' + 'non-COVID' + '/' + iname, cropped)
#     if label == 'COVID-19':
#         cv2.imwrite(data_dir + '/' + 'COVID' + '/' + iname, cropped)
#     # plt.show()
#     # if cnt == 1:
#     #     break

# # Select cases to view
# np.random.seed(14)
# indices = np.random.choice(list(range(len(fnames))), 9)

# # Show a grid of 9 images
# fig, axes = plt.subplots(3, 3, figsize=(16, 16))
# class_names = ('Normal', 'Pneumonia', 'COVID-19')
# for index, ax in zip(indices, axes.ravel()):
#     # Load the CT image
#     image_file = os.path.join(image_dir, fnames[index])
#     image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

#     # Overlay the bounding box
#     image = np.stack([image]*3, axis=-1)  # make image 3-channel
#     bbox = bboxes[index]
#     cv2.rectangle(image, bbox[:2], bbox[2:], color=(255, 0, 0), thickness=3)

#     # Display
#     cls = classes[index]
# #     plt.figure()
#     ax.imshow(image)
#     ax.set_title('Class: {} ({})'.format(class_names[cls], cls))
# plt.show()


