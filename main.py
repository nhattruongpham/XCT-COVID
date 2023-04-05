#!/usr/bin/env python
# coding: utf-8
import os, sys, random
import tensorflow as tf
import numpy as np
import keras
import cv2
import pandas as pd
import keras.backend as K
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import pickle
import math
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import splitfolders

from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# seed = random.randrange(150)
seed = 0
print("Seed was:", seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

currentSecond = datetime.now().second
currentMinute = datetime.now().minute
currentHour = datetime.now().hour

currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                         [tf.config.experimental.VirtualDeviceConfiguration(
#                                                             memory_limit=24576)])

NUM_CLASSES = 1
TRAIN_DATA_PATH = 'data/CT-COVID'
# splitfolders.ratio(TRAIN_DATA_PATH, output='COVID-19-CT/', seed=1234, ratio=(0.85, 0.15), group_prefix=None)
IND_TEST_DATA_PATH = 'data/CT-COVID'

# Preparing dataset
X_train = []
for folder in os.listdir(TRAIN_DATA_PATH):
    sub_path = TRAIN_DATA_PATH + "/" + folder
    for img in os.listdir(sub_path):
        img_path = sub_path + "/" + img
        img_arr = cv2.imread(img_path)
        resized_img = cv2.resize(img_arr, (224, 224))
        X_train.append(resized_img)

print(np.array(X_train).shape)

train_X = np.array(X_train)

train_X = train_X / 255.0

X_ind_test = []
for folder in os.listdir(IND_TEST_DATA_PATH):
    sub_path = IND_TEST_DATA_PATH + "/" + folder
    for img in os.listdir(sub_path):
        img_path = sub_path + "/" + img
        img_arr = cv2.imread(img_path)
        resized_img = cv2.resize(img_arr, (224, 224))
        X_ind_test.append(resized_img)

print(np.array(X_ind_test).shape)
  
test_ind_X = np.array(X_ind_test)

test_ind_X = test_ind_X / 255.0

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
ind_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(TRAIN_DATA_PATH,
                                                 target_size=(224, 224),
                                                 class_mode='binary')

ind_test_set = ind_test_datagen.flow_from_directory(IND_TEST_DATA_PATH,
                                                    target_size=(224, 224),
                                                    class_mode='binary')                                             

train_y = training_set.classes
test_y = ind_test_set.classes

print(training_set.class_indices)

print(train_y.shape)


print(ind_test_set.class_indices)

print(test_y.shape)


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(test_Y, pre_test_y):
    #calculate the F1-score
    Precision = precision(test_Y, pre_test_y)
    Recall = recall(test_Y, pre_test_y)
    f1 = 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))
    return f1 

def TP(test_Y,pre_test_y):
    #calculate numbers of true positive samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    return TP

def FN(test_Y,pre_test_y):
     #calculate numbers of false negative samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    P=K.sum(K.round(K.clip(test_Y, 0, 1)))
    FN = P-TP #FN=P-TP
    return FN

def TN(test_Y,pre_test_y):
    #calculate numbers of True negative samples
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    return TN

def FP(test_Y,pre_test_y):
    #calculate numbers of False positive samples
    N = (-1)*K.sum(K.round(K.clip(test_Y-K.ones_like(test_Y), -1, 0)))#N
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    FP=N-TN
    return FP

def EffNetV2(num_classes):
    backbone = tf.keras.applications.VGG19(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False
    )
    backbone.trainable = False
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    # scale_layer = tf.keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # x = scale_layer(inputs)
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation = 'LeakyReLU')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation = 'LeakyReLU')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model

mcc_trn_scores = []
sn_trn_scores = []
sp_trn_scores = []
acc_trn_scores = []
pcs_trn_scores = []
roc_auc_trn_scores = []

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_number = 0
for train_idx, test_idx in kfold.split(train_X, train_y):
    fold_number = fold_number + 1
    print("Training Fold: ", fold_number)
    train_X1, test_X1, train_y1, test_y1 = train_X[train_idx], train_X[test_idx], train_y[train_idx], train_y[test_idx]
    print("Number of testing KF: ", len(test_y1))
    created_model = EffNetV2(num_classes=NUM_CLASSES)
    created_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', precision,recall,f1,TP,FN,TN,FP])
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    callback2 = tf.keras.callbacks.ReduceLROnPlateau(
                               patience = 5, monitor = 'val_accuracy', 
                               mode = 'max', factor = 0.1
                           )
    created_model.fit(train_X1, train_y1, epochs=200, batch_size=32, validation_data=(test_X1, test_y1), callbacks=[callback1, callback2])
    y_pred = (created_model.predict(test_X1)>0.5).astype(np.float32)
    tn, fp, fn, tp = confusion_matrix(test_y1, y_pred).ravel()
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    mcc_trn_scores.append(mcc)
    print('MCC: {0:.4f}'.format(mcc))
    sn = float(tp) / (tp + fn + 1e-06)
    sn_trn_scores.append(sn)
    print('Sn: {0:.4f}'.format(sn))
    sp = float(tn) / (tn + fp + 1e-06)
    sp_trn_scores.append(sp)
    print('Sp: {0:.4f}'.format(sp))
    acc = float(tp + tn) / (tn + fp + fn + tp + 1e-06)
    acc_trn_scores.append(acc)
    print('ACC: {0:.4f}'.format(acc))
    pcs = float(tp) / (tp + fp + 1e-06)
    pcs_trn_scores.append(pcs)
    print('Precision: {0:.4f}'.format(pcs))
    roc_auc = roc_auc_score(test_y1, y_pred)
    roc_auc_trn_scores.append(roc_auc)
    print('AUC: {0:.4f}'.format(roc_auc))
    
print("*"*80)
print('Mean MCC Training: {0:.4f}'.format(np.mean(mcc_trn_scores)))
print('Mean Sn Training: {0:.4f}'.format(np.mean(sn_trn_scores)))
print('Mean Sp Training: {0:.4f}'.format(np.mean(sp_trn_scores)))
print('Mean ACC Training: {0:.4f}'.format(np.mean(acc_trn_scores)))
print('Mean Precision Training: {0:.4f}'.format(np.mean(pcs_trn_scores)))
print('Mean AUC Training: {0:.4f}'.format(np.mean(roc_auc_trn_scores)))

model = EffNetV2(num_classes=NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', precision,recall,f1,TP,FN,TN,FP])
callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
callback2 = tf.keras.callbacks.ReduceLROnPlateau(
                               patience = 5, monitor = 'val_accuracy', 
                               mode = 'max', factor = 0.1
                           )
train_X1, test_X1, train_y1, test_y1 = train_X, test_ind_X, train_y, test_y
model.fit(train_X1, train_y1, epochs=200, batch_size=32, validation_data=(test_X1, test_y1), callbacks=[callback1, callback2])

model.save('CT_Scan_pretrained_covid_ct.h5')

model.evaluate(test_X1, test_y1)

y_pred = (model.predict(test_X1)>0.5).astype(np.float32)

#get classification report
print(classification_report(test_y1,y_pred))

#get confusion matrix

print(confusion_matrix(test_y1,y_pred))

tn, fp, fn, tp = confusion_matrix(test_y1, y_pred).ravel()
mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
print('MCC: {0:.4f}'.format(mcc))
sn = float(tp) / (tp + fn + 1e-06)
print('Sn: {0:.4f}'.format(sn))
sp = float(tn) / (tn + fp + 1e-06)
print('Sp: {0:.4f}'.format(sp))
acc = float(tp + tn) / (tn + fp + fn + tp + 1e-06)
print('ACC: {0:.4f}'.format(acc))
pcs = float(tp) / (tp + fp + 1e-06)
print('Precision: {0:.4f}'.format(pcs))
roc_auc = roc_auc_score(test_y1, y_pred)
print('AUC: {0:.4f}'.format(roc_auc))
