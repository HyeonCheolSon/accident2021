import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys

############# global parameters #############
# path
train_path = './dataset/features/training/'
test_path = './dataset/features/testing/'
demo_path = './dataset/features/testing/'
default_model_path = './model/demo_model'
save_path = './model/'
video_path = './dataset/videos/testing/positive/'
# batch
train_num = 126
test_num = 46
#############################################

############# train parameters ##############
# Parameters
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)
n_detection = 20 # number of object of each image (include image features)
n_hidden = 512 # hidden layer num of LSTM
n_img_hidden = 256 # embedding image features 
n_att_hidden = 256 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video 
#############################################

def parse_args():
    # 