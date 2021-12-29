import tensorflow as tf
import numpy as np
import math, random
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import *
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def LEFT_OF_U(channel,input,drop=False):
    if input.shape[-1] != 1:
        out = layers.MaxPooling2D(pool_size=(2, 2))(input)
    else:
        out = input
    out = layers.Conv2D(channel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(out)
    out = layers.Conv2D(channel, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out)
    if drop:
        out = layers.Dropout(drop)(out)
    return out



def RIGHT_OF_U(channel,input,prev_left):
    upsamp = layers.UpSampling2D(size=(2,2))(input)
    out = layers.Conv2D(channel,2,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upsamp)
    out = layers.concatenate([prev_left, out], axis = 3)
    out = layers.Conv2D(channel, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out)
    out = layers.Conv2D(channel, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out)
    return out

def build_UNET(channel_list,input_size):
    input = layers.Input(input_size)
    out_lst = []
    for i in range(len(channel_list)):
        if i == len(channel_list)-1:
            drop = 0.5
        else:
            drop = False
        if i == 0:
            output = LEFT_OF_U(channel_list[i],input,drop)
        else:
            output = LEFT_OF_U(channel_list[i],output,drop)
        if i != len(channel_list)-1:
            out_lst.append(output)
    for i in range(len(channel_list)-2,-1,-1):
        # print(out_lst[i].shape,output.shape)
        # if i != 0:
        output = RIGHT_OF_U(channel_list[i],output,out_lst[i])
    output = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(output)
    output = layers.Conv2D(1, 1, activation = 'sigmoid')(output)
    model = Model(inputs=input,outputs=output)
    return model

if __name__ == '__main__':
    channel_lst = [64,128,256,512,1024]
    model = build_UNET(channel_lst,(512,512,1))





