import sys
import os
# quick fix to specify the GPUs to use, comment out if there is no one using 100% of them
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Embedding, Dense, Dropout, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  #create model
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need model path.')
        sys.exit(0)

    model = load_model(sys.argv[1])
    layer_num = 0
    layer_types = []
    is_after_flatten = []
    layers_with_weights = []
    for idx,layer in enumerate(model.layers):
        if isinstance(layer,Conv2D) or isinstance(layer,Dense) or isinstance(layer,MaxPooling2D):
            layer_num += 1
            layers_with_weights.append(layer)
        if isinstance(layer,Dense):
            layer_types.append('0')
            if isinstance(model.layers[idx-1],Flatten):
                is_after_flatten.append(True)
            else:
                is_after_flatten.append(False)
        if isinstance(layer,Conv2D):
            layer_types.append('1')
            is_after_flatten.append(False)
        if isinstance(layer,MaxPooling2D):
            layer_types.append('2')
            is_after_flatten.append(False)

    num_inputs = np.prod([m for m in model.layers[0].input.shape if not m == None])
    num_outputs = np.prod([m for m in model.layers[-1].output.shape if not m == None])

    w = model.get_weights()

    input_sizes = [np.prod([m for m in l.input.shape if not m == None]) for l in model.layers if isinstance(l,Conv2D) or isinstance(l,Dense) or isinstance(l,MaxPooling2D)]
    max_input_size = max(input_sizes)


    print(layer_num,num_inputs,num_outputs,max_input_size,'',sep=',')
    # layer input sizes
    print(','.join(list(map(str,input_sizes)))+','+str(num_outputs)+',')

    print(','.join(layer_types)+',') 
    # for each conv layer: out_channel,in_channel,kernel,stride,padding
    for layer in model.layers:
        if isinstance(layer,Conv2D):
            oc = layer.output.shape[3]
            ic = layer.input.shape[3]
            ks = layer.kernel_size[0]
            st = layer.strides[0]
            if layer.padding == 'valid':
                pd = 0
            else:
                # this is wrong, need to work out actual value here
                pd = 1
            print(oc,ic,ks,st,pd,sep=',',end='')
            print(',')

    idx_counter = 0
    # structure of w is [w1,b1,w2,b2...]
    for wi in w:
        if len(wi.shape)==1:
            # bias
            for i in range(wi.shape[0]):
                print(str(wi[i])+",")
            idx_counter += 1
        if len(wi.shape)==2:
            # flatten or dense
            # if its the layer just after a flatten layer
            if is_after_flatten[idx_counter]:
                orig_shape = wi.shape
                # reshape to the output of the conv later
                conv_s = layers_with_weights[idx_counter-1].output.shape
                new_shape = (conv_s[1],conv_s[2],conv_s[3],orig_shape[1])
                wi = wi.reshape(*new_shape)
                wi = np.transpose(wi,(2,0,1,3))
                wi = wi.reshape(*orig_shape)
            '''
            if wi.shape==(128,2):
                wi = wi.reshape(2,2,32,2)
                wi = np.transpose(wi,(2,0,1,3))
                wi = wi.reshape(128,2)
            '''
            wi = wi.T
            for i in range(wi.shape[0]):
                for j in range(wi.shape[1]):
                    print(str(wi[i,j])+',',end="")
                print("")
        if len(wi.shape)==4:
            #conv layer
            for oc in range(wi.shape[3]):
                for ic in range(wi.shape[2]):
                    #wi[:,:,ic,oc] = wi[:,:,ic,oc].T
                    for w in range(wi.shape[1]):
                        for h in range(wi.shape[0]):
                            print(str(wi[w,h,ic,oc])+',',end="")
                print("")
