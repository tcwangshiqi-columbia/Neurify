import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation
import sys

(x_t,y_t),(_,_) = keras.datasets.mnist.load_data()

x = x_t[3000]/255
y = y_t[3000]

with open('models/test.nnet') as f:
    contents = f.readlines()

contents = [c.strip() for c in contents if not c.startswith('//')]
il = list(map(int,contents.pop(0).split(',')[:-1]))
layers = il[0]
inputs = il[1]
outputs = il[2]
max_layersize = il[3]

# flattened layer sizes starting from input
layer_sizes = list(map(int,contents.pop(0).split(',')[:-1]))

layer_types = list(map(int,contents.pop(0).split(',')[:-1]))

# conv layer 1 info
c1i = list(map(int,contents.pop(0).split(',')[:-1]))
# out channel, in channel, kernel, stride, padding
c1_oc = c1i[0]
c1_ic = c1i[1]
c1_k = c1i[2]
c1_s = c1i[3]
c1_p = c1i[4]

# conv layer 2 info
c2i = list(map(int,contents.pop(0).split(',')[:-1]))
# out channel, in channel, kernel, stride, padding
c2_oc = c2i[0]
c2_ic = c2i[1]
c2_k = c2i[2]
c2_s = c2i[3]
c2_p = c2i[4]

# now read in weights
weights = []
biases = []
for i in range(4):
    if i == 0:
        rows = c1_oc
    elif i == 1:
        rows = c2_oc
    else:
        # add one since layer_sizes includes input
        rows = layer_sizes[i+1]
    weights.append([])
    bias = []
    for r in range(rows):
        weights[i].append(list(map(float,contents.pop(0).split(',')[:-1])))
    for r in range(rows):
        bias.append(float(contents.pop(0)[:-1]))
    bias = np.array(bias)
    biases.append(bias)

weights = np.array(weights)
biases = np.array(biases)

# everything is read in, now try using it
model = Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Conv2D(c1_oc,c1_k,c1_s,"same",activation='relu'))
model.add(Conv2D(c2_oc,c2_k,c2_s,"same",activation='relu'))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy')

new_weights = []
for i in range(5):
    # layer index 2 is the flatten layer, no weights there
    if not i == 2:
        new_weights.append([])
        w_s = model.layers[i].get_weights()[0].shape
        # shape in keras is (k_size, k_size, in_channel, out_channel)
        print("Target shape:",w_s)
        wi = i if i <2 else i-1
        wm = np.array(weights[wi])
        tmp = []
        print("Nnet shape:",wm.shape)
        if i == 0 or i == 1:
            new_shape = (w_s[3],w_s[2],w_s[0],w_s[1])
            # im 99% sure this is correct here
            # nnet format is (out_channel,in_channel,k_size,k_size)
            print("Reshaping weights in conv layer to (out_channel,in_channels,k_size,k_size).")
            wm = wm.reshape(new_shape)
            print("Transposing to order: (k_size,k_size,in_channels,out_channels)")
            wm = np.transpose(wm,(2,3,1,0))
            print("Nnet new shape:",wm.shape)
        else:
            wm = wm.T
            print("Nnet new shape:",wm.shape)
            if i == 3:
                print('Doing some stuff they do just after a flatten layer but I do not know why.')
                # what they do is reshape it, transpose, then reshape it back
                # it was output from: (x, 7, 7, 32)
                print("Reshaping to prev conv output size from nnet.")
                wm = wm.reshape(32,7,7,100)
                print("Transposing to (k_size,k_size,in_channel,out_channel).")
                wm = np.transpose(wm,(1,2,0,3))
                print("Reshaping back to intended output size.")
                wm = wm.reshape(1568,100)
            print("Transposing fc layer.")
        new_weights[wi] = wm

        print()

model.layers[0].set_weights([new_weights[0],biases[0]])
model.layers[1].set_weights([new_weights[1],biases[1]])
model.layers[3].set_weights([new_weights[2],biases[2]])
model.layers[4].set_weights([new_weights[3],biases[3]])

x = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09803921568627451,0.3568627450980392,0.6823529411764706,0.996078431372549,0.996078431372549,1.0,0.996078431372549,0.611764705882353,0.10588235294117647,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.03529411764705882,0.32941176470588235,0.8313725490196079,0.996078431372549,0.996078431372549,0.8784313725490196,0.8470588235294118,0.8470588235294118,0.8588235294117647,0.996078431372549,0.403921568627451,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.15294117647058825,0.6745098039215687,0.996078431372549,0.9803921568627451,0.6313725490196078,0.2196078431372549,0.050980392156862744,0.0,0.0,0.22745098039215686,0.996078431372549,0.6901960784313725,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.27058823529411763,0.8980392156862745,0.996078431372549,0.8509803921568627,0.3058823529411765,0.0,0.0,0.0,0.0,0.0,0.6901960784313725,0.996078431372549,0.4196078431372549,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.17254901960784313,0.9137254901960784,0.996078431372549,0.5137254901960784,0.0392156862745098,0.0,0.0,0.0,0.0,0.0,0.0,0.8117647058823529,0.807843137254902,0.01568627450980392,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7176470588235294,0.996078431372549,0.7215686274509804,0.00784313725490196,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6470588235294118,0.9333333333333333,0.4980392156862745,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3333333333333333,0.9725490196078431,0.9686274509803922,0.3058823529411765,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5647058823529412,0.996078431372549,0.7725490196078432,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.011764705882352941,0.7372549019607844,0.996078431372549,0.4823529411764706,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4,0.9882352941176471,0.996078431372549,0.33725490196078434,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5294117647058824,0.996078431372549,0.6588235294117647,0.027450980392156862,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.03529411764705882,0.7411764705882353,0.996078431372549,0.7098039215686275,0.00784313725490196,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7764705882352941,0.996078431372549,0.3058823529411765,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01568627450980392,0.6941176470588235,0.996078431372549,0.996078431372549,0.6627450980392157,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.09411764705882353,0.9176470588235294,0.996078431372549,0.07450980392156863,0.0,0.0,0.0,0.0,0.0,0.0,0.0392156862745098,0.6980392156862745,0.996078431372549,1.0,0.996078431372549,0.5686274509803921,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.07450980392156863,0.8823529411764706,0.996078431372549,0.48627450980392156,0.08235294117647059,0.0,0.0,0.0196078431372549,0.14901960784313725,0.5176470588235295,0.8627450980392157,0.9882352941176471,0.6,0.9568627450980393,0.984313725490196,0.25098039215686274,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4745098039215686,0.9803921568627451,0.996078431372549,0.8980392156862745,0.7764705882352941,0.7764705882352941,0.8,0.996078431372549,0.996078431372549,0.9607843137254902,0.3764705882352941,0.13725490196078433,0.9529411764705882,0.7686274509803922,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3333333333333333,0.8627450980392157,0.996078431372549,0.996078431372549,0.996078431372549,0.7411764705882353,0.5607843137254902,0.3333333333333333,0.0392156862745098,0.0,0.6627450980392157,0.996078431372549,0.5490196078431373,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.043137254901960784,0.07058823529411765,0.07058823529411765,0.07058823529411765,0.011764705882352941,0.0,0.0,0.0,0.023529411764705882,0.8470588235294118,0.996078431372549,0.20784313725490197,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3411764705882353,0.996078431372549,0.8431372549019608,0.03529411764705882,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6666666666666666,0.996078431372549,0.5490196078431373,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7764705882352941,0.996078431372549,0.44313725490196076,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7764705882352941,0.996078431372549,0.44313725490196076,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7764705882352941,0.996078431372549,0.44313725490196076,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

x = np.array(x)
x = x.reshape(1,28,28,1)
extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[-2].output)
features = extractor.predict(x)
print(features)
for ln in range(5):
    extractor = tf.keras.models.Model(inputs=model.inputs,outputs=model.layers[ln].output)
    features = extractor.predict(x)
    #print(features)
    print('layer',ln)
    print('layer input shape',model.layers[ln].input.shape)
    if not ln == 2:
        print('layer weights shape',model.layers[ln].get_weights()[0].shape)
    print('output shape',features.shape)
    print()
sys.exit(0)

w = model.get_weights()
out_string = ''
for wi in w:
    print(wi)
    if(len(wi.shape)==1):
        # print bias
        for i in range(wi.shape[0]):
            out_string += str(wi[i])+',\n'
    if(len(wi.shape)==2):
        if(wi.shape==(1568,100)):
            #flatten
            wi = wi.reshape(7,7,32,100)
            wi = np.transpose(wi,(2,0,1,3))
            wi = wi.reshape(1568,100)
        wi = wi.T
        # print dense layer
        for i in range(wi.shape[0]):
            for j in range(wi.shape[1]):
                out_string += str(wi[i,j])+','
            out_string += '\n'
    if(len(wi.shape)==4):
        # print conv layer
        for oc in range(wi.shape[3]):
            for ic in range(wi.shape[2]):
                #wi[:,:,ic,oc] = wi[:,:,ic,oc].T
                for w in range(wi.shape[1]):
                    for h in range(wi.shape[0]):
                        out_string += str(wi[w,h,ic,oc]) + ','
            out_string += '\n'

with open('test.nnet','w') as f:
    f.write(out_string)
