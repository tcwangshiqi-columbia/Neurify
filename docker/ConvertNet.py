import math
import sys
import numpy as np
import h5py
import functools
import operator
from keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

if len(sys.argv) != 2:
   print("Excactly 1 argument expected to specify the file to convert.")
   exit()

kerasFile = sys.argv[1]

def writeNNet(model,fileName):
    '''
    Write network data to the .nnet file format
    Args:
        model (kerasModel): the model to be converted
        fileName (String): a path to the file the nnet-file should be written to
    '''
    # Get a list of the model weights
    model_params = model.get_weights()

    # Split the network parameters into weights and biases
    weights = model_params[0:len(model_params):2]
    biases  = model_params[1:len(model_params):2]

    #Open the file we wish to write
    with open(fileName,'w') as f2:

        #Extract the necessary information
        numLayers = len(weights)
        inputSize = functools.reduce(operator.mul, model.layers[0].input_shape[1:])
        outputSize = len(biases[-1])
        maxLayerSize = inputSize

        layerTypes = ""
        convConfigs = ""
        layerSizes = str(inputSize) + ","
        lastLayerSize = inputSize
        flattenLayer = -1
        #extract layer types and some more information
        for (i, l) in enumerate(model.layers):
            if isinstance(l, Dense):
                layerTypes = layerTypes +  "0,"
                layerSizes = layerSizes + str(l.units) + ","
                lastLayerSize = l.units
                maxLayerSize = max(maxLayerSize, l.units)
            elif isinstance(l, Conv2D):
                if l.kernel_size[0] != l.kernel_size[1]:
                    print("Only quadratic kernels support but found shape: {}".format(l.kernal_size))
                    return
                if l.strides[0] != l.strides[1]:
                    print("only quadratic strides supported but found shape: {}".format(l.strides))
                    return
                layerTypes = layerTypes +  "1,"
                convConfigs = convConfigs + "{}, {}, {}, {}, 1, \n".format(l.filters, l.input_shape[-1], l.kernel_size[0], l.strides[0])
                layerSize = l.input_shape[1] * l.input_shape[2] / (l.strides[0] ** 2)
                if layerSize != math.floor(layerSize):
                    print("Error computing layer size of conv layer. (Last layer: {}, factor: {})".format(lastLayerSize, factor))
                    return
                else:
                    layerSize = int(layerSize) * l.filters
                maxLayerSize = max(maxLayerSize, layerSize)
                layerSizes = layerSizes + str(layerSize) + ","
            elif isinstance(l, Flatten):
                if flattenLayer >= 0:
                    print("Multiple flatten layers not supported.")
                    return
                flattenLayer = i
            else:
                print("Unsuported layerType: " + str(type(l)))
                return
        layerTypes = layerTypes + "\n"
        layerSizes = layerSizes + "\n"
        #write header
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write(layerSizes)
        f2.write(layerTypes)
        f2.write(convConfigs)

        lastConvFilters = -1
        for (idx, (w,b)) in enumerate(zip(weights,biases)):
            if len(w.shape) == 4: #thats a convolution layer
                lastConvFilters = w.shape[3]
                for oc in range(w.shape[3]):
                    for ic in range(w.shape[2]):
                        for wi in range(w.shape[1]):
                            for h in range(w.shape[0]):
                                f2.write("%.5e," % w[wi][h][ic][oc])
                    f2.write("\n")
            elif len(w.shape) == 2: #thats a fully connected layer
                #check if there is a flatten layer or not
                if idx == flattenLayer:
                    oldShape = w.shape
                    restDim = math.sqrt(w.shape[0]/lastConvFilters)
                    if restDim != math.floor(restDim):
                        print("Cannot reshape weights of flattenlayer.")
                        return
                    else:
                        restDim = int(restDim)
                    w = w.reshape(restDim, restDim, lastConvFilters, w.shape[1])
                    w = np.transpose(w, (2, 0, 1, 3))
                    w = w.reshape(oldShape)
                w = w.T
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        f2.write("%.5e," % w[i, j])
                    f2.write("\n")
            else:
                print("Unexpected shape of weights ")
                return

            for i in range(len(b)):
                f2.write("%.5e,\n" % b[i])
            pass

model = load_model(kerasFile)


nnetFile = kerasFile[:kerasFile.index('.')] + '.nnet'

# Convert the model 
writeNNet(model, nnetFile)
