import json
import progressbar
import os
import math
import numpy as np
from PIL import Image
import sys

filename = ""
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Expected one argument to specify a folder or a image file.")
    exit()
rgbMode = True

filenames = [filename]
folder = ""
if os.path.isdir(filename):
    (_, _, filenames) = next(os.walk(filename))
    folder = filename

print("Found {} images in given path.".format(len(filenames)))

with progressbar.ProgressBar(redirect_stdout=True, max_value=len(filenames)) as bar:
    convIdx = -1
    for idx, filename in enumerate(filenames):
        with Image.open(folder + filename) as img:
            data = list(img.getdata())
            data = np.array(data).flatten()
            if not rgbMode:
                data = np.reshape(data, [-1, 3])
                data = np.dot(data, [0.2989, 0.5870, 0.1140])
            else:
                dim = int(math.sqrt((data.size/3)))
                data = np.reshape(data, (dim, dim, 3))
                new_data = []
                for k in range(data.shape[2]):
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            new_data.append(data[i, j, k])
            data = list(map(lambda x: str(x), new_data))
            data = ','.join(data)
            data = data + ','
            convIdx += 1
            if not os.path.exists("conv/"):
                os.mkdir("conv")
            newFile = "conv/image{}".format(convIdx)
            while os.path.exists(newFile):
                convIdx = convIdx + 1
                newFile = "conv/image{}".format(convIdx)
            f = open(newFile, 'w')
            f.write(data)
            f.close()
        bar.update(idx)


