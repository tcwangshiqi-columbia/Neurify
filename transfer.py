'''
------------------------------------------------------------------
** Top contributors:
**   Shiqi Wang
** This file is part of the Neurify project.
** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
** and their institutional affiliations.
** All rights reserved.
-----------------------------------------------------------------
'''

import numpy as np
import tensorflow as tf
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data
from model import Model



def print_image(x_test):

	for img in range(10):
		x = x_test[img:img+1]
		#x /= 255
		file = open("images/image"+str(img),"w")
		for i in range(x.shape[1]):
			file.write("%s,"% str(x[0,i]*255))
		file.close()

# python3 transfer.py > conv_kolter.net
def print_model(sess):
	w = sess.run(weights)

	for wi in w:
		
		if(len(wi.shape)==1):
			# print bias
			for i in range(wi.shape[0]):
				print(wi[i], end="")
				print(",")
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
					print(str(wi[i,j])+',', end = "")
				print("")
		if(len(wi.shape)==4):
			# print conv layer
			for oc in range(wi.shape[3]):
				for ic in range(wi.shape[2]):
					#wi[:,:,ic,oc] = wi[:,:,ic,oc].T
					for w in range(wi.shape[1]):
						for h in range(wi.shape[0]):
							print(str(wi[w,h,ic,oc])+',', end = "")
				print("")
		
		#print (wi.shape)


def load_image(img_path):
	file = open(img_path,"r")
	x_adv = file.read().split()
	if(len(x_adv)!=784):
		print ("image format is not correct!")
		exit()
	for i in range(len(x_adv)):
		x_adv[i] = float(x_adv[i])
	x_adv = np.array(x_adv).reshape(-1,784)
	return x_adv





if __name__ == '__main__':
	PATH = "models/adv_trained/adv_model.ckpt"
	#PATH = "models/baseline/checkpoint.ckpt"

	model = Model()
	mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

	np.random.seed(1)
	x = mnist.test.images[:100]
	y = mnist.test.labels[:100]

	#print_image(x)

	x_adv = load_image("advs/adv0")
	
	weights = tf.trainable_variables()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, PATH)
		print ("load model", PATH)
		
		#print_model(sess)
		y_p = sess.run(model.pre_softmax,\
				feed_dict={model.x_input:x[0:1],\
				model.y_input:y[0:1]})

		y_p_adv = sess.run(model.pre_softmax,\
				feed_dict={model.x_input:x_adv[0:1],\
				model.y_input:y[0:1]})
		print (y_p)
		print (y_p_adv)
		#print(y_p[0,:98])
		#print (y_p)
		#print (x[0].reshape(28,28))






		
