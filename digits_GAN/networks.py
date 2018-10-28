#Import the libraries we will need.
import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy

class Network:
	def generator(self,noise_vector,initializer):
		zP = slim.fully_connected(noise_vector,4*4*256,normalizer_fn=slim.batch_norm,\
			activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
		zCon = tf.reshape(zP,[-1,4,4,256])
		
		gen1 = slim.convolution2d_transpose(\
			zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
			padding="SAME",normalizer_fn=slim.batch_norm,\
			activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
		
		gen2 = slim.convolution2d_transpose(\
			gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
			padding="SAME",normalizer_fn=slim.batch_norm,\
			activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
		
		gen3 = slim.convolution2d_transpose(\
			gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
			padding="SAME",normalizer_fn=slim.batch_norm,\
			activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)

		g_out = slim.convolution2d_transpose(\
			gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",\
			biases_initializer=None,activation_fn=tf.nn.tanh,\
			scope='g_out', weights_initializer=initializer)
		
		return g_out

	def discriminator(self,image,initializer, reuse=False):
		
		dis1 = slim.convolution2d(image,16,[4,4],stride=[2,2],padding="SAME",\
			biases_initializer=None,activation_fn=tf.nn.relu,\
			reuse=reuse,scope='d_conv1',weights_initializer=initializer)
		
		dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",\
			normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,\
			reuse=reuse,scope='d_conv2', weights_initializer=initializer)
		
		dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
			normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu,\
			reuse=reuse,scope='d_conv3',weights_initializer=initializer)
		
		d_out = slim.fully_connected(slim.flatten(dis3),1,activation_fn=tf.nn.sigmoid,\
			reuse=reuse,scope='d_out', weights_initializer=initializer)
		
		return d_out