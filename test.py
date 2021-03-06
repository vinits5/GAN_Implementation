import tensorflow as tf 
import helper
from networks import Network
import numpy as np
import os
import datetime
import input_data
import matplotlib.pyplot as plt 

BATCH_SIZE = 128
NOISE_SIZE = 100
EPOCHS = 500000
img_width, img_height, channel=32,32,1
is_training = False

with tf.device('/device:CPU:0'):
	net = Network()
	noise_vector = tf.placeholder(tf.float32,shape=(BATCH_SIZE, NOISE_SIZE))
	image = tf.placeholder(tf.float32,shape=(BATCH_SIZE, img_width, img_height,channel))

	initializer = tf.truncated_normal_initializer(stddev=0.02)

	generated_image = net.generator(noise_vector = noise_vector, initializer = initializer)
	Dx = net.discriminator(image = image, initializer = initializer)
	Dg = net.discriminator(image = generated_image, initializer = initializer, reuse = True)

	generator_loss = -tf.reduce_mean(tf.log(Dg))
	discriminator_loss = -tf.reduce_mean(tf.add(tf.log(Dx), tf.log(1.-Dg)))

	tvars = tf.trainable_variables()
	optimizerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
	optimizerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

	d_grads = optimizerD.compute_gradients(discriminator_loss, tvars[9:])
	g_grads = optimizerG.compute_gradients(generator_loss, tvars[0:9])

	updateG = optimizerG.apply_gradients(g_grads)
	updateD = optimizerD.apply_gradients(d_grads)


# Test the Network.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=20)
saver.restore(sess, 'log/model258400.ckpt')

test_noise = helper.sample_noise(BATCH_SIZE, NOISE_SIZE)
images = sess.run(generated_image, feed_dict = {noise_vector:test_noise})
images = images.reshape((-1,32,32))
plt.imshow(images[0], cmap='gray')
plt.show()