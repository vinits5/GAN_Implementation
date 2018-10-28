import tensorflow as tf
import numpy as np 
import csv
import matplotlib.pyplot as plt
import scipy.misc
import scipy
import os

def read_digits():
	data = []
	with open('data/train.csv','r') as csvfile:
		csvreader = csv.reader(csvfile)
		csvreader.next()
		for row in csvreader:
			row.pop(0)
			row = [int(x) for x in row]
			data.append(row)
	return np.array(data).reshape((len(data),28,28))

def sample_images(batch_size,data):
	samples = np.arange(0,data.shape[0])
	np.random.shuffle(samples)
	samples = samples[0:batch_size]
	sampled_data = np.zeros((batch_size,data.shape[1]+4,data.shape[2]+4))
	for i in range(sampled_data.shape[0]):
		sampled_data[i,2:30,2:30]=data[samples[i],:,:]
	return sampled_data.reshape(batch_size,data.shape[1]+4,data.shape[2]+4,1)

def sample_noise(batch_size,noise_vector_size):
	noises = np.zeros((batch_size,noise_vector_size))
	for i in range(batch_size):
		noises[i,:] = np.random.uniform(-1,1,noise_vector_size)
	noises = np.round(noises,3)
	return noises

def log_file(is_training,LOG_FILE):
	if is_training:
		os.mkdir(LOG_FILE)
		file = open(LOG_FILE+'/log.txt','w')
		file.write('GAN Training\n')
		file.close()

def log_string(data,LOG_FILE):
	with open(LOG_FILE+'/log.txt','a') as file:
		file.write(data)
		file.write('\n')


class Logger(object):
	"""Logging in tensorboard without tensorflow ops."""

	def __init__(self, log_dir):
		"""Creates a summary writer logging to log_dir."""
		self.writer = tf.summary.FileWriter(log_dir)

	def log_scalar(self, tag, value, step):
		"""Log a scalar variable.
		Parameter
		----------
		tag : basestring
			Name of the scalar
		value
		step : int
			training iteration
		"""
		summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
													 simple_value=value)])
		self.writer.add_summary(summary, step)


# Taken from Arthur Juliani's Blog
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img