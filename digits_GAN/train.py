import tensorflow as tf 
import helper
from networks import Network
import numpy as np
import os
import datetime
import input_data

BATCH_SIZE = 128
NOISE_SIZE = 100
EPOCHS = 500000
img_width,img_height,channel=32,32,1
is_training = True
LOG_FILE = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
helper.log_file(is_training,LOG_FILE)

net = Network()
noise_vector = tf.placeholder(tf.float32,shape=(BATCH_SIZE, NOISE_SIZE))
image = tf.placeholder(tf.float32,shape=(BATCH_SIZE, img_width, img_height,channel))

initializer = tf.truncated_normal_initializer(stddev=0.02)

generated_image = net.generator(noise_vector = noise_vector, initializer = initializer)
Dx = net.discriminator(image = image, initializer = initializer)
Dg = net.discriminator(image = generated_image, initializer = initializer, reuse = True)

generator_loss = -tf.reduce_mean(tf.log(Dg))
discriminator_loss = -tf.reduce_mean(tf.add(tf.log(Dx),tf.log(1.-Dg)))

tvars = tf.trainable_variables()
optimizerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
optimizerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)

d_grads = optimizerD.compute_gradients(discriminator_loss,tvars[9:])
g_grads = optimizerG.compute_gradients(generator_loss,tvars[0:9])

updateG = optimizerG.apply_gradients(g_grads)
updateD = optimizerD.apply_gradients(d_grads)


# Train the Network.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=20)
logger = helper.Logger(LOG_FILE)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

for epoch in range(EPOCHS):
	noise_samples = helper.sample_noise(BATCH_SIZE,NOISE_SIZE)
	real_images,_ = mnist.train.next_batch(BATCH_SIZE)
	real_images = (np.reshape(real_images,[BATCH_SIZE,28,28,1]) - 0.5) * 2.0
	real_images = np.lib.pad(real_images, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1))

	feed_dict = {noise_vector:noise_samples, image:real_images}
	loss_d, _ = sess.run([discriminator_loss, updateD], feed_dict = feed_dict)

	feed_dict = {noise_vector:noise_samples}
	loss_g, _ = sess.run([generator_loss, updateG], feed_dict = feed_dict)
	loss_g, _ = sess.run([generator_loss, updateG], feed_dict = feed_dict)

	if (epoch%1000) == 0:
		logger.log_scalar(tag='Generator Loss',value=loss_g,step=epoch)
		logger.log_scalar(tag='Discriminator Loss',value=loss_d,step=epoch)
		print("LossG: {} and LossD: {}".format(float(loss_g), float(loss_d)))
		test_noise = helper.sample_noise(BATCH_SIZE, NOISE_SIZE)
		images = sess.run(generated_image, feed_dict = {noise_vector:test_noise})

		ImagesPath = os.path.join(os.getcwd(), LOG_FILE, 'ImagesPath')
		if not os.path.exists(ImagesPath):
			os.mkdir(ImagesPath)
		helper.save_images(np.reshape(images[0:36],[36,32,32]), [6,6], ImagesPath+'/fig'+str(epoch)+'.png')

		ModelPath = os.path.join(os.getcwd(), LOG_FILE, 'ModelPath')
		if not os.path.exists(ModelPath):
			os.mkdir(ModelPath)
		saver.save(sess,ModelPath+'/model'+str(epoch)+'.ckpt')

	helper.log_string('######'+str(epoch)+'######', LOG_FILE)
	helper.log_string('Generator Loss: '+str(loss_g), LOG_FILE)
	helper.log_string('Discriminator Loss: '+str(loss_d), LOG_FILE)