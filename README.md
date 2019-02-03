# GAN Implementation

This is a Deep Convolutional Generative Adversarial Network for mnist data of handwritten digits.

## Network Architecture:
<p align="center">
  <img src="https://github.com/vinits5/GAN_Implementation/blob/master/results/network.png" title="Network Architecture">
</p>

## Results:

Following is a gif of results obtained during training:
<p align="center">
  <img src="https://github.com/vinits5/GAN_Implementation/blob/master/results/results.gif" width="576" height="576" title="Network Architecture">
</p>

![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig0.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig100.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig1000.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig20000.png)\
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig50000.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig100000.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig200000.png)
![](https://github.com/vinits5/GAN_Implementation/blob/master/results/epochs/fig258000.png)

## Codes:
**networks.py** file contains the network architecture in tensorflow.\
**input_data.py** file is used for i/o operations of mnist data.\
**train.py** file has the algorithm for GAN & used to train the network.\
**test.py** file is used to test the trained network.\

*Network is trained on NVIDIA Geforce GTX1070 with tensorflow-1.4 and python2.7*