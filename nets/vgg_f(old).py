import tensorflow as tf
from misc.layers import *
import numpy as np

keep_prob=1.0
#weigths and biases for tensorflow
net = np.load('weights/vgg_f.npy').item()
weights = {}
biases = {}
for name in net.keys():
    weights[name] = tf.Variable(tf.constant(net[name][0]), dtype='float32' ,name=name+"_weight", trainable=False)
    biases[name] = tf.Variable(tf.constant(net[name][1]), dtype='float32' ,name=name+"_bias", trainable=False)
def vggf(x):
    #check image dimensions
    assert x.get_shape().as_list()[1:] == [224, 224, 3]
    layers = {}
    x = tf.cast(x, tf.float32)
    with tf.name_scope("conv1"):
        layers['conv1'] = conv_layer(x, weights['conv1'], biases['conv1'], s=4, padding='VALID')
        layers['norm1'] = tf.nn.lrn(layers['conv1'],2,2.000,0.0001,0.75) 
        layers['pool1'] = max_pool(layers['norm1'], k=3, s=2)

    with tf.name_scope("conv2"):
        layers['conv2'] = conv_layer(layers['pool1'], weights['conv2'], biases['conv2'])
        layers['norm2'] = tf.nn.lrn(layers['conv2'],2,2.000,0.0001,0.75) 
        layers['pool2'] = max_pool(layers['norm2'], k=3, s=2, padding='VALID')

    with tf.name_scope("conv3"):
        layers['conv3'] = conv_layer(layers['pool2'], weights['conv3'], biases['conv3'])

    with tf.name_scope("conv4"):
        layers['conv4'] = conv_layer(layers['conv3'], weights['conv4'], biases['conv4'])

    with tf.name_scope("conv5"):
        layers['conv5'] = conv_layer(layers['conv4'], weights['conv5'], biases['conv5'])
        layers['pool5'] = max_pool(layers['conv5'], k=3, s=2, padding='VALID')

    flatten = tf.reshape(layers['pool5'], [-1, 9216])

    with tf.name_scope('fc6'):
        layers['fc6'] = tf.nn.relu(fully_connected(flatten, weights['fc6'], biases['fc6']))
        layers['fc6'] = tf.nn.dropout(layers['fc6'], keep_prob=keep_prob)

    with tf.name_scope('fc7'):
        layers['fc7'] = tf.nn.relu(fully_connected(layers['fc6'], weights['fc7'], biases['fc7']))
        layers['fc7'] = tf.nn.dropout(layers['fc7'], keep_prob=keep_prob)

    with tf.name_scope('fc8'):
        layers['fc8'] = fully_connected(layers['fc7'], weights['fc8'], biases['fc8'])
        #layers['prob'] = tf.nn.softmax(layers['fc8'])

    return layers['fc8']


