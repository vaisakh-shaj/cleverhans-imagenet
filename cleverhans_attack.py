from __future__ import print_function
import tensorflow as tf 
import pickle
import numpy as np

import logging

import matplotlib
#matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cleverhans.attacks import SaliencyMapMethod,FastGradientMethod,CarliniWagnerL2,DeepFool,BasicIterativeMethod,MadryEtAl
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
#from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans.model import *

from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from nets.resnet_50 import resnet50
from nets.resnet_152 import resnet152
from nets.inception_v3 import inceptionv3
from misc.utils import *
import tensorflow as tf
import numpy as np
import argparse
import time


#import os
#os.environ["CUDA_DEVICE_ORDER"] = ""


def validate_arguments(args):
    nets = ['vggf', 'caffenet', 'vgg16', 'vgg19', 'googlenet', 'resnet50', 'resnet152', 'inceptionv3']
    if not(args.network in nets):
        logging.info ('invalid network')
        exit (-1)
    
    if args.img_list is None or args.gt_labels is None:
        logging.info ('provide image list and labels')
        exit (-1)

def choose_net(network):    
    MAP = {
        'vggf'     : vggf,
        'caffenet' : caffenet,
        'vgg16'    : vgg16,
        'vgg19'    : vgg19, 
        'googlenet': googlenet, 
        'resnet50' : resnet50,
        'resnet152': resnet152, 
        'inceptionv3': inceptionv3,
    }
    
    if network == 'caffenet':
        size = 227
    elif network == 'inceptionv3':
        size = 299
    else:
        size = 224

    #placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')
    return MAP[network], input_image



def attack_batch(model, in_im, net_name, attack_name, im_list, gt_labels, sample_size, batch_size):
    logging.basicConfig(filename='Logs/'+net_name+"_"+attack_name+'.log', level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s')
    config = tf.ConfigProto(device_count = {'GPU': 2})
    imgs = open(im_list).readlines()  # [::10]
    gt_labels = open(gt_labels).readlines()  # [::10]
    top_1 = 0;top_1_real = 0;fool_rate = 0
    isotropic, size = get_params(net_name)
    imageModel = CallableModelWrapper(model, 'logits')

    
    with tf.Session(config=config) as sess:
        if attack_name=='fgsm':
            attack = FastGradientMethod(imageModel, back='tf')
            adv_x = attack.generate(in_im,eps=8,clip_min=-124, clip_max=155)
        if attack_name=='ifgsm':
            attack = BasicIterativeMethod(imageModel, back='tf')
            adv_x = attack.generate(in_im,eps=8,eps_iter=1,nb_iter=12,clip_min=-124, clip_max=155)
        if attack_name=='cw2':
            attack = CarliniWagnerL2(imageModel, back='tf')
            adv_x = attack.generate(in_im,clip_min=-124, clip_max=155)
        if attack_name=='jsma':
            attack = SaliencyMapMethod(imageModel, back='tf')
            adv_x = attack.generate(in_im)
        if attack_name=='pgd':
            attack = MadryEtAl(imageModel, back='tf')
            adv_x = attack.generate(in_im,eps=8,eps_iter=1,nb_iter=12,clip_min=-124, clip_max=155)
        if attack_name=='deepfool':
            attack = DeepFool(imageModel, back='tf')
            adv_x = attack.generate(in_im, sess=sess, clip_min=-124, clip_max=155)
        
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name, sess, isotropic, size)
        batch_im = np.zeros((batch_size, size, size, 3))
        
        for i in range(sample_size/batch_size):
            lim = min(batch_size, len(imgs)-i*batch_size)
            for j in range(lim):
                im = img_loader(imgs[i*batch_size+j].strip())
                batch_im[j] = np.copy(im)
            gt = np.array([int(gt_labels[i*batch_size+j].strip())
                       for j in range(lim)])
            adv_x_np=adv_x.eval(feed_dict={in_im: batch_im})

            # Calculate the neural probabilities
            y_adv_prob=tf.nn.softmax(model(in_im), name="yadv").eval(feed_dict={in_im: adv_x_np}); y_adv = np.argmax(y_adv_prob,1)
            y_true_prob=tf.nn.softmax(model(in_im), name="ypred").eval(feed_dict={in_im: batch_im}); y_true =  np.argmax(y_true_prob,1)

            # Calculate the top-1, top-1-true accuracies and fooling rate
            top_1 += np.sum(y_adv == gt); top_1_real += np.sum(y_true == gt)
            fool_rate += np.sum(y_true != y_adv )
            

            if i != 0 and i % 2 == 0:
                logging.info("batch: {} ==================================================================".format(i))
                logging.info("fooling rate {}".format((fool_rate)/float((i+1)*batch_size)*100))
            

    logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")           
    logging.info('Real Top-1 Accuracy = {}'.format(
    top_1_real/float(sample_size)*100))
    logging.info('Top-1 Accuracy = {}'.format((top_1/float(sample_size)*100)))
    logging.info('Top-1 Fooling Rate = {}'.format(fool_rate/float(sample_size)*100))
    logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='googlenet', help='The network eg. googlenet')
    parser.add_argument('--attack', default='fgsm', help='The attack method eg. fgsm')
    parser.add_argument('--img_path', default='misc/sample.jpg',  help='Path to input image')
    parser.add_argument('--img_list',  default='text/ilsvrc_test_correct.txt', help='Path to the validation image list')
    parser.add_argument('--gt_labels', default='text/gt_correct.txt',help='Path to the ground truth validation labels')
    parser.add_argument('--sample_size', default=50000,  help='Total Samples to create attack on')
    parser.add_argument('--batch_size', default=50,  help='Mini Batch size for creating attack')
    args = parser.parse_args()
    validate_arguments(args)
    model,inp_im  = choose_net(args.network)
    attack_batch(model, inp_im, args.network, args.attack, args.img_list,
args.gt_labels, int(args.sample_size), int(args.batch_size))


if __name__ == '__main__':
    main()
