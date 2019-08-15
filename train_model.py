import tensorflow as tf
import numpy as np
import os
import os.path as osp
import sys
import gc
from Network import Network
from config import cfg
from Imdb import Imdb
from ObjectDetector import ObjectDetector

if __name__ == '__main__':
    """
    trains the faster rcnn model on the coco train2017 dataset 

    """
    sess = tf.Session(graph = tf.Graph())
    net = Network(cfg,True,sess)


    imdb = Imdb('/content/image_data','train2017',cfg)
    od = ObjectDetector(net,imdb,cfg)
    od.train()




