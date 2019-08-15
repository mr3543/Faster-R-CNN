import tensorflow as tf
import numpy as np
import os.path as osp
import sys
import gc
from Network import Network
from config import cfg
from Imdb import Imdb
from ObjectDetector import ObjectDetector

if __name__ == '__main__':
    """
    tests a pretrained faster rcnn model on the coco test2017 dataset
    """

    sess = tf.Session(graph = tf.Graph())
    net = Network(cfg,False,sess)
    
    imdb = Imdb('/content/image_data','val2017',cfg)
    od = ObjectDetector(net,imdb,cfg)
    od.evaluate('/content/gdrive/My Drive/Faster-R-Cnn/results/coco_results.json',True)


