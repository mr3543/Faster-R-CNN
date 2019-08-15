import os.path as osp
import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.TRAIN = edict()
cfg.NET = edict()
cfg.DATA = edict()
cfg.TEST = edict()

cfg.DATA.SHORTEST_SIDE_SCALE = 600
cfg.DATA.DATA_MEAN = 112.0

cfg.TEST.MAX_BOXES_FOR_DETECTION = 100

cfg.NET.ANCHOR_SCALES = (64.0**2,128.0**2,256.0**2,512.0**2)
cfg.NET.ANCHOR_RATIOS = (0.5,1.0,2.0)
cfg.NET.NUM_ANCHORS = len(cfg.NET.ANCHOR_SCALES)*len(cfg.NET.ANCHOR_RATIOS)

cfg.NET.RPN_CHANNELS = 512
cfg.NET.ROI_POOL_HEIGHT = 7
cfg.NET.ROI_POOL_WIDTH = 7

cfg.NET.NUM_CLASSES = 81

colab_dir = '/content/gdrive/My Drive/Faster-R-CNN'

cfg.NET.BASE_MODEL_WEB_CKPT = colab_dir + '/model/base_model/web/vgg_16.ckpt'
cfg.NET.BASE_MODEL_CKPT = colab_dir + '/model/base_model/base_model.ckpt'
cfg.NET.RPN_NETWORK_CKPT = colab_dir + '/model/rpn/rpn_network/rpn_network.ckpt'
cfg.NET.RCNN_NETWORK_CKPT = colab_dir + '/model/rcnn/rcnn_network/rcnn_network.ckpt'
cfg.NET.RCNN_PT_NETWORK_CKPT = colab_dir + '/model/rcnn/rcnn_network_pt/rcnn_network_pt.ckpt'
cfg.NET.RPN_TRAINING_CKPT = colab_dir + '/model/rpn/rpn_training/rpn_training.ckpt'
cfg.NET.RCNN_TRAINING_CKPT = colab_dir + '/model/rcnn/rcnn_training/rcnn_training.ckpt'

cfg.NET.TENSORBOARD_DIR =  colab_dir + '/graphs'

cfg.TRAIN.RPN_POS_THRESH = 0.7
cfg.TRAIN.RPN_NEG_THRESH_LO = 0.0
cfg.TRAIN.RPN_NEG_THRESH_HI = 0.7

cfg.TRAIN.RPN_BATCH_SIZE = 256
cfg.TRAIN.RPN_PROP_POS = 0.5

cfg.TRAIN.PRE_NMS_TRAIN_TOP_K = 12000
cfg.TRAIN.POST_NMS_TRAIN_TOP_K = 2000
cfg.TRAIN.PRE_NMS_TEST_TOP_K = 6000
cfg.TRAIN.POST_NMS_TEST_TOP_K = 300

cfg.TRAIN.RCNN_POS_THRESH = 0.5
cfg.TRAIN.RCNN_NEG_THRESH_LO = 0.0
cfg.TRAIN.RCNN_NEG_THRESH_HI = 0.5

cfg.TRAIN.RCNN_BATCH_SIZE = 256
cfg.TRAIN.RCNN_PROP_POS = 0.25

cfg.TRAIN.RPN_LAMBDA = 1.0
cfg.TRAIN.RCNN_LAMBDA = 1.0

cfg.TRAIN.RPN_WEIGHT_DECAY = 0.0005
cfg.TRAIN.RPN_MOMENTUM = 0.9

cfg.TRAIN.RCNN_WEIGHT_DECAY = 0.0005
cfg.TRAIN.RCNN_MOMENTUM = 0.9

cfg.TRAIN.LR1_BLOBS = 80
cfg.TRAIN.LR2_BLOBS = 40
cfg.TRAIN.BLOB_SIZE = 100

cfg.TRAIN.RPN_LR1 = 0.001
cfg.TRAIN.RCNN_LR1 = 0.0001

cfg.TRAIN.RPN_LR2 = 0.0001
cfg.TRAIN.RCNN_LR2 = 0.0001

cfg.TRAIN.RANDOM_SEED = 0

