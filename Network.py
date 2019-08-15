import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import gen_nn_ops
import numpy as np
from anchor_utils import *
import sys

class Network(object):
    
    def __init__(self,cfg,is_training,sess):
        """
        inputs: 

        'cfg': python edict, contains network and training parameters
        'is_training': training boolean 
        'sess': tensorflow session object

        returns: None

        initialize empty dictionaries which will store various tensor
        objects from the constructed graph

        """


        self.cfg = cfg
        self.sess = sess
        self.layers = {}
        self.losses = {}
        self.labels = {}
        self.inds = {}
        self.training_ops = {}
        self.init_ops = {}
        self.summary_ops = {}
        self.saver_ops = {}
        self.learning_rates = {}
        self.is_training = is_training

    def construct_graph(self):
        """
        inputs: None
        returns: None
        
        constructs the Faster R-CNN graph 

        creates the network from input image to Fast R-CNN detections

        layers used to contruct the losses and training operations are
        saved in 'self.layers'

        tensorboard summary ops are saved in 'self.summary_ops'

        the boolean 'self.is_training' determines how rois are selected
        from the RPN network and passed to the Fast R-CNN network

        """

        self.graph = self.sess.graph
        with self.graph.as_default():
            self.image = tf.compat.v1.placeholder(shape=[1,None,None,3],dtype = tf.float32)
            self.gt_boxes = tf.compat.v1.placeholder(shape=[None,5],dtype=tf.float32)

            # construct base model
            self.make_base_model()
            
            # make anchors
            self.anchor_boxes = self.make_anchor_boxes()

            # construct rpn network
            self.rpn_network()

            if self.is_training:
                self.anchor_rpn_batch = self.sample_anchor_batch()
            
            # select rois
            self.rois = self.select_rois()
            
            # roi pooling layer
            self.roi_crops = self.roi_pooling()

            # construct fast r-cnn network
            self.rcnn_network()
            
            # add loss and training ops
            if self.is_training:
                self.add_losses()
                self.add_training_ops()
            
            # init tensorboard summaries
            self.merge_summaries()
            
    def make_base_model(self):
        """
        inputs: None
        returns: None

        Make the base VGG16 architecture. adds the top layer to 
        'self.layers'

        """
        with tf.compat.v1.variable_scope('base_model'):
            base = slim.repeat(self.image,2,slim.conv2d,64,[3,3],
            trainable = False,scope = 'conv1')
            
            base = slim.max_pool2d(base,[2,2],padding='SAME',scope = 'pool1')
            
            base = slim.repeat(base,2,slim.conv2d,128,[3,3],
            trainable = False,scope='conv2')
            
            base = slim.max_pool2d(base,[2,2],padding='SAME',scope='pool2')
            
            base = slim.repeat(base,3,slim.conv2d,256,[3,3],
            trainable=True,scope='conv3')
            
            base = slim.max_pool2d(base,[2,2],padding='SAME',scope='pool3')
            
            base = slim.repeat(base,3,slim.conv2d,512,[3,3],
            trainable=True,scope='conv4')
            
            base = slim.max_pool2d(base,[2,2],padding='SAME',scope='pool5')
            
            base = slim.repeat(base,3,slim.conv2d,512,[3,3],
            trainable=True,scope='conv5')

            self.layers['base_model/feature_map'] = base

            vars_to_restore = slim.get_variables(scope='base_model')
            
            vars_to_restore = {self.name_in_ckpt(var):var for var in vars_to_restore}
            saver = tf.train.Saver(vars_to_restore)
            self.saver_ops['base_model'] = saver
            self.fm_sh = tf.shape(base)

    def make_anchor_boxes(self):
        """
        inputs: None
        returns: 'anchors', an [N,4] tensor where anchors[i,:] is an 
        array [x1,y1,x2,y2] which denotes the top left (x1,y1) and bottom
        right (x2,y2) corners of the box. 

        For each height,width location in the feature map we compute an 
        anchor center on the orginal image. Anchors are constructed using
        the anchor center and scale and ratio combinations stored in 
        'self.anchor_scales' and 'self.anchor_ratios'

        Anchors are then converted from centers to corners format. 
        """

        #get image and feature map dims
        im_sh = tf.shape(self.image)
        im_height = tf.cast(im_sh[1],dtype=tf.float32)
        im_width = tf.cast(im_sh[2],dtype=tf.float32)
        fm_sh = tf.shape(self.layers['base_model/feature_map'])
        
        fm_height = tf.cast(fm_sh[1],dtype=tf.float32)
        fm_width = tf.cast(fm_sh[2],dtype=tf.float32)

        #compute the top left anchor centers on the original image
        width_ratio = tf.math.floordiv(im_width,fm_width)
        height_ratio = tf.math.floordiv(im_height,fm_height)
        x_ctr = tf.math.floordiv(width_ratio,2)
        y_ctr = tf.math.floordiv(height_ratio,2)

        #compute the anchor centers across entire image
        x_ctrs = width_ratio*tf.range(0,fm_width) + x_ctr
        y_ctrs = height_ratio*tf.range(0,fm_height) + y_ctr

        #construct grid of center coordinates
        centers = tf.transpose(tf.meshgrid(x_ctrs,y_ctrs))
        centers_t = tf.transpose(centers,perm=(1,0,2))
        centers = tf.reshape(centers_t,(-1,2))

        # make combinations of anchor scales and anchor ratios
        anchor_combs = tf.transpose(tf.meshgrid(self.cfg.NET.ANCHOR_SCALES,self.cfg.NET.ANCHOR_RATIOS))
        anchor_combs = tf.reshape(anchor_combs,(-1,2))

        # compute the height and widths for each anchor scale/ratio combination
        heights = tf.math.round(tf.math.sqrt(anchor_combs[:,0]/anchor_combs[:,1]))
        widths = tf.math.round(anchor_combs[:,1]*heights)

        # merge the centers and widths and heights
        wh = tf.stack((widths,heights),axis=1)
        centers = tf.expand_dims(centers,axis=1)
        centers_ = tf.tile(centers,[1,tf.shape(wh)[0],1])
        centers_ = tf.reshape(centers_,(-1,2))

        wh_ = tf.tile(wh,[tf.shape(centers)[0],1])
        anchors_c = tf.concat((centers_,wh_),axis=1)

        a_x1 = anchors_c[:,0] - 0.5*anchors_c[:,2]
        a_y1 = anchors_c[:,1] - 0.5*anchors_c[:,3]
        a_x2 = anchors_c[:,0] + 0.5*anchors_c[:,2]
        a_y2 = anchors_c[:,1] + 0.5*anchors_c[:,3]

        ret = tf.stack((a_x1,a_y1,a_x2,a_y2),axis=1)
        self.ab_sh = tf.shape(ret)
        return ret

    def rpn_network(self):
        """
        inputs: None
        returns: None

        Constructs the RPN network. This consists of a 3x3 convolutional
        layer on the VGG feature map, followed by two sibling 1x1 convolutional
        layers. The number of output channels in the first conv layer is determined
        by the network attribute 'self.rpn_channels'. The channels in the sibling
        1x1 layers are 'self.num_anchors'*2 and 'self.num_anchors'*4. 

        The outputs 'obj_scores' and 'bbox_adjs' are of shapes 
        [batch,fm_height,fm_width,num_anchors*2] and [batch,fm_height,fm_width,num_anchors*4]

        For each height,width location in the feature map num_anchors*2 scores
        are given. Each is an object/not object score for a scale,ratio 
        combination anchor.

        For each height,width location in the feature map num_anchor*4 bounding box
        adjustments are given. Each is an offset to for a particular 
        scale/ratio combination anchor. 
        """
        init = tf.random_normal_initializer(mean=0,stddev=0.01)
        with slim.arg_scope([slim.conv2d],activation_fn = tf.nn.relu,
        padding = 'SAME',weights_initializer = init), \
        tf.compat.v1.variable_scope('rpn_network'):
            fm = self.layers['base_model/feature_map']
            rpn_net = slim.conv2d(fm,self.cfg.NET.RPN_CHANNELS,[3,3])
            obj_scores = slim.conv2d(rpn_net,self.cfg.NET.NUM_ANCHORS*2,[1,1],activation_fn = None)
            bbox_adjs = slim.conv2d(rpn_net,self.cfg.NET.NUM_ANCHORS*4,[1,1],activation_fn = None)
            self.layers['rpn_network/obj_scores'] = obj_scores
            self.layers['rpn_network/bbox_adjs'] = bbox_adjs

        self.rpn_net_obj_sh = tf.shape(obj_scores)
        self.rpn_net_bbox_sh = tf.shape(bbox_adjs)
        rpn_vars = slim.get_variables(scope='rpn_network')
        saver = tf.train.Saver(rpn_vars)
        self.saver_ops['rpn_network'] = saver

        rpn_init_op = tf.variables_initializer(rpn_vars)
        self.init_ops['rpn_init_op'] = rpn_init_op

    def sample_anchor_batch(self):
        """
        inputs: None

        returns: 
        
        'rpn_training_anchors': [K,4] tensor of anchors in
        [x1,y1,x2,y2] format

        """
        
        im_sh = tf.shape(self.image)
        im_height = im_sh[1]
        im_width = im_sh[2]
        ab_inds = tf.compat.v1.py_func(filter_anchors,[self.anchor_boxes,im_height,im_width],tf.int32)
        ab_inds = tf.expand_dims(ab_inds,axis=1)
        anchor_boxes_xy = tf.gather_nd(self.anchor_boxes,ab_inds)
        self.filtered_anchors_sh = tf.shape(anchor_boxes_xy)

        anchor_labels,bbox_labels,_ = tf.compat.v1.py_func(get_anchor_labels,[anchor_boxes_xy,
                self.gt_boxes,self.cfg.TRAIN.RPN_POS_THRESH,self.cfg.TRAIN.RPN_NEG_THRESH_LO,
                self.cfg.TRAIN.RPN_NEG_THRESH_HI],[tf.int32,tf.float32,tf.int32])
        
        anchor_batch_inds = tf.compat.v1.py_func(sample_anchors_for_training,[anchor_labels,
                self.cfg.TRAIN.RPN_BATCH_SIZE,self.cfg.TRAIN.RPN_PROP_POS],tf.int32)

        anchor_batch_inds = tf.expand_dims(anchor_batch_inds,axis=1)
        
        anchor_boxes_xy_training = tf.gather_nd(anchor_boxes_xy,anchor_batch_inds)
        anchor_labels_training = tf.gather_nd(anchor_labels,anchor_batch_inds)
        bbox_labels_training = tf.gather_nd(bbox_labels,anchor_batch_inds)

        self.training_anchors_sh = tf.shape(anchor_boxes_xy_training)
        self.anchor_labels_training_sh = tf.shape(anchor_labels_training)
        self.bbox_labels_training_sh = tf.shape(bbox_labels_training)

        self.labels['rpn/anchor_labels'] = anchor_labels_training
        self.labels['rpn/bbox_labels'] = bbox_labels_training
        self.inds['rpn_batch_inds'] = anchor_batch_inds
        self.inds['rpn_filter_inds'] = ab_inds

        return anchor_boxes_xy_training

    def select_rois(self):
        """
        inputs: None
        returns: [k,4] tensor where rois[i,:] is in [x1,y1,x2,y2] 
        format and corresponds to sections of the feature map 
        to be pooled for object classification.

        The roi selection process is as follows.
        
        We first select the top K anchors based on objectness score. 
        NMS is the performed with an IOU of 0.7. After NMS we select 
        the top k according to objectness score.
        
        If 'self.is_training' is 'True' we further sample R rois at random
        from the selected k with a pos/neg proportion defined in 'self.cfg'.

        """
        im_sh = tf.shape(self.image)
        im_height = tf.cast(im_sh[1],tf.float32)
        im_width = tf.cast(im_sh[2],tf.float32)

        obj_scores = tf.reshape((self.layers['rpn_network/obj_scores']),(-1,2))
        obj_scores = obj_scores[:,0]
        bbox_adjs = self.layers['rpn_network/bbox_adjs']
        bbox_adjs = tf.reshape(bbox_adjs,(-1,4))

        # format and adjust anchors
        anchor_boxes_xy = tf.compat.v1.py_func(adjust_anchors,
                            [self.anchor_boxes,bbox_adjs],tf.float32)
        self.adjusted_anchors_sh = tf.shape(anchor_boxes_xy)
        
        anchor_boxes_xy = tf.compat.v1.py_func(clip_rois,
                            [anchor_boxes_xy,im_height,im_width],tf.float32)
        self.clipped_anchors_sh = tf.shape(anchor_boxes_xy)

        ab_inds = tf.compat.v1.py_func(clean_anchors,[anchor_boxes_xy],tf.int32)
        ab_inds = tf.expand_dims(ab_inds,axis=1)
        anchor_boxes_xy = tf.gather_nd(anchor_boxes_xy,ab_inds)
        self.clean_anchors_sh = tf.shape(anchor_boxes_xy)

        obj_scores = tf.gather_nd(obj_scores,ab_inds)

        # set selection params
        if self.is_training:
            pre_k = self.cfg.TRAIN.PRE_NMS_TRAIN_TOP_K
            post_k = self.cfg.TRAIN.POST_NMS_TRAIN_TOP_K
        else:
            pre_k = self.cfg.TRAIN.PRE_NMS_TEST_TOP_K
            post_k = self.cfg.TRAIN.POST_NMS_TEST_TOP_K
        
        # pre nms top k selection
        inds_k = tf.compat.v1.py_func(top_k_inds,[obj_scores,pre_k],tf.int32)
        inds_k = tf.expand_dims(inds_k,axis=1)
        anchor_boxes_xy = tf.gather_nd(anchor_boxes_xy,inds_k)
        obj_scores = tf.gather_nd(obj_scores,inds_k)
        self.pre_nms_anchors_sh = tf.shape(anchor_boxes_xy)
        
        # nms
        nms_inds = tf.image.non_max_suppression(anchor_boxes_xy,obj_scores,pre_k,0.7)
        nms_inds = tf.expand_dims(nms_inds,axis=1)
        anchor_boxes_xy = tf.gather_nd(anchor_boxes_xy,nms_inds)
        obj_scores = tf.gather_nd(obj_scores,nms_inds)
        self.nms_anchors_sh = tf.shape(anchor_boxes_xy)

        # post nms top k selection
        inds_k = tf.compat.v1.py_func(top_k_inds,[obj_scores,post_k],tf.int32)
        inds_k = tf.expand_dims(inds_k,axis=1)
        anchor_boxes_xy = tf.gather_nd(anchor_boxes_xy,inds_k)
        self.post_nms_anchors_sh = tf.shape(anchor_boxes_xy)

        if self.is_training:
            pos_thresh = self.cfg.TRAIN.RCNN_POS_THRESH
            neg_thresh_lo = self.cfg.TRAIN.RCNN_NEG_THRESH_LO
            neg_thresh_hi = self.cfg.TRAIN.RCNN_NEG_THRESH_HI

            anchor_labels,bbox_gts,cls_labels = tf.compat.v1.py_func(get_anchor_labels,
            [anchor_boxes_xy,self.gt_boxes,pos_thresh,neg_thresh_lo,neg_thresh_hi],
            [tf.int32,tf.float32,tf.int32])
            
            batch_inds = tf.compat.v1.py_func(sample_anchors_for_training,
                    [anchor_labels,self.cfg.TRAIN.RCNN_BATCH_SIZE,self.cfg.TRAIN.RCNN_PROP_POS],tf.int32)
            batch_inds = tf.expand_dims(batch_inds,axis=1)
            self.labels['rcnn_training/roi_labels'] = tf.gather_nd(anchor_labels,batch_inds)
            self.labels['rcnn_training/bbox_gts'] = tf.gather_nd(bbox_gts,batch_inds)
            self.labels['rcnn_training/cls_labels'] = tf.gather_nd(cls_labels,batch_inds)

            anchor_boxes_xy = tf.gather_nd(anchor_boxes_xy,batch_inds)
            self.anchor_boxes_rcnn_training_sh = tf.shape(anchor_boxes_xy)

        return anchor_boxes_xy

    def roi_pooling(self):
        """
        inputs: None

        returns:
    
        'roi_crops': a tensor of [K,pool_h,pool_w,channels] where K is
        the number of rois stored in 'self.rois' and pool_h and pool_w 
        are defined in self.cfg
        """
        fm_sh = tf.shape(self.layers['base_model/feature_map'])
        fm_height = tf.cast(fm_sh[1],tf.float32)
        fm_width = tf.cast(fm_sh[2],tf.float32)
        im_sh = tf.shape(self.image)
        im_height = tf.cast(im_sh[1],tf.float32)
        im_width = tf.cast(im_sh[2],tf.float32)

        rois = tf.compat.v1.py_func(image_to_fm_rois,[self.rois,fm_height,fm_width,im_height,im_width],tf.float32)

        roi_crops = tf.map_fn(self.roi_pooling_single,rois,infer_shape=False)

        roi_crops = tf.reshape(roi_crops,(-1,7,7,512))
        self.roi_sh = tf.shape(roi_crops)

        return roi_crops

    def roi_pooling_single(self,box):
        """
        inputs: 

        'box': a single tensor of shape [4,] of format
        [x1,y1,x2,y2], this is a box on the feature map 
        for which we will perform roi pooling on.

        returns:

        'roi_pool': a tensor of shape [pool_h,pool_w,channels] 
        which is a max pooled slice of the feature map corresponding
        to the input region defined by 'box'.

        performs roi pooling on a single RoI box of the 
        form [x1,y1,x2,y2]. 

        we zero pad the cropped feature map to ensure that the 
        resulting crop's height and width are evenly divisible
        by the pool height and width
        """
       
        box_h = tf.cast(tf.math.ceil(box[3] - box[1]),dtype=tf.int32)
        box_w = tf.cast(tf.math.ceil(box[2] - box[0]),dtype=tf.int32)
        box_x = tf.cast(box[0],dtype=tf.int32)
        box_y = tf.cast(box[1],dtype=tf.int32)
        fm = self.layers['base_model/feature_map']
        fm_chan = tf.shape(fm)[-1]

        crop = tf.image.crop_to_bounding_box(fm,box_y,box_x,box_h,box_w)   

        pool_h = self.cfg.NET.ROI_POOL_HEIGHT
        pool_w = self.cfg.NET.ROI_POOL_WIDTH

        left_pad = tf.to_int32(tf.math.ceil(((pool_w*(tf.math.floordiv(box_w,pool_w) + 1))-box_w)/2))
        right_pad = tf.to_int32(tf.math.floor(((pool_w*(tf.math.floordiv(box_w,pool_w) + 1))-box_w)/2))
        top_pad = tf.to_int32(tf.math.ceil(((pool_h*(tf.math.floordiv(box_h,pool_h) + 1))-box_h)/2))
        bottom_pad = tf.to_int32(tf.math.floor(((pool_h*(tf.math.floordiv(box_h,pool_h) + 1))-box_h)/2))

        pads = [[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]]
        crop = tf.pad(crop,pads)
        c_sh = tf.shape(crop)

        k_size = [1,tf.cast(c_sh[1]/7,dtype=tf.int32),tf.cast(c_sh[2]/7,dtype=tf.int32),1]
 
        mp = gen_nn_ops.max_pool_v2(crop,k_size,k_size,padding='VALID')
    
        return mp
    
    def rcnn_network(self):
        """
        inputs: None
        returns: None

        Constructs the R-CNN network. This consists of two fully connected layers
        with 4096 outputs followed by relus, then two sibling fully connected 
        layers with 'self.num_classes' and 'self.num_classes*4' outputs.

        'cls_scores' is a tensor of shape [#rois,num_classes], this gives the 
        detection score for each class.

        'bbox_adjs' is a tensor of shape [#rois,num_classes*4], this gives
        the bounding box adjustment for each class.

        Both tensors are added to 'self.layers'.
        
        """

        init_1 = tf.random_normal_initializer(mean=0,stddev=0.01)
        init_2 = tf.random_normal_initializer(mean=0,stddev=0.001)
        with tf.compat.v1.variable_scope('fast_rcnn_network'):
            fc6 = slim.conv2d(self.roi_crops, 4096, [7, 7], padding='VALID', scope='fc6')
            if self.is_training:
                fc6 = slim.dropout(fc6,keep_prob=0.5,is_training=True,scope='dropout6')
            fc7 = slim.conv2d(fc6, 4096, [1, 1], scope='fc7')
            if self.is_training:
                fc7 = slim.dropout(fc7,keep_prob=0.5,is_training=True,scope='dropout7')
            cls_scores = slim.fully_connected(fc7,self.cfg.NET.NUM_CLASSES,activation_fn=None,weights_initializer=init_1,scope='fc_cls')
            bbox_adjs = slim.fully_connected(fc7,(self.cfg.NET.NUM_CLASSES-1)*4,activation_fn=None,weights_initializer=init_2,scope='fc_bbox')
            self.layers['fast_rcnn_network/cls_scores'] = cls_scores
            self.layers['fast_rcnn_network/bbox_adjs'] = bbox_adjs
            self.layers['fast_rcnn_network/cls_probs'] = tf.nn.softmax(cls_scores)

        self.rcnn_cls_sh = tf.shape(cls_scores)
        self.rcnn_bbox_sh = tf.shape(bbox_adjs)
        
        pt_vars_to_restore = slim.get_variables(scope='fast_rcnn_network/fc6')
        pt_vars_to_restore.extend(slim.get_variables(scope='fast_rcnn_network/fc7'))
        pt_vars_dict = {self.name_in_ckpt(var):var for var in pt_vars_to_restore}
    
        rcnn_vars = slim.get_variables(scope='fast_rcnn_network/fc_cls')
        rcnn_vars.extend(slim.get_variables(scope='fast_rcnn_network/fc_bbox'))
        
        pt_saver = tf.train.Saver(pt_vars_dict)
        saver = tf.train.Saver(rcnn_vars)

        self.saver_ops['pt_fc_layers'] = pt_saver
        self.saver_ops['rcnn_network'] = saver

        rcnn_init_op = tf.variables_initializer(rcnn_vars)
        self.init_ops['rcnn_init_op'] = rcnn_init_op

    def add_losses(self):
        """
        inputs: None
        returns: None

        adds both rpn and rcnn losses to the graph.
        """
        
        self.add_rpn_loss()
        self.add_rcnn_loss()
    
    def add_rpn_loss(self):
        """
        inputs: None
        returns: None

        constructs the rpn loss and stores the tensor
        in 'self.losses'
        
        """
        obj_labels = self.labels['rpn/anchor_labels']
        bbox_labels = self.labels['rpn/bbox_labels']
        pos_samples = tf.where(tf.equal(obj_labels,1))

        obj_scores = self.layers['rpn_network/obj_scores']
        obj_scores = tf.reshape(obj_scores,(-1,2))
        obj_scores = tf.gather_nd(obj_scores,self.inds['rpn_filter_inds'])
        obj_scores = tf.gather_nd(obj_scores,self.inds['rpn_batch_inds'])
        bbox_scores = self.layers['rpn_network/bbox_adjs']
        bbox_scores = tf.reshape(bbox_scores,(-1,4))
        bbox_scores = tf.gather_nd(bbox_scores,self.inds['rpn_filter_inds'])
        bbox_scores = tf.gather_nd(bbox_scores,self.inds['rpn_batch_inds'])

        self.rpn_loss_obj_scores_sh = tf.shape(obj_scores)
        self.rpn_loss_bbox_scores_sh = tf.shape(bbox_scores)

        obj_labels = (obj_labels + 1)/2
        obj_labels = 1 - obj_labels
        obj_labels = tf.cast(obj_labels,dtype=tf.int32)
        obj_label_one_hots = tf.one_hot(obj_labels,2)
        obj_ce = tf.nn.softmax_cross_entropy_with_logits(labels=obj_label_one_hots,logits=obj_scores)
        obj_loss = tf.reduce_mean(obj_ce)

        bbox_labels_for_loss = tf.gather_nd(bbox_labels,pos_samples)
        bbox_scores_for_loss = tf.gather_nd(bbox_scores,pos_samples)
        num_pos = tf.shape(bbox_scores_for_loss)[0]
        num_pos_inv = tf.cast(1/num_pos,dtype=tf.float32)

        bbox_loss = self.cfg.TRAIN.RPN_LAMBDA*num_pos_inv*tf.losses.huber_loss(labels=bbox_labels_for_loss,predictions=bbox_scores_for_loss)
        self.losses['rpn/cls_loss'] = obj_loss
        self.losses['rpn/bbox_loss'] = bbox_loss
        self.losses['rpn/total_loss'] = obj_loss + bbox_loss

        tf.summary.scalar('rpn/obj_loss',obj_loss)
        tf.summary.scalar('rpn/bbox_loss',bbox_loss)
        tf.summary.scalar('rpn/total_loss',obj_loss + bbox_loss)

    def add_rcnn_loss(self):
        """
        inputs: None
        returns: None

        computes the rcnn loss and stores the loss
        tensor in 'self.losses'

        """

        roi_labels = self.labels['rcnn_training/roi_labels']        
        bbox_gts = self.labels['rcnn_training/bbox_gts']
        cls_labels = self.labels['rcnn_training/cls_labels']

        cls_scores = self.layers['fast_rcnn_network/cls_scores']
        bbox_adjs = self.layers['fast_rcnn_network/bbox_adjs']

        cls_scores = tf.reshape(cls_scores,(-1,self.cfg.NET.NUM_CLASSES))
        bbox_adjs = tf.reshape(bbox_adjs,(-1,(self.cfg.NET.NUM_CLASSES-1),4))

        self.rcnn_loss_cls_scores_sh = tf.shape(cls_scores)
        self.rcnn_loss_bbox_scores_sh = tf.shape(bbox_adjs)

        pos_rois = tf.cast(tf.where(tf.equal(roi_labels,1)),tf.int32)
        self.non_zero_cls_label_inds = tf.where(tf.not_equal(cls_labels,0))
        self.roi_selected_inds = pos_rois
        pos_roi_cls_labels = tf.gather_nd(cls_labels,pos_rois)

        class_one_hots = tf.one_hot(cls_labels,self.cfg.NET.NUM_CLASSES)
        class_ce = tf.nn.softmax_cross_entropy_with_logits(labels=class_one_hots,logits=cls_scores)
        class_loss = tf.reduce_mean(class_ce)

        bbox_gt_pos = tf.gather_nd(bbox_gts,pos_rois)
        pos_roi_cls_labels = pos_roi_cls_labels - 1
        pos_roi_cls_labels = tf.expand_dims(pos_roi_cls_labels,axis=1)
        bbox_adjs_inds = tf.concat((pos_rois,pos_roi_cls_labels),axis=1)

        bbox_adjs_for_loss = tf.gather_nd(bbox_adjs,bbox_adjs_inds)
        self.rcnn_loss_bbox_pos_sh = tf.shape(bbox_adjs_for_loss)

        num_pos = tf.shape(bbox_adjs_for_loss)[0]
        num_pos_inv = tf.cast(1/num_pos,tf.float32)

        bbox_loss = self.cfg.TRAIN.RCNN_LAMBDA*num_pos_inv*tf.losses.huber_loss(labels=bbox_gt_pos,predictions=bbox_adjs_for_loss)

        self.losses['rcnn/class_loss'] = class_loss
        self.losses['rcnn/bbox_loss'] = bbox_loss
        self.losses['rcnn/total_loss'] = class_loss + bbox_loss

        tf.summary.scalar('rcnn/class_loss',class_loss)
        tf.summary.scalar('rcnn/bbox_loss',bbox_loss)
        tf.summary.scalar('rcnn/total_loss',class_loss + bbox_loss)

    def add_training_ops(self):
        """
        inputs: None
        returns: None

        computes gradients and adds training operations to
        the graph
        """
        with tf.compat.v1.variable_scope('rpn_training_op'):
            rpn_learning_rate = tf.get_variable('rpn_learning_rate',[],tf.float32)
            rpn_optimizer = tf.contrib.opt.MomentumWOptimizer(self.cfg.TRAIN.RPN_WEIGHT_DECAY,
                    rpn_learning_rate,self.cfg.TRAIN.RPN_MOMENTUM)
            rpn_grads = rpn_optimizer.compute_gradients(self.losses['rpn/total_loss'])
            rpn_training_op = rpn_optimizer.apply_gradients(rpn_grads)

            self.training_ops['rpn_training_op'] = rpn_training_op
            self.learning_rates['rpn_learning_rate'] = rpn_learning_rate
        
        with tf.compat.v1.variable_scope('rcnn_training_op'):
            rcnn_learning_rate = tf.get_variable('rcnn_learning_rate',[],tf.float32)
            rcnn_optimizer = tf.contrib.opt.MomentumWOptimizer(self.cfg.TRAIN.RCNN_WEIGHT_DECAY,
                    rcnn_learning_rate,self.cfg.TRAIN.RCNN_MOMENTUM)

            rcnn_grads = rcnn_optimizer.compute_gradients(self.losses['rcnn/total_loss'])
            rcnn_training_op = rcnn_optimizer.apply_gradients(rcnn_grads)
            
            self.training_ops['rcnn_training_op']  = rcnn_training_op
            self.learning_rates['rcnn_learning_rate'] = rcnn_learning_rate

        rpn_training_vars = slim.get_variables(scope='rpn_training_op')
        rpn_saver = tf.train.Saver(rpn_training_vars)
        self.saver_ops['rpn_training_op'] = rpn_saver
        
        rpn_training_op_init = tf.variables_initializer(rpn_training_vars)
        self.init_ops['rpn_training_op'] = rpn_training_op_init

        rcnn_training_vars = slim.get_variables(scope='rcnn_training_op')
        rcnn_saver = tf.train.Saver(rcnn_training_vars)
        self.saver_ops['rcnn_training_op'] = rcnn_saver

        rcnn_training_op_init = tf.variables_initializer(rcnn_training_vars)
        self.init_ops['rcnn_training_op'] = rcnn_training_op_init

    def name_in_ckpt(self,var):
        """
        inputs: 
        
        'var': tensorflow variable

        returns:

        string used retrieve variable from vgg ckpt file
        """

        s1 = var.name.split(':')[0]
        l = s1.split('/')
        return 'vgg_16/' + '/'.join(l[1:])

    def load_base_model_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads vgg variable weights from ckpt file

        """
        
        saver = self.saver_ops['base_model']
        saver.restore(self.sess,file)

    def load_rpn_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads rpn network variable weights from ckpt file

        """
        
        saver = self.saver_ops['rpn_network']
        saver.restore(self.sess,file)
    
    def load_rcnn_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads rcnn network variable weights from ckpt file

        """
        
        rcnn_saver = self.saver_ops['rcnn_network']
        rcnn_saver.restore(self.sess,file)

    def load_rcnn_pt_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads rcnn fc layer variable weights from ckpt file

        """
        
        fc_saver = self.saver_ops['pt_fc_layers']
        fc_saver.restore(self.sess,file)

    def load_rpn_training_ops_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads rpn training variables from ckpt file

        """
        
        saver = self.saver_ops['rpn_training_op']
        saver.restore(self.sess,file)

    def load_rcnn_training_ops_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        loads rcnn training variables from ckpt file

        """
        
        
        saver = self.saver_ops['rcnn_training_op']
        saver.restore(self.sess,file)

    def save_base_model_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves vgg variable weights to ckpt file

        """
        
        
        saver = self.saver_ops['base_model']
        saver.save(self.sess,file)

    def save_rpn_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves rpn network variable weights to ckpt file

        """
        saver = self.saver_ops['rpn_network']
        saver.save(self.sess,file)
    
    def save_rcnn_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves rcnn network variable weights to ckpt file

        """
        rcnn_saver = self.saver_ops['rcnn_network']
        rcnn_saver.save(self.sess,file)
    
    def save_rcnn_pt_network_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves rcnn network fc variable weights to ckpt file

        """
        
        
        fc_saver = self.saver_ops['pt_fc_layers']
        fc_saver.save(self.sess,file)

    def save_rpn_training_ops_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves rpn training variables to ckpt file

        """
        
        
        saver = self.saver_ops['rpn_training_op']
        saver.save(self.sess,file)
    
    def save_rcnn_training_ops_weights(self,file):
        """
        inputs:

        'file': path to ckpt file 

        saves rcnn training variables to ckpt file

        """
        
        saver = self.saver_ops['rcnn_training_op']
        saver.save(self.sess,file)

    def merge_summaries(self):
        """
        inputs: 
        
        creates merged summary op for tensorboard
        adds graph to summary writer
        """
        
        self.summary_ops['merged_summary_op'] = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.cfg.NET.TENSORBOARD_DIR)
        self.writer.add_graph(self.sess.graph)
    
    def initialize_variables_clean(self):
        """
        inputs:

        loads the vgg weights and the rcnn fc layer weights
        from the imagenet pretrained vgg ckpt file

        initializes all other network variables

        """

        self.load_base_model_weights(self.cfg.NET.BASE_MODEL_WEB_CKPT)
        self.load_rcnn_pt_network_weights(self.cfg.NET.BASE_MODEL_WEB_CKPT)
        for _,val in self.init_ops.items():
            self.sess.run(val)
    
    def initialize_varaibles_from_ckpts(self):
        """
        inputs: 

        loads all network weights from their pre-specified 
        ckpt files

        """

        self.load_base_model_weights(self.cfg.NET.BASE_MODEL_CKPT)
        self.load_rpn_network_weights(self.cfg.NET.RPN_NETWORK_CKPT)
        self.load_rcnn_network_weights(self.cfg.NET.RCNN_NETWORK_CKPT)
        self.load_rcnn_pt_network_weights(self.cfg.NET.RCNN_PT_NETWORK_CKPT)
        if self.is_training:
            self.load_rpn_training_ops_weights(self.cfg.NET.RPN_TRAINING_CKPT)
            self.load_rcnn_training_ops_weights(self.cfg.NET.RCNN_TRAINING_CKPT)
    
    def save_weights_to_ckpt(self):
        """
        inputs:

        saves all network weights to their pre-specified ckpt files
        """
        self.save_base_model_weights(self.cfg.NET.BASE_MODEL_CKPT)
        self.save_rpn_network_weights(self.cfg.NET.RPN_NETWORK_CKPT)
        self.save_rcnn_network_weights(self.cfg.NET.RCNN_NETWORK_CKPT)
        self.save_rcnn_pt_network_weights(self.cfg.NET.RCNN_PT_NETWORK_CKPT)
        self.save_rpn_training_ops_weights(self.cfg.NET.RPN_TRAINING_CKPT)
        self.save_rcnn_training_ops_weights(self.cfg.NET.RCNN_TRAINING_CKPT)

    def set_rpn_learning_rate(self,lr):
        """
        inputs:

        'lr': learning rate to set for rpn optimization

        assigns 'lr' to rpn learning rate variable

        """
        rpn_lr = self.learning_rates['rpn_learning_rate']
        self.sess.run(tf.assign(rpn_lr,lr))

    def set_rcnn_learning_rate(self,lr):
        """
        inputs:

        'lr': learning rate to set for rcnn optimization

        assigns 'lr' to rcnn learning rate variable

        """        
        rcnn_lr = self.learning_rates['rcnn_learning_rate']
        self.sess.run(tf.assign(rcnn_lr,lr))

    def train_step(self,training_image,training_gt_boxes):
        """
        inputs:

        'training_image': [1,h,w,3] np array in BGR format
        'gt_boxes': [K,5] np array in format [x,y,w,h,class]

        runs one training step of joint training. both rpn and rcnn
        networks are trained simultaneously

        """
        
        feed_dict = {self.image:training_image,self.gt_boxes:training_gt_boxes}
        
        self.sess.run([self.training_ops['rpn_training_op'],self.training_ops['rcnn_training_op']],
                        feed_dict=feed_dict)
        
    def train_step_with_summary(self,training_image,training_gt_boxes,iters):
        """
        inputs:

        'training_image': [1,h,w,3] np array in BGR format
        'gt_boxes': [K,5] np array in format [x,y,w,h,class]
        'iters': training iteration number 

        runs one training step of joint training. both rpn and rcnn
        networks are trained simultaneously

        updates tensorboard stats

        """
        feed_dict = {self.image:training_image,self.gt_boxes:training_gt_boxes}

        s,rpn_loss,rcnn_loss,_,__ = self.sess.run([self.summary_ops['merged_summary_op'],
                    self.losses['rpn/total_loss'],self.losses['rcnn/total_loss'].
                    self.training_ops['rpn_training_op'],self.training_ops['rcnn_training_op']],
                        feed_dict=feed_dict)
        self.writer.add_summary(s,iters)
        print('RPN LOSS: ',rpn_loss)
        print('RCNN LOSS: ',rcnn_loss)

    def train_step_debug(self,training_image,training_gt_boxes):
        """
        inputs:

        'training_image': [1,h,w,3] np array in BGR format
        'gt_boxes': [K,5] np array in format [x,y,w,h,class]

        runs one training step of joint training. both rpn and rcnn
        networks are trained simultaneously

        prints all network tensor shapes for debugging

        """
        feed_dict = {self.image:training_image,self.gt_boxes:training_gt_boxes}
        
        _,__,fm_sh,ab_sh,rpn_net_obj_sh,rpn_net_bbox_sh,training_anchors_sh,anchor_labels_training_sh, \
        bbox_labels_training_sh,adjusted_anchors_sh, \
        clipped_anchors_sh,pre_nms_anchors_sh,nms_anchors_sh, \
        post_nms_anchors_sh,anchor_boxes_rcnn_training_sh, \
        roi_sh,rcnn_cls_sh,rcnn_bbox_sh,rpn_loss_obj_scores_sh,\
        rpn_loss_bbox_scores_sh, non_zero_cls_label_inds,roi_selected_inds,\
        rcnn_loss_cls_scores_sh,rcnn_loss_bbox_scores_sh,rcnn_loss_bbox_pos_sh = self.sess.run([self.training_ops['rpn_training_op'],self.training_ops['rcnn_training_op'],
                        self.fm_sh,self.ab_sh,self.rpn_net_obj_sh,self.rpn_net_bbox_sh,
                        self.training_anchors_sh,self.anchor_labels_training_sh,
                        self.bbox_labels_training_sh,self.adjusted_anchors_sh,
                        self.clipped_anchors_sh,self.pre_nms_anchors_sh,self.nms_anchors_sh,
                        self.post_nms_anchors_sh,self.anchor_boxes_rcnn_training_sh,
                        self.roi_sh,self.rcnn_cls_sh,self.rcnn_bbox_sh,self.rpn_loss_obj_scores_sh,
                        self.rpn_loss_bbox_scores_sh,self.non_zero_cls_label_inds,self.roi_selected_inds,
                        self.rcnn_loss_cls_scores_sh,self.rcnn_loss_bbox_scores_sh,
                        self.rcnn_loss_bbox_pos_sh],feed_dict=feed_dict)
        
        print('Feature Map Shape: ',fm_sh)
        print('Anchor Boxes Shape: ',ab_sh)
        print('RPN Net Object Layer Shape: ',rpn_net_obj_sh)
        print('RPN Net Bbox Layer Shape: ',rpn_net_bbox_sh)
        print('Sampled Anchors For RPN Training Shape: ',training_anchors_sh)
        print('Sampled Anchors Object Labels for RPN Training Shape: ',anchor_labels_training_sh)
        print('Sampled Anchors Bbox Labels for RPN Training Shape: ',bbox_labels_training_sh)
        print('Adjusted Anchors in Select Rois Shape: ',adjusted_anchors_sh)
        print('Clipped Anchors Shape: ',clipped_anchors_sh)
        print('Pre NMS Anchors Shape: ',pre_nms_anchors_sh)
        print('NMS Anchors Shape: ',nms_anchors_sh)
        print('Post NMS Anchors Shape: ',post_nms_anchors_sh)
        print('Anchors Boxes for RCNN Training Sh: ',anchor_boxes_rcnn_training_sh)
        print('ROI Crops Shape: ',roi_sh)
        print('RCNN Network Classification Layer Shape: ',rcnn_cls_sh)
        print('RCNN Network Bbox Layer Shape: ',rcnn_bbox_sh)
        print('RPN Loss Object Scores Shape: ',rpn_loss_obj_scores_sh)
        print('RPN Loss Bbox Scores Shape: ',rpn_loss_bbox_scores_sh)
        print('Non Zero Class Label Inds: ',non_zero_cls_label_inds)
        print('RoI == 1 Selected Class Label Inds: ',roi_selected_inds)
        print('RCNN Loss Classification Scores Shape: ',rcnn_loss_cls_scores_sh)
        print('RCNN Loss Bbox Scores Shape: ',rcnn_loss_bbox_scores_sh)
        print('RCNN Loss Positive Bbox Scores Shape: ',rcnn_loss_bbox_pos_sh)

    def evaluate_image(self,image):
        """
        inputs:

        'image': [1,h,w,3] np array in BGR format
        
        returns:

        'rois': [K,4] array of bounding boxes in [x1,y1,x2,y2] format, only rois 
        that are labeled as a non-background class are returned
        'cls_labels': the category labels for the returned rois
        'max_probs': the model probability assigned to the category

        runs the network and gets the output of the rcnn network layer. extracts
        the model classification. if necessary reduces the number of detections
        to the top N as specified by coco evaluations specs

        """


        feed_dict = {self.image:image}
        cls_probs,bbox_adjs,rois = self.sess.run([self.layers['fast_rcnn_network/cls_probs'],
                                            self.layers['fast_rcnn_network/bbox_adjs'],self.rois],
                                            feed_dict=feed_dict)
        cls_probs = np.reshape(cls_probs,(-1,self.cfg.NET.NUM_CLASSES))
        bbox_adjs = np.reshape(bbox_adjs,(-1,self.cfg.NET.NUM_CLASSES-1,4))
        cls_labels = np.argmax(cls_probs,axis=1)
        max_probs = np.max(cls_probs,axis=1)
        pos_dets = np.where(cls_labels > 0)[0]
        
        if len(pos_dets) > self.cfg.TEST.MAX_BOXES_FOR_DETECTION:
            det_inds = top_k_inds(max_probs[pos_dets],self.cfg.TEST.MAX_BOXES_FOR_DETECTION)
            pos_dets = pos_dets[det_inds]
        
        max_probs = max_probs[pos_dets]
        cls_labels = cls_labels[pos_dets] - 1
        rois = rois[pos_dets,:]
        roi_bbox_adjs = bbox_adjs[pos_dets,cls_labels,:]
        rois = adjust_anchors(rois,roi_bbox_adjs)
        rois = clip_rois(rois,np.shape(image)[1],np.shape(image)[2])

        return rois,cls_labels,max_probs


