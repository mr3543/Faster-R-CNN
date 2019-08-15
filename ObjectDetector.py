import tensorflow as tf
import numpy as np
import json
import os.path as osp
import sys
import gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time

class ObjectDetector(object):
    def __init__ (self,net,imdb,cfg):
        """ 
        inputs: 

        'net': Network object
        'imdb': Imdb object
        'cfg': edict object
        
        """

        self.net = net
        self.imdb = imdb
        self.cfg = cfg
        
    def train(self):
        """
        inputs: 

        returns:

        performs training with the self.net Network and self.imdb Imdb. 
        Tensorflow graph is constructed and the base model weights are loaded
        from pretrained imagenet model. 

        training proceeds following the joint training strategy where rpn and rcnn
        are both trained simultaneously. 

        after the specified number of iterations the learning rate is reduced 

        ckpt and tensorboard updates are made itermittently 

        """
        self.net.construct_graph()
        self.net.initialize_variables_clean()
        self.net.set_rpn_learning_rate(self.cfg.TRAIN.RPN_LR1)
        self.net.set_rcnn_learning_rate(self.cfg.TRAIN.RCNN_LR1)
        self.imdb.shuffle_images(self.cfg.TRAIN.RANDOM_SEED)
        i=0
        while i < self.cfg.TRAIN.LR1_BLOBS:
            image_blob,roi_blob,image_ids = self.imdb.make_training_blob(self.cfg.TRAIN.BLOB_SIZE)
            j=0
            while j < len(image_blob):
                self.net.train_step(image_blob[j],roi_blob[j])
                if i*len(image_blob) + j % 100 == 0:
                    self.net.save_weights_to_ckpt()
                    print('TRAINING ON IMAGE NO {} IN BATCH {}\n'.format(j,i))
                    self.net.train_step_with_summary(image_blob[j],roi_blob[j],i*len(image_blob)+j)
                    print('SAVING LR1 STEP: ',i*len(image_blob)+j)
                j+=1
            del image_blob,roi_blob
            gc.collect()
            i+=1
        self.net.set_rpn_learning_rate(self.cfg.TRAIN.RPN_LR2)
        self.net.set_rcnn_learning_rate(self.cfg.TRAIN.RCNN_LR2)
        i=0 
        while i < self.cfg.TRAIN.LR2_BLOBS:
            image_blob,roi_blob = self.imdb.make_blob(self.cfg.TRAIN.BLOB_SIZE)
            j=0
            while j < len(image_blob):
                self.net.train_step(image_blob[j],roi_blob[j])
                j+=1
                if i*len(image_blob) + j % 100 == 0:
                    self.net.save_weights_to_ckpt()
                    print('TRAINING ON IMAGE NO {} IN BATCH {}\n'.format(j,i))
                    self.train_step_with_summary(image_blob[j],roi_blob[j])
                    print('SAVING LR2 STEP: ',i*len(image_blob)+j)
                    j+=1
            del image_blob,roi_blob
            gc.collect()
            i+=1
        self.net.save_weights_to_ckpt()
        

    def evaluate(self,results_file,load_net_weights=True):
        """ 
        inputs:

        'results_file': path to .json file where detection results are written
        'load_net_weights': boolean denoting whether to load network weights before
        running detection task

        """
        if load_net_weights:
            self.net.construct_graph()
            self.net.initialize_varaibles_from_ckpts()
        
        self.write_results_file(results_file)
        self.coco_detection(results_file)

    
    def write_results_file(self,results_file):
        """
        inputs:

        'results_file': path to .json file where detection results are written

        loops thru images in self.imdb, runs detection on self.net and 
        writes results to json file.
        """

        results = []
        image_ids = self.imdb.get_id_perm()
        for j,im_id in enumerate(image_ids):
            image = self.imdb.get_image(im_id)
            orig_height = np.shape(image)[0]
            orig_width = np.shape(image)[1]
            image = self.imdb.scale_and_format_image(image)
            new_height = np.shape(image)[1]
            new_width = np.shape(image)[2]

            boxes,labels,scores = self.net.evaluate_image(image)

            boxes[:,2] = boxes[:,2] - boxes[:,0]
            boxes[:,3] = boxes[:,3] - boxes[:,1]

            boxes[:,0] = boxes[:,0]*(orig_width/new_width)
            boxes[:,1] = boxes[:,1]*(orig_height/new_height)
            boxes[:,2] = boxes[:,2]*(orig_width/new_width)
            boxes[:,3] = boxes[:,3]*(orig_height/new_height)

            for i,box in enumerate(boxes):
                cat_id = self.imdb.model_to_coco_dict[labels[i]]

                d = {'image_id':im_id,'category_id':cat_id,
                'bbox':list(boxes[i,:].astype(np.float64)),'score':scores[i].astype(np.float64)}
                results.append(d)
            if j % 100 == 0:
                print('done {} images'.format(j))

        with open(results_file,'w') as f:
            json.dump(results,f)


    def coco_detection(self,results_file):
        """
        inputs:

        'results_file': path to .json file to read results from

        uses coco api to print detection performance stats

        """
        coco_det = self.imdb.coco.loadRes(results_file)
        coco_eval = COCOeval(self.imdb.coco,coco_det)
        coco_eval.params.useSegm = False
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


