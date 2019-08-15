import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO
import os.path as osp
import cv2

class Imdb(object):

    def __init__(self,data_dir,data_name,cfg):
        """
        inputs: 

        'data_dir': path to directory containing image data
        'data_name': dataset name
        'cfg': edict object with params

        Imdb object encapsulates the dataset and the methods to retreive and format
        images. 
        """


        self.data_dir = data_dir
        self.data_name = data_name
        self.ann_file = '{}/annotations/instances_{}.json'.format(data_dir,data_name)
        self.coco = COCO(self.ann_file)
        self.blob_image_index = 0
        self.id_perm = self.coco.getImgIds()
        self.cfg = cfg
        self.data_mean = self.cfg.DATA.DATA_MEAN
        cat_ids = self.coco.getCatIds()
        self.coco_to_model_dict = dict(zip(cat_ids,np.arange(len(cat_ids))))
        self.model_to_coco_dict = dict(zip(np.arange(len(cat_ids)),cat_ids))

    def get_id_perm(self):
        """
        inputs: 

        returns: returns the random permutation of dataset image ids
        """
        return self.id_perm
    
    def get_image(self,image_id):
        """
        inputs:

        'image_id': coco id of desired image

        returns: 

        'img': [h,w,3] np array in BGR format
        """

        im_ann = self.coco.loadImgs(image_id)
        file_name = im_ann[0]['file_name']
        image_path = osp.join(self.data_dir,self.data_name,file_name)
        img = cv2.imread(image_path)
        return img

    def scale_and_format_image(self,image):
        """
        inputs: 

        'image': [h,w,3] np array 

        returns:

        'res_im': [1,new_h,new_w,3] np array 

        scales image according to cfg specs, adds new 0th dimension to image
        """
        im_height = np.shape(image)[0]
        im_width = np.shape(image)[1]
        
        if im_width < im_height:
            new_width = self.cfg.DATA.SHORTEST_SIDE_SCALE
            new_height = np.round(im_height*(new_width/im_width)).astype(np.int32)
        else:
            new_height = self.cfg.DATA.SHORTEST_SIDE_SCALE
            new_width = np.round(im_width*(new_height/im_height)).astype(np.int32)
        
        res_im = np.array(cv2.resize(image,dsize=(new_width,new_height),interpolation=cv2.INTER_CUBIC))
        res_im = res_im - self.data_mean
        res_im = res_im[np.newaxis,:,:,:]
        return res_im

    def shuffle_images(self,random_seed):
        """
        input: 
    
        'random_seed': seed to set np random with 

        sets self.id_perm to new random permuation of dataset ids
        """
        np.random.seed(random_seed)
        self.id_perm = np.random.permutation(self.id_perm)

    def make_image_blob(self,blob_size):
        """
        input:

        'blob_size': batch size of images to retreive from disc

        returns:

        'resised_images': list of images [1,h,w,3] np arrays 

        fetches batch of images from disc formats images 
        according to cfg specs
        """

        index_start = self.blob_image_index
        index_end = (index_start + blob_size)%len(self.id_perm)
        self.blob_image_index = index_end
        if index_start < index_end:
            im_list = self.id_perm[index_start:index_end]
        else:
            im_list = np.append(self.id_perm[index_end:],self.id_perm[:index_start])

        im_anns = self.coco.loadImgs(im_list)
        resized_images = []
        for i,im_ann in enumerate(im_anns):
            image_path = osp.join(self.data_dir,self.data_name,im_ann['file_name'])
            img = cv2.imread(image_path) - self.data_mean
            im_height = im_ann['height']
            im_width = im_ann['width']

            if im_width < im_height:
                new_width = self.cfg.DATA.SHORTEST_SIDE_SCALE
                new_height = np.round(im_height*(new_width/im_width)).astype(np.int32)
            else:
                new_height = self.cfg.DATA.SHORTEST_SIDE_SCALE
                new_width = np.round(im_width*(new_height/im_height)).astype(np.int32)
            
            res_im = np.array(cv2.resize(img,dsize=(new_width,new_height),interpolation=cv2.INTER_CUBIC))
            res_im = res_im[np.newaxis,:,:,:]
            resized_images.append(res_im)

            return resized_images
    
    def make_training_blob(self,blob_size):
        """
        inputs: 

        'blob_size': batch size of images to retreive from disc

        returns: 

        'resized_images': list of images [1,h,w,3] np arrays 
        'gt_boxes': list of [K_i,5] np arrays which contain the K_i gt bounding boxes 
        for image i. boxes are in the format [x,y,w,h,class]
        'im_ids': the ids of the images in the batch
        
        fetches a batch of images along with gt boxes and ids from disc. formats
        images according to cfg specs

        """

        index_start = self.blob_image_index
        index_end = (index_start + blob_size)%len(self.id_perm)
        self.blob_image_index = index_end
        if index_start < index_end:
            im_list = self.id_perm[index_start:index_end]
        else:
            im_list = np.append(self.id_perm[index_end:],self.id_perm[:index_start])

        im_anns = self.coco.loadImgs(im_list)
        gt_boxes = []
        resized_images = []
        im_ids = []
        for i,im_ann in enumerate(im_anns):
            ann_ids = self.coco.getAnnIds(imgIds=im_list[i],iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            if len(anns) == 0:
                continue
            image_path = osp.join(self.data_dir,self.data_name,im_ann['file_name'])
            img = cv2.imread(image_path) - self.data_mean
            im_height = im_ann['height']
            im_width = im_ann['width']

            if im_width < im_height:
                new_width = self.cfg.DATA.SHORTEST_SIDE_SCALE
                new_height = np.round(im_height*(new_width/im_width)).astype(np.int32)
            else:
                new_height = self.cfg.DATA.SHORTEST_SIDE_SCALE
                new_width = np.round(im_width*(new_height/im_height)).astype(np.int32)
            
            res_im = np.array(cv2.resize(img,dsize=(new_width,new_height),interpolation=cv2.INTER_CUBIC))
            res_im = res_im[np.newaxis,:,:,:]
            w_scale = new_width/im_width
            h_scale = new_height/im_height

            gts = []
            for j,ann in enumerate(anns):
                bbox = np.array(ann['bbox'])
                bbox[0] = bbox[0]*w_scale
                bbox[1] = bbox[1]*h_scale
                bbox[2] = bbox[2]*w_scale
                bbox[3] = bbox[3]*h_scale
                b_x,b_y,b_w,b_h = bbox[0],bbox[1],bbox[2],bbox[3]
                if (ann['area'] > 0 and b_x >= 0 and b_y>= 0 and b_x <= new_width and b_y <= new_height
                and b_w >= 0 and b_h >= 0 and b_w <= new_width and b_h <= new_height):
                    model_id = self.coco_to_model_dict[ann['category_id']]
                    bbox = np.append(bbox,model_id)
                    gts.append(bbox)
            if len(gts) > 0:
                gt_boxes.append(np.array(gts))
                resized_images.append(res_im)
                im_ids.append(im_ann['id'])
        
        return resized_images,gt_boxes,im_ids
    
    def get_dataset_img_mean(self):
        """
        inputs: 

        returns: 

        'img_mean': mean pixel value among images in the dataset 
        """

        im_anns = self.coco.loadImgs(coco.getImgIds())
        img_mean = 0
        for ann in im_anns:
            im_path = osp.join(self.data_dir,self.data_name,ann['file_name'])
            img = cv2.imread(im_path)
            img_mean = img_mean + np.mean(img)

        img_mean = img_mean/len(im_anns)
        return img_mean
