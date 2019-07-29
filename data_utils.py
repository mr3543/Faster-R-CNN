import numpy as np
import cv2
from pycocotools.coco import COCO
import os.path as osp
import params

data_dir = './image_data'

def get_ann_file(data_name):
    return '{}/annotations/instances_{}.json'.format(data_dir,data_name)

def get_image_permutation(random_seed,data_name):
    np.random.seed(random_seed)
    ann_file = get_ann_file(data_name) 
    coco = COCO(ann_file)
    ids = coco.getImgIds()
    return np.random.permutation(ids)

def get_training_data(step,image_permutation,image_batch_size,data_name):
    
    ann_file = get_ann_file(data_name) 
    coco = COCO(ann_file)
    
    index_start = (step*image_batch_size) % len(image_permutation)
    index_end = ((step + 1)*image_batch_size) % len(image_permutation)
    print('making new training image batch from {} to {}'.format(index_start,index_end))
    if index_start < index_end:
      im_list = image_permutation[index_start:index_end]
    else:
      im_list = np.append(image_permutation[index_end:],image_permutation[:index_start]) 
      
    # load the coco image meta-info
    im_anns = coco.loadImgs(im_list)
    resized_images = []
    gt_boxes = []
    im_infos = []
    print('loading {} image meta-info dicts'.format(len(im_anns)))
    for i,im_ann in enumerate(im_anns):
      ann_ids = coco.getAnnIds(imgIds = im_list[i],iscrowd = False)
      anns = coco.loadAnns(ann_ids)
    #  print('loading image {} with {} annotations'.format(im_list[i],len(anns)))
      if len(anns) == 0:
     #   print('found image w/ 0 annotations... skipping')
        continue
      image_path = osp.join(data_dir,data_name,im_ann['file_name'])
      img = cv2.imread(image_path) - params.dataset_img_mean
      im_width = im_ann['width']
      im_height = im_ann['height']
      if im_width < im_height:
        new_width = params.shortest_side
        new_height = np.round(im_height*(new_width/im_width)).astype(np.int32)
      else:
        new_height = params.shortest_side
        new_width = np.round(im_width*(new_height/im_height)).astype(np.int32)

      res_im = np.array(cv2.resize(img,dsize=(new_width,new_height),interpolation=cv2.INTER_CUBIC))
      res_im = res_im[np.newaxis,:,:,:]
      im_info = {'width':new_width,'height':new_height,'id':im_list[i]}
      im_infos.append(im_info)
    
      # now scale and label the gt boxes
      scaling_factor = new_width/im_width
      gts = []
      for j,ann in enumerate(anns):
          bbox = np.array(ann['bbox'])*scaling_factor
          b_x,b_y,b_w,b_h = bbox[0],bbox[1],bbox[2],bbox[3]
          if (ann['area'] > 0 and b_x >= 0 and b_y>= 0 and b_x <= new_width and b_y <= new_height
              and b_w >= 0 and b_h >= 0 and b_w <= new_width and b_h <= new_height):
                bbox = np.append(bbox,ann['category_id'])
                gts.append(bbox)
      if len(gts) > 0:
        gt_boxes.append(np.array(gts))
        resized_images.append(res_im)
  
    return resized_images,gt_boxes,im_infos

def _get_dataset_img_mean(data_name):
  ann_file = get_ann_file(data_name)
  coco = COCO(ann_file)
  im_anns = coco.loadImgs(coco.getImgIds())
  img_mean = 0
  for ann in im_anns:
    im_path = osp.join(data_dir,data_name,ann['file_name'])
    img = cv2.imread(im_path)
    img_mean = img_mean + np.mean(img)
    
  img_mean = img_mean/len(im_anns)
  return img_mean
