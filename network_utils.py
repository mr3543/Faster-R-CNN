import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import params



def get_anchor_boxes(im_info,feature_map):

  with tf.variable_scope('build_anchors'):
    # get image and feature map dims
    im_height = im_info[0]
    im_width = im_info[1]
    fm_height = tf.cast(tf.shape(feature_map)[1],dtype = tf.float32)
    fm_width = tf.cast(tf.shape(feature_map)[2],dtype = tf.float32)

    # compute the top left anchor centers on the orginal image
    width_ratio = tf.math.floordiv(im_width,fm_width)
    height_ratio = tf.math.floordiv(im_height,fm_height)
    x_ctr = tf.math.floordiv(width_ratio,2)
    y_ctr = tf.math.floordiv(height_ratio,2)

    # compute the anchor centers across entire image
    x_ctrs = width_ratio*tf.range(0,fm_width) + x_ctr
    y_ctrs = height_ratio*tf.range(0,fm_height) + y_ctr

    # construct grid of center coordinates
    centers = tf.transpose(tf.meshgrid(x_ctrs,y_ctrs))
    centers_t = tf.transpose(centers,perm=(1,0,2))
    centers = tf.reshape(centers_t,(-1,2))
    
    
    # make combinations of anchor scales and anchor ratios
    anchor_combs = tf.transpose(tf.meshgrid(params.anchor_scales,params.anchor_ratios))
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
    anchors = tf.concat((centers_,wh_),axis=1)

  return anchors


def get_anchor_labels(anchor_boxes,gt_boxes,pos_thresh,neg_thresh_lo,neg_thresh_hi,ignore_cbas=True):
    
    # label the anchors object or not object
    anchor_labels = np.zeros(np.shape(anchor_boxes)[0],dtype = np.float32)
    bbox_adjs = np.zeros(np.shape(anchor_boxes),dtype = np.float32)

    # get the ious and label according to params
    ious = _compute_ious(anchor_boxes,gt_boxes[:,:5])
    max_ious = np.max(ious,axis=1)
    max_gt_boxes = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious > pos_thresh)[0]
    neg_anchors = np.where(np.logical_and(max_ious > neg_thresh_lo, max_ious < neg_thresh_hi))[0]
    
    anchor_labels[pos_anchors] = 1
    anchor_labels[neg_anchors] = -1
    anchor_labels[np.argmax(max_ious)] = 1
    
    # if ingnore_cbas is set we ignore cross-boundary anchors and set these labels to 0
    if ignore_cbas:
      a_boxes = _centers_to_corners(anchor_boxes)
      inds = np.where(np.any(a_boxes < 0,axis=1))[0]
      anchor_labels[inds] = 0
    
    # compute the bbox adjs between the anchors and the gt_boxes
    bbox_adjs[:,0] = ((gt_boxes[max_gt_boxes,0] + gt_boxes[max_gt_boxes,2]/2) - anchor_boxes[:,0])/anchor_boxes[:,2]
    bbox_adjs[:,1] = ((gt_boxes[max_gt_boxes,1] + gt_boxes[max_gt_boxes,3]/2) - anchor_boxes[:,1])/anchor_boxes[:,3]
    
    bbox_adjs[:,2] = np.log(gt_boxes[max_gt_boxes,2]/anchor_boxes[:,2])
    bbox_adjs[:,3] = np.log(gt_boxes[max_gt_boxes,3]/anchor_boxes[:,3])
    
    # lables the anchors with coco category classes
    cls_labels = np.zeros(np.shape(anchor_boxes)[0],dtype=np.int32)
    cls_labels[pos_anchors] = gt_boxes[max_gt_boxes[pos_anchors],4] + 1
    
    return anchor_labels,bbox_adjs,cls_labels

def _compute_ious(a_boxes,gt_boxes):
    
    # convert the boxes to [x1,y1,x2,y2] format
    a_boxes = _centers_to_corners(a_boxes)
    gt_boxes = _xy_to_corners(gt_boxes)
    l = np.shape(gt_boxes)[0] 
      
    # compute the coordinates of the intersection boxes
    x1 = np.array([np.max(np.stack((np.repeat(a_boxes[i,0],l),gt_boxes[:,0]),axis=1),axis=1) 
                    for i in range(len(a_boxes))])
    y1 = np.array([np.max(np.stack((np.repeat(a_boxes[i,1],l),gt_boxes[:,1]),axis=1),axis=1) 
                    for i in range(len(a_boxes))])
    x2 = np.array([np.min(np.stack((np.repeat(a_boxes[i,2],l),gt_boxes[:,2]),axis=1),axis=1) 
                    for i in range(len(a_boxes))])
    y2 = np.array([np.min(np.stack((np.repeat(a_boxes[i,3],l),gt_boxes[:,3]),axis=1),axis=1) 
                    for i in range(len(a_boxes))])
    
    # get widths and heights of the intersection boxes
    overlaps = np.stack((x1,y1,x2,y2),axis=2)
    widths = overlaps[:,:,2] - overlaps[:,:,0]
    heights = overlaps[:,:,3] - overlaps[:,:,1]
    overlap_areas = widths*heights
    # filter out boxes that don't intersect
    overlap_areas[np.where(widths < 0)] = 0
    overlap_areas[np.where(heights < 0)] = 0

    # comupte the final iou
    area_a = (a_boxes[:,2] - a_boxes[:,0])*(a_boxes[:,3] - a_boxes[:,1])
    area_gt = (gt_boxes[:,2] - gt_boxes[:,0])*(gt_boxes[:,3] - gt_boxes[:,1])
    union_areas = [a + area_gt for a in area_a]
    ious = overlap_areas/(union_areas - overlap_areas)
    
    return ious

def _centers_to_corners(a_boxes):
  # change box format from x_c,y_c,w,h to x1,y1,x2,y2
  
  x1 = np.round(a_boxes[:,0] - a_boxes[:,2]/2)
  y1 = np.round(a_boxes[:,1] - a_boxes[:,3]/2)
  x2 = np.round(a_boxes[:,0] + a_boxes[:,2]/2)
  y2 = np.round(a_boxes[:,1] + a_boxes[:,3]/2)
  
  return np.stack((x1,y1,x2,y2),axis=1)

def _xy_to_corners(gt_boxes):
  # change box format from x,y,w,h to x1,y1,x2,y2
  x1 = gt_boxes[:,0]
  y1 = gt_boxes[:,1]
  x2 = gt_boxes[:,0] + gt_boxes[:,2]
  y2 = gt_boxes[:,1] + gt_boxes[:,3]
  
  return np.stack((x1,y1,x2,y2),axis=1)


def sample_anchors_for_training(anchor_labels,mini_batch_size,prop_pos):
  # sample positive and negative anchors accroding to params
  # returns an array of type np.float32, where positive anchors selected are
  # denoted by 1, negative anchors selected are denoted by -1, anchors not
  # selected are denoted by 0 
  
  anchor_mask = np.zeros(np.shape(anchor_labels)[0],dtype=np.int32)
  num_anchors = np.shape(anchor_labels)[0]
  num_pos_anchors = np.round(mini_batch_size * prop_pos).astype(np.int32)
  pos_anchors = np.where(anchor_labels == 1)[0]
  neg_anchors = np.where(anchor_labels == -1)[0]

  # if more than desired number of pos anchors sample randomly 
  if len(pos_anchors) >= num_pos_anchors:
    pos_anchors = np.random.choice(pos_anchors,num_pos_anchors,replace = False)
  
  anchor_mask[pos_anchors] = 1
  num_neg_anchors = mini_batch_size - len(pos_anchors)
  
  if len(neg_anchors) >= num_neg_anchors:
    neg_anchors = np.random.choice(neg_anchors,num_neg_anchors,replace = False)
  
  anchor_mask[neg_anchors] = -1
  
  return anchor_mask

def make_anchor_ce_labels(anchor_labels,anchor_mask):
  # remake anchor labels from vector of 1s and -1s to 
  # one hot encoding
  anchor_labels_ce = anchor_labels[np.where(anchor_mask != 0)]
  anchor_pos_labels = np.zeros(len(anchor_labels_ce),dtype = np.int32)
  anchor_neg_labels = np.zeros(len(anchor_labels_ce),dtype = np.int32)
  anchor_pos_labels[np.where(anchor_labels_ce == 1)] = 1
  anchor_neg_labels[np.where(anchor_labels_ce == -1)] = 1
  return np.array(np.stack((anchor_pos_labels,anchor_neg_labels),axis =1))

def select_rois(anchor_boxes,gt_boxes,bbox_adjs,obj_scores,cls_preds,train_flag):
  # select rois for rcnn 
  anchor_boxes = tf.py_func(_centers_to_corners,[anchor_boxes],tf.float32)
  #anchor_boxes = tf.py_func(_adjust_rois,[anchor_boxes,bbox_adjs,cls_preds],tf.float32)
  selected_inds = tf.expand_dims(tf.image.non_max_suppression(anchor_boxes,obj_scores,2000),axis=1)
  anchor_rois = tf.gather_nd(anchor_boxes,selected_inds)
  roi_adjs = tf.gather_nd(bbox_adjs,selected_inds)
  roi_scores = tf.gather_nd(obj_scores,selected_inds)
  
  roi_batch,batch_labels,bbox_adjs = tf.py_func(_make_roi_batch,[anchor_rois,roi_adjs,gt_boxes,
                                                    roi_scores,train_flag],[tf.float32,tf.int32,tf.float32])
  
  return roi_batch,batch_labels,bbox_adjs

def _make_roi_batch(anchor_rois,roi_adjs,gt_boxes,roi_scores,train_flag):
  # if the training flag is set we label the rois with fg/bg labels
  # then sample a batch with a specified fg/bg ratio
  # if training is not set we take the top N according to predicted object score from the rpn layer
  if train_flag:
    roi_fg_labels,bbox_adjs,roi_cls_labels = get_anchor_labels(anchor_rois,gt_boxes,params.rcnn_pos_iou_thresh,
                                      params.rcnn_neg_iou_thresh_lo,params.rcnn_neg_iou_thresh_hi,ignore_cbas=False)
    
    roi_mask = sample_anchors_for_training(roi_fg_labels,params.rcnn_mini_batch_size,params.rcnn_prop_pos)
    roi_inds = np.where(roi_mask != 0)[0]
    return anchor_rois[roi_inds],roi_cls_labels[roi_inds],bbox_adjs[roi_inds]
  else:
    sorted_inds = np.argsort(roi_scores)
    adjusted_boxes = _adjust_rois((anchor_rois[sorted_inds])[:params.rcnn_top_N],roi_adjs)
    return adjusted_boxes,None,None
  
def _adjust_rois(boxes,offsets,preds):
  
  inds = np.where(preds == 1)[0]
  anch_w = boxes[inds,2]-boxes[inds,0]
  anch_h = boxes[inds,3]-boxes[inds,1]
  
  boxes[inds,0] = boxes[inds,0] + offsets[inds,0]*anch_w
  boxes[inds,1] = boxes[inds,1] + offsets[inds,1]*anch_h
  
  new_ws = np.exp(offsets[inds,2])*anch_w
  new_hs = np.exp(offsets[inds,3])*anch_h
  
  boxes[inds,2] = boxes[inds,0] + new_ws
  boxes[inds,3] = boxes[inds,1] + new_hs
  
  return boxes

def roi_pooling(rois,feature_map,im_info):
  im_height,im_width = im_info[0],im_info[1]
  rois = tf.py_func(_clip_rois,[rois,im_height,im_width],tf.float32)
  fm_height,fm_width = tf.shape(feature_map)[1],tf.shape(feature_map)[2]
  roi_fm_regions = tf.py_func(_get_roi_regions,[rois,fm_height,fm_width,im_height,im_width],tf.float32)
  box_inds = tf.zeros(tf.shape(roi_fm_regions)[0],dtype=tf.int32)
  roi_fm_regions = tf.image.crop_and_resize(feature_map,roi_fm_regions,box_ind=box_inds,crop_size=[params.crop_height,params.crop_width])
  return roi_fm_regions

def _clip_rois(rois,im_height,im_width):
  
  rois[:,0] = np.maximum(rois[:,0],0)
  rois[:,1] = np.maximum(rois[:,1],0)
  rois[:,2] = np.minimum(rois[:,2],im_width)
  rois[:,3] = np.minimum(rois[:,3],im_height)
  
  return rois

def _get_roi_regions(rois,fm_height,fm_width,im_height,im_width):

  fm_x1 = (rois[:,0]/im_width)
  fm_y1 = (rois[:,1]/im_height)
  fm_x2 = (rois[:,2]/im_width)
  fm_y2 = (rois[:,3]/im_height)
  
  return np.stack((fm_x1,fm_y1,fm_x2,fm_y2),axis=1)