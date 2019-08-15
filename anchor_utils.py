import numpy as np
import sys

def centers_to_corners(a_boxes):
    """
    inputs:

    'a_boxes': [N,4] array where a_boxes[i,:] is a
    [x_c,y_c,w,h] format bounding box

    returns:

    [N,4] array where arr[i,:] is now in [x1,y1,x2,y2]
    format
    """
    x1 = a_boxes[:,0] - a_boxes[:,2]/2
    y1 = a_boxes[:,1] - a_boxes[:,3]/2
    x2 = a_boxes[:,0] + a_boxes[:,2]/2
    y2 = a_boxes[:,1] + a_boxes[:,3]/2

    return np.stack((x1,y1,x2,y2),axis=1)

def _xy_to_corners(gt_boxes):
  """
  inputs: 

  'gt_boxes': a [K,4] array in format [x,y,w,h] where
  (x,y) is the top left corner of the box

  returns:

  [K,4] array in format [x1,y1,x2,y2]
  
  """
  x1 = gt_boxes[:,0]
  y1 = gt_boxes[:,1]
  x2 = gt_boxes[:,0] + gt_boxes[:,2]
  y2 = gt_boxes[:,1] + gt_boxes[:,3]
  
  return np.stack((x1,y1,x2,y2),axis=1)

def adjust_anchors(boxes,offsets):
    """
    inputs: 

    'boxes': [N,4] array where boxes[i,:] is a 
    [x1,y1,x2,y2] format bounding box
    'offsets': [N,4] array where offsets[i,:] is
    a [tx,ty,tw,th] format bounding box adjustment

    returns:

    'boxes': modified [N,4] array where boxes[i,:] has
    been adjusted by 'offsets[i,:]
    """

    anch_w = boxes[:,2] - boxes[:,0]
    anch_h = boxes[:,3] - boxes[:,1]

    boxes[:,0] = boxes[:,0] + offsets[:,0]*anch_w
    boxes[:,1] = boxes[:,1] + offsets[:,1]*anch_h

    new_ws = np.exp(offsets[:,2])*anch_w
    new_hs = np.exp(offsets[:,3])*anch_h

    boxes[:,2] = boxes[:,0] + new_ws
    boxes[:,3] = boxes[:,1] + new_hs

    return boxes

def clip_rois(rois,im_height,im_width):
    """
    inputs:

    'rois': an [N,4] array in format [x1,y1,x2,y2]
    'im_height': height of image
    'im_width': width of image

    returns:

    'rois': [N,4] array in format [x1,y1,x2,y2] 
    where the coordinates have been clipped to the 
    boundaries of the image
    """
    
    rois[:,0] = np.maximum(rois[:,0],0)
    rois[:,1] = np.maximum(rois[:,1],0)
    rois[:,2] = np.minimum(rois[:,2],im_width-1)
    rois[:,3] = np.minimum(rois[:,3],im_height-1)
  
    return rois

def image_to_fm_rois(rois,fm_height,fm_width,im_height,im_width):
    """
    inputs:

    'rois': an [N,4] array in format [x1,y1,x2,y2] of boxes
    on the original image
    'fm_height': height of feature map
    'fm_width': width of feature map
    'im_height': height of image
    'im_width': width of image

    returns:

    returns an [N,4] array in format [x1,y1,x2,y2] of boxes in the 
    feature map that correspond to the input boxes
    """

    fm_x1 = (rois[:,0]/im_width)*fm_width
    fm_y1 = (rois[:,1]/im_height)*fm_height
    fm_x2 = (rois[:,2]/im_width)*fm_width
    fm_y2 = (rois[:,3]/im_height)*fm_height
  
    return np.stack((fm_x1,fm_y1,fm_x2,fm_y2),axis=1)

def top_k_inds(arr,k):
    """
    inputs: 
    
    'arr': 1-D array of scores
    'k': number of indices to return

    returns:

    'inds': 1-D array of indices, when used to index 'arr'
    gives the 'k' largest values of 'arr'

    """

    inds = np.argsort(-arr)
    inds = inds[:k]
    return (inds).astype(np.int32)

def clean_anchors(boxes):
    """
    inputs:

    'boxes': [N,4] array where boxes[i,:] is in the [x1,y1,x2,y2]
    format. 
    'im_height': height of input image
    'im_width': width of input image

    returns:

    'inds': indices of valid anchors. valid anchors do not have
    negative width or height

    """
    b1 = boxes[:,2] > boxes[:,0]
    b2 = boxes[:,3] > boxes[:,1]
    b = np.logical_and(b1,b2)
    inds = np.arange(len(boxes),dtype=np.int32)[b]
    return (inds).astype(np.int32)

def filter_anchors(boxes,im_height,im_width):
    """
    inputs:

    'boxes': [N,4] array where boxes[i,:] is in the [x1,y1,x2,y2]
    format.
    'im_height': height of input image
    'im_width': width of input image

    returns:

    'inds': indices of valid anchors. valid anchors do not cross
    the image boundary.
    """

    b1 = boxes[:,0] >= 0
    b2 = boxes[:,1] >= 0 
    b3 = boxes[:,2] <= im_width - 1
    b4 = boxes[:,3] <= im_height - 1

    b = np.logical_and.reduce((b1,b2,b3,b4))
    inds = np.arange(len(boxes),dtype=np.int32)[b]
    return (inds).astype(np.int32)
    
def get_anchor_labels(anchor_boxes,gt_boxes,pos_thresh,neg_thresh_lo,neg_thresh_hi):
    """
    inputs:

    'anchor_boxes': an [N,4] array in [x1,y1,x2,y2] format
    'gt_boxes': an [N,5] array in [x,y,w,h,class] format
    'pos_thresh': if an anchor box has an IoU with any gt box
    above 'pos_thresh' we label it a foreground anchor
    'neg_thresh_lo': lower bound on IoU for background classification
    'neg_thresh_hi': upper bound on IoU for background classification

    We label an anchor box as background if its max IoU with all gt boxes
    is in the interval ['neg_thresh_lo','neg_thresh_hi').

    returns:

    'anchor_labels': a 1-D array of length N. Elements with value 1 denote
    foreground anchors, elements with value -1 denote background anchors,
    elements with value 0 denote unlabeled anchors.
    'bbox_adjs': an [N,4] array in [tx,ty,tw,th] format. bbox_adjs[i,:] is 
    the bounding box offset to the ground truth box with maximal IoU to anchor i.
    'cls_labels': a 1-D array of length N. Each element denotes the 
    class label of the respective anchor. Anchors with a foreground labeling 
    are assigned the class label of the gt box with maximal IoU. Anchors
    with a background labeling are assigned a class label of 0. 

    """
    # label the anchors object or not object
    anchor_labels = np.zeros(np.shape(anchor_boxes)[0],dtype = np.float32)
    bbox_adjs = np.zeros(np.shape(anchor_boxes),dtype = np.float32)

    cls_gt_labels = gt_boxes[:,4]
    gt_boxes = _xy_to_corners(gt_boxes[:,:5])
    # get the ious and label according to params
    ious = _compute_ious(anchor_boxes,gt_boxes)
    max_ious = np.max(ious,axis=1)

    max_gt_boxes = np.argmax(ious,axis=1)
    pos_anchors = np.where(max_ious >= pos_thresh)[0]
    neg_anchors = np.where(np.logical_and(max_ious >= neg_thresh_lo, max_ious < neg_thresh_hi))[0]

    anchor_labels[pos_anchors] = 1
    anchor_labels[neg_anchors] = -1
    anchor_labels[np.argmax(max_ious)] = 1
    
    a_width = anchor_boxes[:,2] - anchor_boxes[:,0]
    a_height = anchor_boxes[:,3] - anchor_boxes[:,1]
    a_ctr_x = anchor_boxes[:,0] + (a_width)/2
    a_ctr_y = anchor_boxes[:,1] + (a_height)/2

    gt_width = gt_boxes[max_gt_boxes,2]
    gt_height = gt_boxes[max_gt_boxes,3]
    gt_ctr_x = gt_boxes[max_gt_boxes,0] + gt_width/2
    gt_ctr_y = gt_boxes[max_gt_boxes,1] + gt_height/2

    # compute the bbox adjs between the anchors and the gt_boxes
    tx = (gt_ctr_x - a_ctr_x)/a_width
    ty = (gt_ctr_y - a_ctr_y)/a_height
    tw = np.log(gt_width/a_width)
    th = np.log(gt_height/a_height)

    bbox_adjs = np.stack((tx,ty,tw,th),axis=1)
    
    # label the anchors with coco category classes
    cls_labels = np.zeros(np.shape(anchor_boxes)[0],dtype=np.int32)
    cls_labels[pos_anchors] = cls_gt_labels[max_gt_boxes[pos_anchors]] + 1
    
    return anchor_labels.astype(np.int32),bbox_adjs,cls_labels.astype(np.int32)

def _compute_ious(a_boxes,gt_boxes):
    """
    inputs:
    
    'a_boxes': an [N,4] array in format [x1,y1,x2,y2]
    'gt_boxes': an [K,4] array in format [x1,y1,x2,y2] 

    returns:

    'ious': an [N,K] array where ious[i,j] is the IoU of the a_boxes[i,:] 
    and gt_boxes[j,:]
    """


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

def sample_anchors_for_training(anchor_labels,batch_size,prop_pos):
        """
        inputs:
        
        'anchor_labels': a 1-D array of length N. Elements
        with value 1 denote foreground anchors. Elements with value
        -1 denote background anchors. Elements with value 0 denote 
        unlabeled anchors.

        'prop_pos': a value in [0,1] which denotes the desired 
        percentage of positive anchors in the batch.

        returns:

        'inds': a 1-D array of length k <= N which contains
        the indices of the selected anchors for training.

        Anchors are selected by trying to sample the specified 
        number of positive anchors from N. If there are less
        positive anchors than specified negative anchors are
        used to fill the batch.

        """
        if batch_size >= len(anchor_labels):
            #include all anchors - should be rare
            print('BATCH SIZE > ANCHOR LABELS')
            print('RETURNING RANGE OBJECT')
            return np.arange(len(anchor_labels)).astype(np.int32)

        num_pos_anchors = np.round(batch_size*prop_pos).astype(np.int32)

        pos_anchors = np.where(anchor_labels == 1)[0]
        neg_anchors = np.where(anchor_labels == -1)[0]
        neut_anchors = np.array([])

        if len(pos_anchors) > num_pos_anchors:
            np.random.seed(0)
            pos_anchors = np.random.choice(pos_anchors,num_pos_anchors,replace=False)

        neg_to_sample = batch_size - len(pos_anchors)
        if len(neg_anchors) > neg_to_sample:
            np.random.seed(0)
            neg_anchors = np.random.choice(neg_anchors,neg_to_sample,replace=False)
        
        neut_to_sample = batch_size - (len(neg_anchors) + len(pos_anchors))
        if neut_to_sample > 0:
            #include neutral anchors as negative examples
            neut_anchors = np.where(anchor_labels == 0)[0]
            num_neut_anchors = batch_size - (len(pos_anchors) + len(neg_anchors))
            np.random.seed(0)
            neut_anchors = np.random.choice(neut_anchors,num_neut_anchors,replace=False)
        
        return np.concatenate((pos_anchors,neg_anchors,neut_anchors)).astype(np.int32)
       