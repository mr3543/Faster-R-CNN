#Faster R-CNN With TensorFlow and Google Colab

This repo contains a tensorflow implementation of [Faster R-Cnn](https://arxiv.org/abs/1506.01497). We use vgg16 and the MS COCO database. This implementation borrows from the original implementation [here](https://github.com/rbgirshick/py-faster-rcnn) and another good tensorflow implementation [here](https://github.com/endernewton/tf-faster-rcnn).

Several modifications are made to the original [paper](https://arxiv.org/abs/1506.01497).

1) We forgo the alternate training method detailed in the original paper in favor of the approximate joint training method. 

2) We use an SGD mini-batch size of 1 instead of 2 for both the RPN and Fast RCNN layers.

3) We use a slightly modified RoI pooling operation discussed below. 

4) For simplicity, only vgg16 and MS COCO are supported. Some simplifications to the training process are also made. We do not apply image flipping, and due to computational constraints our training time is much less.

#RoI Pooling:

RoI Pooling is an operation that takes a variable size crop of the vgg16 feature map and max pools the map down to a new 7x7 feature map. For example if the crop is 14x14 we would pool over 2x2 blocks to reduce the size to 7x7.

Our roi pooling implementation symmetrically pads the feature map crop with zeros to ensure that both the length and width are divisible by 7. If the crop is 19x11 we would pad with zeros to get a 21x14 map. 

We then use the variable sized max pooling operation ```max_pool_v2``` in ```tensorflow.python_ops.gen.nn.ops``` to reduce the padded feature map to 7x7. See [here](https://github.com/tensorflow/tensorflow/pull/11875) for more details. 

To run, open the file 'faster_rcnn.ipynb' in Google colab and follow the instructions. 


