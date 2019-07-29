#Faster-R-CNN

This repo contains a tensorflow implementation of Faster R-CNN written to run on Google Colab. In this implementation we use Inception V3 as the pre-trained deep network. 

Due to computational constraints we make several modifications to the orginial paper. We do not fine tune the base model. We only train the RPN and Fast R-CNN layers. We do this by first training the RPN layer, then using the ROI proposals from the RPN network to train a Fast R-CNN detection layer. 

We use the MS COCO dataset for training. 

 
