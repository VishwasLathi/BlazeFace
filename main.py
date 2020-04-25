import tensorflow as tf
import numpy as np 
import cv2
# import pickle
# import glob 
import os 
import time 
import argparse

from network import network 
from loss import smooth_l1_loss
from utils import dataloader

class BlazeFace():
    
    def __init__(self, config):
        
        self.input_shape = config.input_shape
        self.feature_extractor = network(self.input_shape)
        
        #number of anchor boxes corresponding to the  16*16 and 8*8 feature map respectively.
        self.n_boxes = [1,1] 
        
        self.model = self.build_model()
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            
        self.checkpoint_path = config.checkpoint_path
        
    def build_model(self):
        
        model = self.feature_extractor
        #model.output[0], model.output[1] corresponds to the 16*16 and the 8*8 feature map respectively. 

        #conv layer for predicting the confidence scores for anchors corresponding to the 16*16 feature map.
        bb_16_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[0], 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[0])
        bb_16_conf_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 1))(bb_16_conf)
        
        
        #conv layer for predicting the confidence scores for anchors corresponding to the 8*8 feature map.
        bb_8_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[1], 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[1])
        bb_8_conf_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 1))(bb_8_conf)

        # Concatenate confidence predictions for the above feature maps. 
        conf_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])
        

        #conv layer for predicting the [x_center,y_center,width,height] for anchors corresponding to the 16*16 feature map.
        bb_16_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 4,
                                            kernel_size=3, 
                                            padding='same')(model.output[0])
        bb_16_loc_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 4))(bb_16_loc)
        
        
        #conv layer for predicting the [x_center,y_center,width,height] for anchors corresponding to the 8*8 feature map.
        bb_8_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 4,
                                          kernel_size=3,
                                          padding='same')(model.output[1])
        bb_8_loc_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 4))(bb_8_loc)
        
        # Concatenate the location precdictions. 
        loc_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])
        
        # concatenate the confidence scores as the first channel along with the location predicitions.
        output_combined = tf.keras.layers.Concatenate(axis=-1)([conf_of_bb, loc_of_bb])
        
        #printing the output tensors.
        #conf_of_bb : [batch_size, 320, 1]

        #output_combined : [batch_size,320,5] where 320 corresponds to the total number of feature points (16*16 + 8*8)
        #last dimension (5) corresponds to score and the 4 predicted location values for each anchor box. 
        print(conf_of_bb, output_combined)

        #return a model with image as input to give confidence and confidence_and_location as outputs. 
        return tf.keras.models.Model(model.input, [conf_of_bb, output_combined])

    def train(self):
        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model = self.model
        #apply binary cross entropy for the loss of confidence scores and custom smooth l1 loss for regressing the anchor coordinates.
        model.compile(loss=['binary_crossentropy', smooth_l1_loss], optimizer=opt)
        
        X,Y = dataloader()
        for epoch in range(self.nb_epoch):
            res = model.fit(x=X,y=Y, batch_size=self.batch_size,initial_epoch=epoch,epochs=epoch+1)
            if epoch % 100 == 0:
                model.save_weights(os.path.join(config.checkpoint_path, str(epoch)))

if __name__ == "__main__":

    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=(128,128,3))
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./")
    args.add_argument('--dataset_dir', type=str, default="./")
    args.add_argument('--label_path', type=str, default="./")

    config = args.parse_args()

    blazeface = BlazeFace(config)
    if config.train:
        blazeface.train()


