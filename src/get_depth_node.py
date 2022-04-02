#!/usr/bin/env python3
import sys
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
from imantics import Mask
import rospy
from sensor_msgs.msg import CameraInfo, Image
from rospy.numpy_msg import numpy_msg
import rospkg
from dynamic_reconfigure import client
import time

from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.encoder import resnet_encoder
from models.network.rsu_decoder import RSUDecoder
from models.network.depth_decoder import DepthDecoder

from PIL import Image as ImagePIL

use_gpu = True


SUBSCRIBING_TOPIC_NAME = 'camera/rgb/image_raw'
SUBSCRIBING_TOPIC_INFO = 'camera/rgb/camera_info'

#SUBSCRIBING_TOPIC_NAME = '/image_publisher_1648910127171650689/image_raw'
#SUBSCRIBING_TOPIC_INFO = '/image_publisher_1648910127171650689/camera_info'

PUBLISHING_TOPIC_NAME = 'pred_depth'


class Depth_Prediction_Service:

    def __init__(self, camera_info):
        # comment this line what is running on cpu/gpu
        # tf.debugging.set_log_device_placement(True)
        self.camera_info = camera_info

        rospack = rospkg.RosPack()
        base_path = rospack.get_path("depth_prediction")

        self.model_depth = {}
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

        self.model_depth["encoder"] = resnet_encoder(num_layers=18, num_inputs=1,
                                                pretrained=True).to(self.device)

        if True:
            self.model_depth["depth_decoder"] = RSUDecoder(num_output_channels=1, use_encoder_disp=True,
                                                      encoder_layer_channels=self.model_depth["encoder"].layer_channels).to(self.device)
        else:
            self.model_depth["depth_decoder"] = DepthDecoder(num_output_channels=1,
                                                        encoder_layer_channels=self.model_depth["encoder"].layer_channels).to(self.device)

        self.checkpoint = torch.load(base_path + '/src/models/model18_192x640.pth.tar',map_location=torch.device(self.device))

        for m_name, _ in self.model_depth.items():
            if m_name in self.checkpoint:
                self.model_depth[m_name].load_state_dict(self.checkpoint[m_name])
            else:
                print("There is no weight in checkpoint for model {}".format(m_name))
        for m in self.model_depth.values():
            m.eval()

        print('model_loaded')

        self.publisher = rospy.Publisher(PUBLISHING_TOPIC_NAME, Image, queue_size=1)




    def predict(self, image):

        image = image[np.newaxis]
        image_tensor = torch.tensor(image.copy())
        image_tensor = image_tensor.float()/255.

        embedding = self.model_depth["encoder"](image_tensor)
        depth_image = self.model_depth["depth_decoder"](embedding)

        return depth_image




    def depth_predict(self, image):

        # see: https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
        img_array = np.frombuffer(image.data, dtype=np.uint8).reshape(-1, image.width,image.height, order='F')
        img_array = np.transpose(img_array, (2, 1, 0 ))
        print('number of channels: ',img_array.shape[2])
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :-1]
        img_array = img_array[:, :, ::-1]
        img_array = np.transpose(img_array, (2, 0, 1 ))

        output = self.predict(img_array)

        np_arr = output[0].cpu().detach().numpy()
        np_arr = 255 * np_arr * 6 # Multiplied by 6 for better visualization. But yields to bitoverflow
        np_arr = np_arr.astype(np.uint8)


        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = 192
        msg.width = 640
        msg.encoding = "mono8"
        msg.is_bigendian = False
        msg.data = np_arr.tobytes()
        self.publisher.publish(msg)

        print(np_arr.shape)
        ##############################



def depth_prediction():
    rospy.init_node("Depth_Prediction", anonymous=True)
    # fetch a single message from camera_info
    rospy.loginfo("Waiting on camera info")
    camera_info = rospy.wait_for_message(SUBSCRIBING_TOPIC_INFO, CameraInfo)
    rospy.loginfo("done")
    rospy.loginfo(camera_info)

    rospy.loginfo('Start prediction Depth from RGB Image')
    depth_node = Depth_Prediction_Service(camera_info=camera_info)
    rospy.Subscriber(SUBSCRIBING_TOPIC_NAME, numpy_msg(Image), depth_node.depth_predict,queue_size=1)

    rospy.spin()


if __name__ == '__main__':
    try:
        print("Node started")
        depth_prediction()
    except rospy.ROSInterruptException:
        pass
