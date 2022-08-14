#!/usr/bin/env python

from __future__ import division, print_function

import rospy

import time

import numpy as np
import cv2

from tf import transformations as tft
from deep_grasp_msgs.srv import GraspPrediction, GraspPredictionResponse
from sensor_msgs.msg import Image, CameraInfo

class ControlClass:
    def __init__(self):
        ggcnn_service_name = '/ggcnn_service'
        rospy.wait_for_service(ggcnn_service_name + '/predict')
        self.ggcnn_srv = rospy.ServiceProxy(ggcnn_service_name + '/predict', GraspPrediction)

    def move_to_capture_pose(self):
        pass

    def grasp_pose_predict(self):
        pass

    def grasp_pose_execution(self):
        pass

if __name__ == '__main__':
    rospy.init_node('deep_grasp_control')
    controller = ControlClass()
    rospy.spin()
