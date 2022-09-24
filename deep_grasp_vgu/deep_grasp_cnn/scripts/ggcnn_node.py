#!/usr/bin/env python3

from __future__ import division, print_function
import rospy
import sys

import time

import numpy as np
import cv2
import copy

from tf import transformations as tft
import tf

import dougsm_helpers.tf_helpers as tfh
import angles

from dougsm_helpers.timeit import TimeIt

from dougsm_helpers.gridshow import gridshow

from ggcnn.ggcnn_torch import predict, process_depth_image, load_model_by_ros
from deep_grasp_msgs.srv import GraspPrediction, GraspPredictionResponse

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker

from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates, LinkStates

import ros_numpy
from ros_numpy import numpify, msgify

from loguru import logger
logger.remove()
logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>', level="INFO")

TimeIt.print_output = False
MAX_GRIPPER = 0.8
MIN_GRIPPER = 0.0

MOVABLE_GRIPPER_VALUE = MAX_GRIPPER - MIN_GRIPPER

MAX_GRIPPER_WIDTH  = 0.016055490931148223
MIN_GRIPPER_WIDTH = 0.0998695579020194
MOVABLE_WIDTH = MIN_GRIPPER_WIDTH - MAX_GRIPPER_WIDTH

A = (MAX_GRIPPER-MIN_GRIPPER)/(MAX_GRIPPER_WIDTH - MIN_GRIPPER_WIDTH)
B = MIN_GRIPPER - A*MIN_GRIPPER_WIDTH

class GGCNNService:
    def __init__(self):
        # Get the camera parameters
        self.waiting = False
        self.received = False
        self.enable_loop = False
        self.done_loading_model = False

        namespace = "/ggcnn_node/"
        cam_info_topic = rospy.get_param(namespace + 'camera/info_topic')
        camera_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_K = np.array(camera_info_msg.K).reshape((3, 3))

        self.br = tf.TransformBroadcaster()

        self.base_frame = rospy.get_param(namespace + 'camera/robot_base_frame')
        self.camera_frame = rospy.get_param(namespace + 'camera/camera_frame')
        self.img_crop_size = rospy.get_param(namespace + 'camera/crop_size')
        self.img_crop_y_offset = rospy.get_param(namespace + 'camera/crop_y_offset')
        self.cam_fov = rospy.get_param(namespace + 'camera/fov')

        self.counter = 0
        self.curr_depth_img = None
        self.curr_img_time = 0
        self.last_image_pose = None

        rospy.Subscriber(rospy.get_param(namespace + 'camera/depth_topic'), Image, self._depth_img_callback, queue_size=1)

        rospy.Service(namespace + 'predict', GraspPrediction, self.compute_service_handler)
        rospy.Service(namespace + 'predict_loop', Trigger, self.trigger_predict_loop)

        self.grasp_pred_pub = rospy.Publisher(namespace + 'grasp_pred', PoseStamped, queue_size=1)
        self.predict_map_pub = rospy.Publisher(namespace + 'predict_map', Image, queue_size=1)

        self.done_loading_model = load_model_by_ros()
        logger.info("Done initialization")

    def _depth_img_callback(self, msg):
        # Doing a rospy.wait_for_message is super slow, compared to just subscribing and keeping the newest one.
        if not self.waiting:
          return
        self.curr_img_time = time.time()
        self.last_image_pose = tfh.current_robot_pose(self.base_frame, self.camera_frame)
        self.curr_depth_img = ros_numpy.numpify(msg)

        self.received = True

        if self.enable_loop:
            with TimeIt('Total'):
                ret = self.grasp_predict()

    def compute_service_handler(self, req):
        self.waiting = True
        while not self.received:
          rospy.sleep(0.01)
        self.waiting = False
        self.received = False

        if not self.done_loading_model:
            return

        if not self.enable_loop:
            with TimeIt('Total'):
                ret = self.grasp_predict()
                return ret

    def trigger_predict_loop(self, req):
        logger.info("Enable predict loop")

        self.enable_loop = not self.enable_loop
        self.waiting = True

        res = TriggerResponse()
        res.success = True
        res.message = "Current loop status: {}".format(self.enable_loop)

        return res

    def gripper_command(self, gripper_width, wait=True):
        """ In meters
        """
        if (gripper_width < 0.0) & (gripper_width > (MIN_GRIPPER_WIDTH - MAX_GRIPPER_WIDTH)):
            logger.warning("Gripper width out of bound")

        gripper_width += MAX_GRIPPER_WIDTH
        gripper_width = np.clip(gripper_width, MAX_GRIPPER_WIDTH, MIN_GRIPPER_WIDTH)

        gripper_value = np.clip(A*gripper_width + B, MIN_GRIPPER, MAX_GRIPPER)

        signed = np.array([1, -1, 1, 1, -1, 1])
        self.gripper_home_pose = list(signed*gripper_value)

        self.move_group_gripper.go(self.gripper_home_pose, wait=wait)

    def grasp_predict(self):
        logger.info("Running grasp predict")
        depth = self.curr_depth_img.copy()

        camera_pose = self.last_image_pose
        cam_p = camera_pose.position
        camera_rot = tft.quaternion_matrix(tfh.quaternion_to_list(camera_pose.orientation))[0:3, 0:3]

        # Do grasp prediction
        depth_crop, depth_nan_mask = process_depth_image(depth, self.img_crop_size, 300, return_mask=True, crop_y_offset=self.img_crop_y_offset)
        points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask, filters=(2.0, 2.0, 2.0))

        # Mask Points Here
        angle -= np.arcsin(camera_rot[0, 1])  # Correct for the rotation of the camera
        angle = (angle + np.pi/2) % np.pi - np.pi/2  # Wrap [-np.pi/2, np.pi/2]

        # Convert to 3D positions.
        imh, imw = depth.shape
        x = ((np.vstack((np.linspace((imw - self.img_crop_size) // 2, (imw - self.img_crop_size) // 2 + self.img_crop_size, depth_crop.shape[1], np.float), )*depth_crop.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * depth_crop).flatten()
        y = ((np.vstack((np.linspace((imh - self.img_crop_size) // 2 - self.img_crop_y_offset, (imh - self.img_crop_size) // 2 + self.img_crop_size - self.img_crop_y_offset, depth_crop.shape[0], np.float), )*depth_crop.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * depth_crop).flatten()
        pos = np.dot(camera_rot, np.stack((x, y, depth_crop.flatten()))).T + np.array([[cam_p.x, cam_p.y, cam_p.z]])

        width_m = width_img / 300.0 * 2.0 * depth_crop * np.tan(self.cam_fov * self.img_crop_size/depth.shape[0] / 2.0 / 180.0 * np.pi)

        best_g = np.argmax(points)
        best_g_unr = np.unravel_index(best_g, points.shape)

        ret = GraspPredictionResponse()
        ret.success = True
        g = ret.best_grasp
        g.pose.position.x = pos[best_g, 0]
        g.pose.position.y = pos[best_g, 1]
        g.pose.position.z = pos[best_g, 2]
        g.pose.orientation = tfh.list_to_quaternion(tft.quaternion_from_euler(np.pi, 0, ((angle[best_g_unr]%np.pi) - np.pi/2)))
        g.width = width_m[best_g_unr]
        g.quality = points[best_g_unr]

        ## Publish grasp pose
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = self.base_frame
        grasp_pose.pose = g.pose

        self.grasp_pred_pub.publish(grasp_pose)

        ## Publish predict map
        norm_image = cv2.normalize(points, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        color_heat_map = cv2.applyColorMap(norm_image,  cv2.COLORMAP_JET)
        self.predict_map_pub.publish(msgify(Image, color_heat_map, encoding='rgb8'))

        return ret

if __name__ == '__main__':
    rospy.init_node('ggcnn_node')
    GGCNN = GGCNNService()
    rospy.spin()
