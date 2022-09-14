#!/usr/bin/env python3

from __future__ import division, print_function

import rospy

import time

import numpy as np
import cv2

from tf import transformations as tft
import tf

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.timeit import TimeIt

from dougsm_helpers.gridshow import gridshow

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose

from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates

import moveit_commander
import moveit_msgs.msg

# import cv_bridge
# bridge = cv_bridge.CvBridge()
import ros_numpy
from ros_numpy import numpify, msgify

TimeIt.print_output = False

class GGCNNService:
    def __init__(self):
        # Get the camera parameters
        namespace = "/ggcnn_service/"
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
        rospy.Subscriber(rospy.get_param(namespace + 'camera/color_topic'), Image, self._color_img_callback, queue_size=1)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_callback, queue_size=1)
        rospy.Service(namespace + 'save_info', Trigger, self._save_info)

        self.waiting = False
        self.received = False

    def _color_img_callback(self, msg):
        # Doing a rospy.wait_for_message is super slow, compared to just subscribing and keeping the newest one.

        self.curr_img_time = time.time()
        self.curr_color_img = ros_numpy.numpify(msg)

        self.received = True

    def _depth_img_callback(self, msg):
        # Doing a rospy.wait_for_message is super slow, compared to just subscribing and keeping the newest one.

        self.curr_img_time = time.time()
        self.last_image_pose = tfh.current_robot_pose(self.base_frame, self.camera_frame)
        self.curr_depth_img = ros_numpy.numpify(msg)

        self.received = True

    def _model_callback(self, msg):
        target_object = "cube2"
        object_index = msg.name.index(target_object)
        object_pose_to_world = msg.pose[object_index]
        
        obj_to_world_matrix = numpify(object_pose_to_world)

        target_robot = "robot"
        robot_index = msg.name.index(target_robot)
        robot_pose_to_world = msg.pose[robot_index]

        robot_to_world_matrix = numpify(robot_pose_to_world)

        ## Save this matrix for testing
        self.object_to_robot_matrix = np.dot(np.linalg.inv(robot_to_world_matrix), obj_to_world_matrix)
        object_pose_to_robot =  msgify(Pose, object_to_robot_matrix)

        camera_pose_to_robot = tfh.current_robot_pose(self.base_frame, self.camera_frame)
        camera_to_robot_matrix = numpify(camera_pose_to_robot)

        self.object_to_camera_matrix = np.dot(np.linalg.inv(camera_to_robot_matrix), object_to_robot_matrix)
        object_pose_to_camera =  msgify(Pose, object_to_camera_matrix)

    def _save_info(self, req):
        root = "/root/ros_ws/src/deep_grasp_vgu/deep_grasp_cnn/self_data/"
        suff = str(int(time.time())) + "_"

        np.save(root + suff +"cam_K.npy", self.cam_K)

        depth = self.curr_depth_img.copy()
        np.save(root + suff +"depth.npy", depth)

        color = self.curr_color_img.copy()
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(root + suff +"color.png", color)

        camera_pose = self.last_image_pose
        cam_p = camera_pose.position
        np.save(root + suff +"cam_p.npy", np.array([cam_p.x, cam_p.y, cam_p.z]))

        camera_rot = tft.quaternion_matrix(tfh.quaternion_to_list(camera_pose.orientation))[0:3, 0:3]
        np.save(root + suff +"camera_rot.npy", camera_rot)

        np.save(root + suff +"object_to_robot_matrix.npy", self.object_to_robot_matrix)
        np.save(root + suff +"object_to_camera_matrix.npy", self.object_to_camera_matrix)

        res = TriggerResponse()
        res.success = True
        res.message = "Saved"

        return res


if __name__ == '__main__':
    rospy.init_node('ggcnn_service')
    GGCNN = GGCNNService()
    rospy.spin()
