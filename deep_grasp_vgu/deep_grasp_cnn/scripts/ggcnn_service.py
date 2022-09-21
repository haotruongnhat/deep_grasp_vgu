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
from dougsm_helpers.timeit import TimeIt

from dougsm_helpers.gridshow import gridshow

from ggcnn.ggcnn_torch import predict, process_depth_image, load_model_by_ros
from deep_grasp_msgs.srv import GraspPrediction, GraspPredictionResponse

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose

from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates, LinkStates

import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list

import ros_numpy
from ros_numpy import numpify, msgify

TimeIt.print_output = False

class GGCNNService:
    def __init__(self):
        # Get the camera parameters
        self.waiting = False
        self.received = False
        self.enable_loop = False

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

        rospy.Service(namespace + 'predict', GraspPrediction, self.compute_service_handler)
        rospy.Service(namespace + 'predict_loop', Trigger, self.trigger_predict_loop)

        self.grasp_pred_pub = rospy.Publisher(namespace + 'grasp_pred', PoseStamped, queue_size=1)
        self.grasp_gt_pub = rospy.Publisher(namespace + 'grasp_gt', PoseStamped, queue_size=1)

        self.predict_map_pub = rospy.Publisher(namespace + 'predict_map', Image, queue_size=1)

        rospy.Subscriber('/gazebo/link_states', LinkStates, self._link_callback, queue_size=1)

        ### Init MoveIt Interface
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_name = "arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

        ## Move to home position
        self.home_pose = self.group.get_current_pose().pose
        self.home_pose.position.x = -0.111
        self.home_pose.position.y = 0.297
        self.home_pose.position.z = 0.40

        (plan, fraction) = self.group.compute_cartesian_path(
                                   [self.home_pose],   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0)         # jump_threshold

        self.group.execute(plan, wait=True)

        load_model_by_ros()
        print("Done initialization")

    def _link_callback(self, msg):
        target_object = "cube2::link"
        object_index = msg.name.index(target_object)
        object_pose_to_world = msg.pose[object_index]

        obj_to_world_matrix = numpify(object_pose_to_world)

        target_robot = "robot::base_link"
        robot_index = msg.name.index(target_robot)
        robot_pose_to_world = msg.pose[robot_index]

        robot_to_world_matrix = numpify(robot_pose_to_world)

        ## Checked: OK
        object_to_robot_matrix = np.dot(np.linalg.inv(robot_to_world_matrix), obj_to_world_matrix)

        gt_pose = PoseStamped()
        gt_pose.header.frame_id = self.base_frame

        gt_pose.pose = msgify(Pose, object_to_robot_matrix)

        self.grasp_gt_pub.publish(gt_pose)

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

        if not self.enable_loop:
            with TimeIt('Total'):
                ret = self.grasp_predict()
                return ret

    def trigger_predict_loop(self, req):
        print("Enable predict loop")

        self.enable_loop = not self.enable_loop
        self.waiting = True

        res = TriggerResponse()
        res.success = True
        res.message = "Current loop status: {}".format(self.enable_loop)

        return res

    def grasp_predict(self):
        print("Running grasp predict")
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
    rospy.init_node('ggcnn_service')
    GGCNN = GGCNNService()
    rospy.spin()
