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

from deep_grasp_msgs.srv import GraspPrediction, GraspPredictionResponse

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from visualization_msgs.msg import MarkerArray, Marker

from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.msg import ModelStates, LinkStates, ContactsState

import moveit_commander
import moveit_msgs.msg
from moveit_commander.conversions import pose_to_list

import ros_numpy
from ros_numpy import numpify, msgify

from ur_control.arm import Arm
from ur_control import transformations, traj_utils, conversions

from loguru import logger
logger.remove()
logger.add(sys.stderr, format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>', level="INFO")

TimeIt.print_output = False


class GGCNNService:
    def __init__(self):
        # Get the camera parameters
        self.waiting = False
        self.received = False
        self.enable_loop = False

        self.br = tf.TransformBroadcaster()

        self.base_frame = "base_link"
        self.eef_frame = "ur_gripper_tip_link"

        self.grasp_predict_srv = rospy.ServiceProxy('/ggcnn_node/predict', GraspPrediction)

        namespace = "/ggcnn_grasp_control/"
        rospy.Service(namespace + 'grasp', Trigger, self.trigger_grasp)
        rospy.Service(namespace + 'grasp_loop', Trigger, self.trigger_grasp_loop)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self._link_callback, queue_size=1)
        self.grasp_gt_pub = rospy.Publisher(namespace + 'grasp_gt', PoseStamped, queue_size=1)
        self.object_meshes_pub = rospy.Publisher(namespace + 'object_meshes', MarkerArray, queue_size=1)

        self.arm = Arm(robot_urdf='ur3',
            ft_sensor=True,  # get Force/Torque data or not
            gripper=True,  # Enable gripper
            base_link= 'base_link',
            ee_link = 'ur_gripper_tip_link'
        )

        ### Init MoveIt Interface
        self.home_pose = self.arm.end_effector()

        # pose = copy.deepcopy(self.home_pose)
        # pose[0] = -0.123
        # pose[1] = 0.382
        # pose[2] = 0.05

        # pose[0] = 0.0
        # pose[1] = 0.34
        # pose[2] = 0.03

        # self.arm.gripper.command(0.05)

        # self.arm.set_target_pose(pose=pose, wait=True, t=1.0)
        # self.arm.gripper.command(0.02)

        # pose[0] = -0.123
        # pose[1] = 0.382
        # pose[2] = 0.1

        # pose[0] = 0.0
        # pose[1] = 0.34
        # pose[2] = 0.1
        # self.arm.set_target_pose(pose=pose, wait=True, t=1.0)

        # self.arm.gripper.release_current()

        logger.info("Done initialization")

    def _link_callback(self, msg):
        target_object = "bar_clamp"
        object_index = msg.name.index(target_object)
        object_pose_to_world = msg.pose[object_index]

        obj_to_world_matrix = numpify(object_pose_to_world)

        target_robot = "robot"
        robot_index = msg.name.index(target_robot)
        robot_pose_to_world = msg.pose[robot_index]

        robot_to_world_matrix = numpify(robot_pose_to_world)

        ## Checked: OK
        object_to_robot_matrix = np.dot(np.linalg.inv(robot_to_world_matrix), obj_to_world_matrix)

        gt_pose = PoseStamped()
        gt_pose.header.frame_id = self.base_frame

        gt_pose.pose = msgify(Pose, object_to_robot_matrix)

        array_msg = MarkerArray()

        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.ns = "objects"
        marker.type = Marker.MESH_RESOURCE
        marker.id = 0
        marker.mesh_resource = "file:///root/ros_ws/src/deep_grasp_vgu/datasets/object_meshes/bar_clamp.dae"
        marker.pose = msgify(Pose, object_to_robot_matrix)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.mesh_use_embedded_materials = True

        array_msg.markers.append(marker)

        self.grasp_gt_pub.publish(gt_pose)
        self.object_meshes_pub.publish(array_msg)

    def trigger_grasp(self, req):
        logger.info("Grasp trigger")

        res = TriggerResponse()
        pred_res = self.grasp_predict_srv()

        logger.info("Grasp prediction: {}".format(pred_res))

        if not pred_res.success:
            res.success = False
            res.message = "Failed to find grasp pose" 
            return res

        grasp_candidate = pred_res.best_grasp

        current_pose = self.arm.end_effector(rot_type = 'euler')

        grasp_pose = conversions.from_pose_to_list(grasp_candidate.pose)
        grasp_angles = list(tft.euler_from_quaternion(grasp_pose[3:]))
        grasp_yaw = grasp_angles[2]

        alternative_yaw = angles.normalize_angle(grasp_yaw + np.pi)

        current_eef_yaw = current_pose[-1]

        if (current_eef_yaw - grasp_yaw) > (current_eef_yaw - alternative_yaw):
            grasp_angles[2] = alternative_yaw
            alternative_quaternion = list(tft.quaternion_from_euler(grasp_angles[0], grasp_angles[1], grasp_angles[2]))
            grasp_pose[3:] = alternative_quaternion

            logger.info("Found more approriate grasp pose")

        logger.info("Initial grasp width: {}".format(grasp_candidate.width/2.0*0.5))
        self.arm.gripper.command(grasp_candidate.width/2.0*1.1)

        self.arm.set_target_pose(pose=grasp_pose, wait=True, t=1.0)

        # self.arm.gripper.command(grasp_candidate.width/2.0)
        self.arm.gripper.close()

        self.arm.gripper.grab(link_name="bar_clamp::bar_clamp")

        self.arm.set_target_pose(pose=self.home_pose, wait=True, t=1.0)
        self.arm.gripper.open()
        self.arm.gripper.release(link_name="bar_clamp::bar_clamp")

        res = TriggerResponse()
        res.success = True
        res.message = "Current grasp status"

        return res
        
    def trigger_grasp_loop(self, req):
        logger.info("Enable predict loop")

        self.enable_loop = not self.enable_loop
        self.waiting = True

        res = TriggerResponse()
        res.success = True
        res.message = "Current loop status: {}".format(self.enable_loop)

        return res

if __name__ == '__main__':
    rospy.init_node('ggcnn_grasp_control')
    GGCNN = GGCNNService()
    rospy.spin()
