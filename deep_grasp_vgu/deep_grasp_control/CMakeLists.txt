cmake_minimum_required(VERSION 2.8.3)
project(deep_grasp_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

find_package(catkin REQUIRED COMPONENTS rospy roslaunch)

# catkin_install_python(
#   PROGRAMS scripts/open_loop_control.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)