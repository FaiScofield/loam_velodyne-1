cmake_minimum_required(VERSION 2.8.3)
project(loam_velodyne)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${EIGEN3_INCLUDE_DIR} 
	${PCL_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

# Original (Velodyne VLP16)
add_executable(scanRegistration src/scanRegistration.cpp)
target_link_libraries(scanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserOdometry src/laserOdometry.cpp)
target_link_libraries(laserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserMapping src/laserMapping.cpp)
target_link_libraries(laserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(transformMaintenance src/transformMaintenance.cpp)
target_link_libraries(transformMaintenance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# For Velodyne 64ES3
add_executable(scanRegistration-64E src/scanRegistration_64E.cpp)
target_link_libraries(scanRegistration-64E ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})


# For KITTI data processing
# add_executable(scanRegistration_kitti src/scanRegistration_kitti.cpp)
# target_link_libraries(scanRegistration_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ButVELODYNE_LIBRARIES})
# add_executable(laserOdometry_kitti src/laserOdometry_kitti.cpp)
# target_link_libraries(laserOdometry_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ButVELODYNE_LIBRARIES})
# add_executable(laserMapping_kitti src/laserMapping_kitti.cpp)
# target_link_libraries(laserMapping_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ButVELODYNE_LIBRARIES})
# add_executable(transformMaintenance_kitti src/transformMaintenance_kitti.cpp)
# target_link_libraries(transformMaintenance_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ButVELODYNE_LIBRARIES})



