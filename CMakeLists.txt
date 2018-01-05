cmake_minimum_required(VERSION 2.8.3)
project(loam_velodyne_vance)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
#  velodyne_pointcloud
  )


#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(but_velodyne REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${ButVELODYNE_INCLUDE_DIRS})

LINK_DIRECTORIES(
  ${PCL_LIBRARY_DIRS}
  ${ButVELODYNE_LIBRARY_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

# Original (Velodyne VLP16)
add_executable(scanRegistration_vance src/scanRegistration.cpp)
target_link_libraries(scanRegistration_vance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserOdometry_vance src/laserOdometry.cpp)
target_link_libraries(laserOdometry_vance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserMapping_vance src/laserMapping.cpp)
target_link_libraries(laserMapping_vance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(transformMaintenance_vance src/transformMaintenance.cpp)
target_link_libraries(transformMaintenance_vance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# For Velodyne 64ES3
add_executable(scanRegistration-64E_vance src/scanRegistration_64E.cpp)
target_link_libraries(scanRegistration-64E_vance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

# For KITTI data processing
add_executable(scanRegistrationKITTI src/scanRegistrationKITTI.cpp)
target_link_libraries(scanRegistrationKITTI ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserOdometryKITTI src/laserOdometryKITTI.cpp)
target_link_libraries(laserOdometryKITTI ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(laserMappingKITTI src/laserMappingKITTI.cpp)
target_link_libraries(laserMappingKITTI ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})
add_executable(transformMaintenanceKITTI src/transformMaintenanceKITTI.cpp)
target_link_libraries(transformMaintenanceKITTI ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} )
add_executable(saveOdometryKITTI src/saveOdometryKITTI.cpp)
target_link_libraries(saveOdometryKITTI ${catkin_LIBRARIES})


