cmake_minimum_required(VERSION 2.8.3)

project(loam_velodyne_64e)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
  velodyne_pointcloud
  mrpt_bridge
  mrpt_map
  mrpt_msgs
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MRPT REQUIRED base slam gui opengl maps obs nav)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${mrpt_INCLUDE_DIRS}
)

LINK_DIRECTORIES(
  ${PCL_LIBRARY_DIRS}
  ${OpenCV_LIBRARIES}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL OpenCV
  INCLUDE_DIRS include
)

# Original (Velodyne VLP16)
add_executable(scanRegistration_64e src/scanRegistration.cpp)
target_link_libraries(scanRegistration_64e ${catkin_LIBRARIES} ${PCL_LIBRARIES} )

add_executable(laserOdometry_64e src/laserOdometry.cpp)
target_link_libraries(laserOdometry_64e ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(laserMapping_64e src/laserMapping.cpp)
target_link_libraries(laserMapping_64e ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(transformMaintenance_64e src/transformMaintenance.cpp)
target_link_libraries(transformMaintenance_64e ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(saveMap src/saveMap.cpp)
target_link_libraries(saveMap ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${MRPT_LIBS})


# For PeRL dataset
#add_executable(array2point src/array2point.cpp)
#target_link_libraries(array2point ${catkin_LIBRARIES} ${PCL_LIBRARIES} )
