#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>

#include <iostream>
//#include <math.h>
#include <iomanip>  // std::setprecision
#include <fstream>

void odomCallback(const nav_msgs::OdometryConstPtr& odom)
{
  geometry_msgs::Point p = odom->pose.pose.position;
 // geometry_msgs::Quaternion q = odom->pose.pose.orientation;

  std::ofstream f;
  f.open("laser_trajectory.txt", std::ios_base::app);
  f << std::fixed;
  f << std::setprecision(6) << p.x << " " << p.y << " " << p.z << std::endl;

  f.close();
}



int main(int argc, char **argv)
{
  ros::init(argc, argv, "saveOdometryKITTI");
  ros::NodeHandle nh;

  ROS_INFO("Saving camera trajectory to laser_trajectory.txt");
  std::ofstream f;
  f.open("~/vance_ws/laser_trajectory.txt");
  f << std::fixed;
  f.close();

  ros::Subscriber sub = nh.subscribe("/integrated_to_init", 1000, odomCallback);

  while (ros::ok()) {
    ros::spinOnce();
  }
//  ros::spin();
  ROS_INFO("Trajectory saved.");

  return 0;
}
