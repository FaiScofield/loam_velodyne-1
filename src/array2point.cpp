#include <cmath>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/Float64MultiArray.h>
#include <velodyne_pointcloud/point_types.h>

class A2P
{
public:
  A2P()
  {
    sub_ = nh_.subscribe("velodyne_packet", 1, &A2P::array2point, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2> ("velodyne_point", 1);
  }

  void array2point(const std_msgs::Float64MultiArray& farray)
  {
//    pcl::PointCloud<velodyne_pointcloud::PointXYZIR> vel_point;
    sensor_msgs::PointCloud2 vel_point;

    vel_point.height = 1;
    vel_point.width = farray.layout.dim[0].size;
//    ROS_INFO("sum of points:%d", vel_point.width);
    vel_point.is_dense = false;
    vel_point.is_bigendian = false;
    vel_point.data.resize(vel_point.height * vel_point.width * 5);
    for (size_t j=0; j<vel_point.width*5; j++){
////      vel_point.points[j].x =  farray.data[5*j+0];
////      vel_point.points[j].y =  farray.data[5*j+1];
////      vel_point.points[j].z =  farray.data[5*j+2];
////      vel_point.points[j].intensity =  farray.data[5*j+3];
////      vel_point.points[j].ring =  farray.data[5*j+4];
      vel_point.data[j] =  farray.data[j];
    }

//    vel_point.point_step = sizeof (vel_point.width*20);
//    vel_point.row_step   = static_cast<uint32_t> (sizeof (PointT) * msg.width);


//    pcl::toROSMsg(vel_point, output);
    vel_point.header.frame_id = "velo_link";
    vel_point.header.stamp = ros::Time::now();

    ros::Rate r(100);
    while (ros::ok()){
      pub_.publish(vel_point);
      ros::spinOnce();
      r.sleep();
    }

  }

private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;

};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "array_to_point");
  ROS_INFO("coverting the Array to velodyne points...");

  A2P a2p;

  ros::spin();

  return 0;


}

