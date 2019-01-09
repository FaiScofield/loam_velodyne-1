# !/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import rospy
import rosbag

from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import CameraInfo, Imu, PointField, NavSatFix, PointCloud2
import sensor_msgs.point_cloud2 as pcl2

import numpy as np




def array2point(array_msg):
    
    velo_topic = '/velodyne_point'

    # vel_point = PointCloud2()
    # vel_point.header = Header()
    # vel_point.header.frame_id = 'velo_link'
    # vel_point.header.stamp = rospy.Time.now()
    # vel_point.height = 1
    # vel_point.width = array_msg.layout.dim[0].size
    # vel_point.is_dense = False
    # vel_point.is_bigendian = False
    # vel_point.fields = [
    #     PointField('x', 0, PointField.FLOAT32, 1),
    #     PointField('y', 4, PointField.FLOAT32, 1),
    #     PointField('z', 8, PointField.FLOAT32, 1),
    #     PointField('intensity', 12, PointField.FLOAT32, 1),
    #     PointField('ring', 16, PointField.FLOAT32, 1)]
    # vel_point.data = array_msg.data

    # bar = progressbar.ProgressBar()

    # read binary data
    data = array_msg.data

    # create header
    header = Header()
    header.frame_id = 'velo_link'
    header.stamp = rospy.Time.now()

    # fill pcl msg
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
                PointField('ring', 16, PointField.FLOAT32, 1)]
    pcl_msg = pcl2.create_cloud(header, fields, data)


    rate = rospy.Rate(100)
    rospy.Publisher("velodyne_point", pcl_msg, queue_size=100)


def main():
    rospy.init_node("array_to_point", anonymous=True)
    
    try:
        rospy.Subscriber("velodyne_packet", Float64MultiArray, array2point)
        rospy.spin()
    
    except Exception as e:
        print 'Error occurred.'
        print e.message

    finally:
        print("## Done ##")

   
if __name__ == '__main__':
    main()
