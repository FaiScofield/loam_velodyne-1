# !/usr/bin/python
# -*- coding:utf-8 -*-

#
# Convert the velodyne_hits binary files to a rosbag
#
# To call:
#
#   python vel_to_rosbag.py velodyne_hits.bin vel.bag
#

import sys
import struct
import numpy as np

import rospy
import rosbag
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2
# import pcl


def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def verify_magic(s):
    magic = 44444   
    m = struct.unpack('<HHHH', s)   # pack(),unpack(),用于C语言数据与Python数据类型间转换

    return len(m)>=3 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic


def main(args):
    # if len(sys.argv) < 2:
    #     print 'Please specify velodyne hits file and output rosbag file'
    #     return 1

    # if len(sys.argv) < 3:
    #     print 'Please specify output rosbag file'
    #     return 1

    # f_bin = open(sys.argv[1], "r")

    # bag = rosbag.Bag(sys.argv[2], 'w')

    f_bin = open("/home/vance/dataset/PeRL/2013-01-10/velodyne_hits.bin", "r")
    bag = rosbag.Bag("/home/vance/aa.bag", 'w')
    print 'coverting...'
    print 'this work would take a while, please wait...'

    seq = 0
    try:
        while True:
            magic = f_bin.read(8)
            if magic == '': # eof
                break

            if not verify_magic(magic):
                print "Could not verify magic"

            num_hits = struct.unpack('<I', f_bin.read(4))[0]
            utime = struct.unpack('<Q', f_bin.read(8))[0]
            print "num_hits & utime:"
            print num_hits, utime

            f_bin.read(4) # padding

            # Read all hits
            # data = [[] for i in range(300)]
            data = []
            
            for j in range(num_hits):
                x = struct.unpack('<H', f_bin.read(2))[0]   # 每次读2个字节
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]    # 每次读1个字节
                r = struct.unpack('B', f_bin.read(1))[0]

                x, y, z = convert(x, y, z)
                # data.append([np.float(x), np.float(y), np.float(z), np.float(i), np.float(r)])
                data += [x, y, z, float(i), int(r)]
                # data += [np.uint8(x), np.uint8(y), np.uint8(z), np.uint8(i), np.uint8(r)]
                # print x,y,z,i,r
            
            # Now write out to rosbag
            timestamp = rospy.Time.from_sec(utime/1e6)

            # layout = MultiArrayLayout()
            # layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            # layout.dim[0].label = "hits"
            # layout.dim[0].size = num_hits
            # layout.dim[0].stride = 5
            # layout.dim[1].label = "xyzil"
            # layout.dim[1].size = 5
            # layout.dim[1].stride = 1

            # vel = Float64MultiArray()
            # vel.data = data
            # vel.layout = layout


            # vel = PointCloud2()
            # vel.header = Header()
            # vel.header.frame_id = 'velo_link'
            # vel.header.stamp = timestamp
            # vel.header.seq = seq
            # vel.height = 1
            # vel.width = len(data)/5
            # vel.is_dense = False
            # vel.is_bigendian = False
            # vel.fields = [
            #     PointField('x', 0, PointField.FLOAT32, 1),
            #     PointField('y', 4, PointField.FLOAT32, 1),
            #     PointField('z', 8, PointField.FLOAT32, 1),
            #     PointField('intensity', 12, PointField.FLOAT32, 1),
            #     PointField('ring', 16, PointField.FLOAT32, 1)]
            # vel.data = data
            # bag.write('velodyne_point', vel, t=timestamp)
            # bag.write('velodyne_point', vel)


            # fill pcl msg
            seq = seq + 1
            header = Header()
            header.frame_id = 'velo_link'
            header.stamp = timestamp
            header.seq = seq
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
                PointField('ring', 16, PointField.FLOAT32, 1)]
                
            pcl_msg = sensor_msgs.point_cloud2.create_cloud(header, fields, data)
            
            bag.write('velodyne_point', pcl_msg, timestamp)
            # bag.write('velodyne_point', pcl_msg)

    except Exception as e:
        print 'End of File'
        print e.message
    finally:
        f_bin.close()
        bag.close()
        print 'done.'

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
