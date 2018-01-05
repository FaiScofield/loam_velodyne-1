# loam_velodyne

This is a LOAM (Lidar Odometry and Mapping) ROS package for Velodyne VLP-16 3D laser scanner. This package is a simple modified copy of [loam_velodyne git repository](https://github.com/daobilige-su/loam_velodyne) from **daobilige-su**, who fixed a bug on laserOdometry.cpp to get rid of the matrix NaN error during L-M optimization step. Please cite Zhang et al.'s paper if this package is used. 

J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time. Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.([PDF](http://www.frc.ri.cmu.edu/~jizhang03/Publications/RSS_2014.pdf))([VIDEO](https://www.youtube.com/watch?feature=player_embedded&v=8ezyhTAEyHs))([SRC FILES](http://docs.ros.org/indigo/api/loam_velodyne/html/files.html))

Wiki Webpage by the Author: [http://wiki.ros.org/loam_velodyne](http://wiki.ros.org/loam_velodyne)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=8ezyhTAEyHs" target="_blank"><img src="http://img.youtube.com/vi/8ezyhTAEyHs/0.jpg" alt="LOAM back and forth" width="240" height="180" border="10" /></a>

# how to use
Check https://github.com/laboshinl/loam_velodyne 

# What I have done
- Add some comments based on others and what I took for.
- Make it suitable for KITTI dateset.(no good result by now, still ongoing)
