#!/bin/bash

while [ true ]; do
  /bin/sleep 0.5
  #sudo ifconfig eth0 192.168.3.043
  sudo ifconfig enp0s31f6 192.168.3.043
  echo "velodyne config.. "
done 


