# create data dir if not exists
mkdir data
rosbag record -a -o data/test-$1.bag