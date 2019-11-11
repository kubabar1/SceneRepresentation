# script downloading 10% of rooms_ring_camera deepmind dataset

mkdir ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/test
mkdir ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/train

gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/test/0[0-1][0-9]-of-240.tfrecord ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/test
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/test/02[0-4]-of-240.tfrecord ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/test

gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/0[0-1][0-9][0-9]-of-2160.tfrecord ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/train
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/020[0-9]-of-2160.tfrecord ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/train
gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/021[0-6]-of-2160.tfrecord ../../dataset/deepmind_dataset/datasets/rooms_ring_camera/train
