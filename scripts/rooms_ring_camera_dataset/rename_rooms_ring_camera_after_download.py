import os

rooms_ring_camera = '../../dataset/deepmind_dataset/datasets/rooms_ring_camera'


# ===============================================================================================
# 10% dataset-u
# ===============================================================================================
# gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/test/0[0-1][0-9]-of-240.tfrecord .
# gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/test/02[0-4]-of-240.tfrecord .
# ===============================================================================================
# gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/0[0-1][0-9][0-9]-of-2160.tfrecord .
# gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/020[0-9]-of-2160.tfrecord .
# gsutil -m cp -r gs://gqn-dataset/rooms_ring_camera/train/021[0-6]-of-2160.tfrecord .
# ===============================================================================================

def rename_rooms_ring_camera_after_download():
    for filename in os.listdir(rooms_ring_camera + "/test"):
        new_name = filename[1:7] + str(24) + filename[10:]
        print(new_name)
        # os.rename(os.path.join(rooms_ring_camera + "/test", filename), new_name)
    print("#####################################")
    for filename in os.listdir(rooms_ring_camera + "/train"):
        new_name = filename[1:8] + str(216) + filename[12:]
        print(new_name)
        # os.rename(os.path.join(rooms_ring_camera + "/train", filename), new_name)


if __name__ == "__main__":
    rename_rooms_ring_camera_after_download()
