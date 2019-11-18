from GQN.Properties import Properties
from GQN.Train import Train

if __name__ == "__main__":
    properties = Properties(
        data_path='dataset/local_dataset/blender_dataset',
        json_path='dataset/local_dataset/blender_dataset/observations.json',
        deepmind_dataset='rooms_ring_camera',
        deepmind_dataset_root_path='dataset/deepmind_dataset/datasets')
    train = Train(properties)
    train.continue_train("results/models/model_16000.pt")
