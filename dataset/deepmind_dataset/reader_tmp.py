import tensorflow as tf
import torch
import matplotlib.pyplot as plt

from data_reader import DataReader


def show_batch(images):
    fig = plt.figure()
    # plt.subplots_adjust(hspace=0.5)
    for i in range(len(images)):
        ax1 = fig.add_subplot(3, 3, i + 1)
        # ax1.set_xlabel(vocab[b][1][i], fontsize=8)
        ax1.imshow(images[i])
    plt.show()


if __name__ == "__main__":
    root_path = "datasets"
    data_reader = DataReader(dataset='rooms_ring_camera', context_size=9, root=root_path)
    data = data_reader.read(batch_size=12)
    with tf.train.SingularMonitoredSession() as sess:
        d = sess.run(data)
        print("########################################")
        frames = torch.from_numpy(d.query.context.frames).float()
        cameras = torch.from_numpy(d.query.context.cameras).float()
        query_camera = torch.from_numpy(d.query.query_camera).float()
        target = torch.from_numpy(d.target).float()
        context = {"frames": frames, "cameras": cameras}
        query = {"context": context, "query_camera": query_camera}
        data_res = {"query": query, "target": target}
        # print(data_res)
        print(target.size())
        for b in range(frames.size()[0]):
            show_batch(data_res["query"]["context"]["frames"][b])

        for b in range(target.size()[0]):
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.imshow(target[b])
            plt.show()
