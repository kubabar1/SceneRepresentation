import random
import tensorflow as tf
import torch


def sample_batch_deepmind(data_to_run, B, M=None, K=None, device=None):
    with tf.train.SingularMonitoredSession() as sess:
        d = sess.run(data_to_run)
        frames = torch.from_numpy(d.query.context.frames).float()
        cameras = torch.from_numpy(d.query.context.cameras).float()
        query_camera = torch.from_numpy(d.query.query_camera).float()
        target = torch.from_numpy(d.target).float()
        # context = {"frames": frames, "cameras": cameras}
        # query = {"context": context, "query_camera": query_camera}
        # data = {"query": query, "target": target}

    N = len(frames)
    K = len(frames[0]) if K is None else K
    M = random.randint(1, K) if M is None else M

    x_tensors = [torch.Tensor().to(device) for _ in range(M)]
    v_tensors = [torch.Tensor().to(device) for _ in range(M)]
    x_q_tensor = torch.Tensor().to(device)
    v_q_tensor = torch.Tensor().to(device)

    for b in random.sample(range(N), N):
        i = 0
        for r in random.sample(range(K), M):
            x_tensors[i] = torch.cat([x_tensors[i], torch.unsqueeze(frames[b][r].permute(2, 0, 1), 0).to(device)])
            v_tensors[i] = torch.cat([v_tensors[i], cameras[b][r].view(1, 7, 1, 1).to(device)])
            i += 1
        x_q_tensor = torch.cat([x_q_tensor, torch.unsqueeze(target[b].permute(2, 0, 1), 0).to(device)])
        v_q_tensor = torch.cat([v_q_tensor, query_camera[b].view(1, 7, 1, 1).to(device)])

    return [x_tensors, v_tensors], [x_q_tensor, v_q_tensor]

# def show_batch(images):
#    fig = plt.figure()
#    # plt.subplots_adjust(hspace=0.5)
#    for i in range(len(images)):
#        ax1 = fig.add_subplot(3, 3, i + 1)
#        # ax1.set_xlabel(vocab[b][1][i], fontsize=8)
#        ax1.imshow(images[i])
#    plt.show()
#
#
# if __name__ == "__main__":
#    root_path = "datasets"
#    data_reader = DataReader(dataset='rooms_ring_camera', context_size=9, root=root_path)
#    data = data_reader.read(batch_size=12)
#    with tf.train.SingularMonitoredSession() as sess:
#        d = sess.run(data)
#        print("########################################")
#        frames = torch.from_numpy(d.query.context.frames).float()
#        cameras = torch.from_numpy(d.query.context.cameras).float()
#        query_camera = torch.from_numpy(d.query.query_camera).float()
#        target = torch.from_numpy(d.target).float()
#        context = {"frames": frames, "cameras": cameras}
#        query = {"context": context, "query_camera": query_camera}
#        data_res = {"query": query, "target": target}
#        # print(data_res)
#        print(target.size())
#        for b in range(frames.size()[0]):
#            show_batch(data_res["query"]["context"]["frames"][b])
#
#        for b in range(target.size()[0]):
#            fig = plt.figure()
#            ax1 = fig.add_subplot(1, 1, 1)
#            ax1.imshow(target[b])
#            plt.show()
#
