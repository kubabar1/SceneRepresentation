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
