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

    for b in range(B):
        i = random.randint(0, N - 1)
        for k in range(M):
            rand = random.randint(0, K - 1)
            x_tensors[k] = torch.cat([x_tensors[k], torch.unsqueeze(frames[i][rand].permute(2, 0, 1), 0).to(device)])
            v_tensors[k] = torch.cat([v_tensors[k], cameras[i][rand].view(1, 7, 1, 1).to(device)])
        x_q_tensor = torch.cat([x_q_tensor, torch.unsqueeze(target[i].permute(2, 0, 1), 0).to(device)])
        v_q_tensor = torch.cat([v_q_tensor, query_camera[i].view(1, 7, 1, 1).to(device)])

    return [x_tensors, v_tensors], [x_q_tensor, v_q_tensor]
