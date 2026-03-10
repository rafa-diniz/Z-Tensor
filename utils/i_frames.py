import torch


def select_i_frames(edges_video, troi_slices):
    frame_variances = torch.var(edges_video, dim=(1,2))
    i_frame_indices = set()
    i_frame_indices.add(0)

    for slice in troi_slices:
        start_idx, end_idx = slice[0], slice[1]+1
        sliced_variances   = frame_variances[start_idx : end_idx]

        best_local_idx = torch.argmax(sliced_variances)
        real_idx       = best_local_idx + start_idx

        i_frame_indices.add(real_idx.item())

    i_frame_indices = torch.as_tensor(sorted(list(i_frame_indices)), device=edges_video.device, dtype=torch.int64)

    return i_frame_indices