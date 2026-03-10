import torch
import torch.nn.functional as F


def video_histogram(video_grayscale, MEMBUDGET):
    """
    Since torch.histc does not run generate histograms using dimensions, this function uses a one-hot encoding trick to generate histograms
    for the frames in parallel on the GPU.

    Args:
        video_grayscale: The grayscale video
        MEMBUDGET: The maximum VRAM usage allowed
    """

    batch_size = int(MEMBUDGET // (video_grayscale.numel() * 8)) # video_grayscale.numel() * 8 because the video needs to be converted into int64 for F.one_hot(), so each "bin" is 8 bytes long.
    batch_size = max(1, batch_size)
    batched_video_grayscale = video_grayscale.split(batch_size)
    
    video_histogram = []

    for batch in batched_video_grayscale:
        batch_gray_int64   = batch.to(torch.int64)
        batch_histogram_3d = F.one_hot(batch_gray_int64, num_classes=256)
        batch_histogram_3d = batch_histogram_3d.sum(dim=(1, 2)).to(torch.float32)
        video_histogram.append(batch_histogram_3d)
    
    video_histogram = torch.cat(video_histogram)
    video_histogram = video_histogram

    return video_histogram


def temporal_region_of_interest(video_histogram):
    """
    Uses the histogram deltas to pinpoint motion-based temporal region of interest (TROIs) in the video.
    These TROIs suggest where to add I-Frames.

    Args:
        video_histogram: the video histogram (numFrames, 256)
    """

    video_histogram  = video_histogram.to(torch.float32)

    histogram_deltas = torch.diff(video_histogram, dim=0).abs().mean(dim=-1)

    # Threshold is 1.5 standard deviations to the right of the mean.
    threshold      = torch.mean(histogram_deltas, dim=-1) + (torch.std(histogram_deltas, dim=-1) * 1)
    
    # The left and right neighbors are used to filter for the local peaks. It fixes a subsequent selection of 
    left_neighbor  = F.pad(histogram_deltas[:-1], (1,0), value=float('inf'))
    right_neighbor = F.pad(histogram_deltas[1:],  (0,1), value=float('inf'))

    # To be in the TROI, the delta has to be higher than the threshold and be the peak in its neighborhood
    is_in_troi     =    (histogram_deltas > left_neighbor) & \
                        (histogram_deltas > right_neighbor) & \
                        (histogram_deltas > threshold)

    troi_slices   = is_in_troi.nonzero().squeeze(1) + 1 # +1 because torch.diff reduces the temporal dimension in 1. So the
    # i-th frame in the TROI slices is actually the i+1-th frame in the real video.

    # This logic transforms the indices into pairs of indices to delimitate the TROIs.
    troi_slices = troi_slices.repeat_interleave(2)
    # The first frame is always an I-frame, so we add its index here.
    troi_slices = torch.cat([torch.tensor([0], device=troi_slices.device), troi_slices])
    troi_slices = torch.cat([troi_slices, torch.tensor([video_histogram.shape[0]-1], device=troi_slices.device)])
    troi_slices = troi_slices.split(2)

    return troi_slices