import torch
import zstandard

import numpy as np


def decode(compressed_bytes):
    video, i_frames = decompress_video(compressed_bytes)
    video, i_frames = video.copy(), i_frames.copy()

    video = torch.as_tensor(video, dtype=torch.int16)

    scene_boundaries = i_frames.tolist()
    scene_boundaries.append(len(video))

    for i in range(len(scene_boundaries)-1):
        scene_start = scene_boundaries[i]
        scene_end   = scene_boundaries[i+1]

        video[scene_start : scene_end] = torch.cumsum(video[scene_start : scene_end], dim=0)

    return video

def decompress_video(compressed_bytes):
    decompressed_bytes = zstandard.decompress(compressed_bytes)

    shape_bytes = decompressed_bytes[0:16]
    shape       = np.frombuffer(shape_bytes, dtype=np.int32)
    
    num_i_frames_bytes = decompressed_bytes[16:20]
    num_i_frames       = np.frombuffer(num_i_frames_bytes, dtype=np.int32)[0]

    i_frames_bytes = decompressed_bytes[20 : 20 + (num_i_frames*4)]
    i_frames       = np.frombuffer(i_frames_bytes, dtype=np.int32)

    video_bytes = decompressed_bytes[20 + (num_i_frames*4) : ]
    video       = np.frombuffer(video_bytes, dtype=np.int16)

    video = video.reshape(shape)
    

    return video, i_frames