import torch
import zstandard

import numpy as np

from typing import List

def encode_video(planes: List[torch.Tensor], i_frame_indices: torch.Tensor, compression_factor: int, num_threads: int, pixel_format: str) -> bytes:

    header  = pixel_format.encode('ascii')
    header += len(planes).to_bytes(4)                                  # int32 the number of planes in the video. This is necessary for decoding the video
    header += len(i_frame_indices).to_bytes(4)                         # int32 the number of i-frames in the video
    header += i_frame_indices.cpu().numpy().astype(np.int32).tobytes() # int32 indices of the i-frames

    payload = bytes()

    for plane_tensor in planes:
        # Cast to int16 to calculate P-frames. Since the P-frames are only integer values, int16 will do just fine.
        plane_tensor = plane_tensor.to(torch.int16)

        # Calculate the P-frames
        p_frames = torch.diff(plane_tensor, dim=0)
        # Because torch.diff reduces the dimension in 1, we squeeze back a full-zero frame back into the P-frames vector so it keeps the same
        # shape as the video
        p_frames = torch.cat([torch.zeros_like(plane_tensor[0]).unsqueeze(0), p_frames])
    
        # tells me which frames are P-frames
        p_frame_mask                  = torch.full(size=(plane_tensor.shape[0],), fill_value=True)
        p_frame_mask[i_frame_indices] = False
        
        # Substitutes the frames in the p_frame_mask for the P-frames calculated with diff
        plane_tensor[p_frame_mask] = p_frames[p_frame_mask]

        plane_shape    = np.array(plane_tensor.shape, dtype=np.int32).tobytes()
        plane_bytes    = plane_tensor.cpu().numpy().tobytes()
        
        header        += plane_shape # int32 shape for the current plane
        payload       += plane_bytes # the plane data in bytes
        
    compressed_video = compress_video(header + payload, compression_factor, num_threads)

    return compressed_video



def compress_video(video_bytes: bytes, compression_factor: int, num_threads:int) -> bytes:
    # The compression step has to run on GPU.
    compressor             = zstandard.ZstdCompressor(level=compression_factor, threads=num_threads)
    video_bytes_compressed = compressor.compress(video_bytes)

    return video_bytes_compressed