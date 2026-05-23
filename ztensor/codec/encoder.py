import torch
import zstandard

import numpy as np

from typing import List

from ztensor.codec import block_matching, padding, serialization
from ztensor.effects import quantization


def encode_video(planes: List[torch.Tensor], i_frame_indices: torch.Tensor, compression_factor: int, num_threads: int, pixel_format: str, quantization_parameter: int) -> bytes:

    serialized_video = bytes()

    block_width   = 8
    search_window = 8

    num_planes = len(planes)
    num_frames = len(planes[0]) # All planes have the same number of frames, so just take this info from the first plane
    
    header  = serialization.serialize_header(pixel_format, 
                                             quantization_parameter, 
                                             i_frame_indices, 
                                             block_width, 
                                             num_frames,
                                             num_planes)
    serialized_video += header

    for plane_tensor in planes:
        # Cast to int16 to calculate P-frames. Since the P-frames are only integer values, int16 will do just fine.
        plane_tensor  = plane_tensor.to(torch.int16)

        _, original_plane_h, original_plane_w = plane_tensor.shape

        plane_tensor  = padding.pad_plane(plane_tensor, block_width)

        motion_vectors, block_residuals = block_matching.block_matching(plane_tensor, block_width, search_window)

        payload = serialization.serialize_payload(motion_vectors, 
                                                  block_residuals, 
                                                  plane_tensor, 
                                                  i_frame_indices, 
                                                  original_plane_h, 
                                                  original_plane_w
                                                  )

        serialized_video += payload

        if quantization_parameter :
            # TODO figure out how to quantize with block-matching.
            plane_tensor = quantization.quantize(plane_tensor, quantization_parameter).to(torch.int8)
        else:
            plane_tensor = plane_tensor.to(torch.uint8)


    compressed_video = compress_video(serialized_video, compression_factor, num_threads)

    return compressed_video



def compress_video(video_bytes: bytes, compression_factor: int, num_threads:int) -> bytes:
    # The compression step has to run on CPU.
    compressor             = zstandard.ZstdCompressor(level=compression_factor, threads=num_threads)
    video_bytes_compressed = compressor.compress(video_bytes)

    return video_bytes_compressed