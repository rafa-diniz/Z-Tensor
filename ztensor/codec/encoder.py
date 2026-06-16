import torch
import zstandard

from typing import List

from ztensor.codec import block_matching, padding, serialization
from ztensor.effects import quantization


def encode_video(planes: List[torch.Tensor], 
                 i_frame_indices: torch.Tensor, 
                 compression_factor: int, 
                 num_threads: int, 
                 pixel_format: str, 
                 quantization_parameter: int,
                 block_size: int,
                 search_window: int
                 ) -> bytes:

    compressor       = zstandard.ZstdCompressor(level=compression_factor, threads=num_threads)
    serialized_video = []

    num_planes = len(planes)
    num_frames = len(planes[0]) # All planes have the same number of frames, so just take this info from the first plane
    header     = serialization.serialize_header(pixel_format, 
                                                quantization_parameter, 
                                                i_frame_indices, 
                                                block_size, 
                                                num_frames,
                                                num_planes)
    serialized_video.append(header)

    for plane_tensor in planes:
        plane_tensor  = plane_tensor.to(torch.int16)

        _, original_plane_h, original_plane_w = plane_tensor.shape
        plane_tensor                          = padding.pad_plane(plane_tensor, block_size)
        _, padded_plane_h, padded_plane_w     = plane_tensor.shape

        motion_vectors, block_residuals = block_matching.block_matching(plane_tensor, block_size, search_window, i_frame_indices)

        if quantization_parameter:
            block_residuals = quantization.quantize(block_residuals, quantization_parameter).to(torch.int8)
        else:
            block_residuals = block_residuals.to(torch.uint8)

        payload = serialization.serialize_payload(motion_vectors, 
                                                  block_residuals, 
                                                  plane_tensor, 
                                                  i_frame_indices, 
                                                  original_plane_h, 
                                                  original_plane_w,
                                                  padded_plane_h,
                                                  padded_plane_w
                                                  )

        serialized_video.append(payload)



    serialized_video = b"".join(serialized_video)
    compressed_video = compress_video(compressor, serialized_video)

    return compressed_video


def compress_video(compressor: zstandard.ZstdCompressor, video_bytes: bytes) -> bytes:
    # The compression step has to run on CPU.
    video_bytes_compressed = compressor.compress(video_bytes)

    return video_bytes_compressed