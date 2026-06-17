import torch

from ztensor.utils import video
from ztensor.codec import encoder, decoder, i_frames
from ztensor.effects import chroma, histogram, blur, edge_detect

from typing import Tuple


def encode_pipeline(input_path: str, 
                    device: torch.Device, 
                    memory_budget: int, 
                    compression_factor: int, 
                    num_threads: int, 
                    chroma_subsample: str, 
                    quantization_parameter: bool,
                    block_size: int,
                    search_window: int
                    ) -> Tuple[torch.Tensor, bytes]:

    video_bgr         = video.read_video(input_path).to(device)
    video_grayscale   = video.bgr_to_grayscale(video_bgr)

    video_histogram   = histogram.video_histogram(video_grayscale, memory_budget)

    video_edges       = edge_detect.sobel(video_grayscale)

    troi_slices       = histogram.temporal_region_of_interest(video_histogram)
    i_frame_indices   = i_frames.select_i_frames(video_edges, troi_slices)

    if chroma_subsample == 'quarter':
        pixel_format        = 'I420'

        # Use bt709 for high-res video and bt601 for lower-res videos
        _, h, w, _   = video_bgr.shape
        if w >= 1280 and h >= 720:
            matrix_coefficients = "bt709"
        else:
            matrix_coefficients = "bt601"
        
        video_yuv         = chroma.bgr2ycbcr(video_bgr, matrix_coefficients)

        subsampled_video  = chroma.subsample_chroma_420(video_yuv)
        y, u, v           = subsampled_video
        
        # Round YUV values to avoid error acumullation when they get converted to ints
        y = y.round().clip(0, 255)
        u = u.round().clip(0, 255)
        v = v.round().clip(0, 255)

        planes       = [ y, u, v ]
    
    elif chroma_subsample == 'half-width':
        pixel_format = 'I422'

        # Use bt709 for high-res video and bt601 for lower-res videos
        _, h, w, _   = video_bgr.shape
        if w >= 1280 and h >= 720:
            matrix_coefficients = "bt709"
        else:
            matrix_coefficients = "bt601"

        video_yuv         = chroma.bgr2ycbcr(video_bgr, matrix_coefficients)

        subsampled_video  = chroma.subsample_chroma_422(video_yuv)
        y, u, v           = subsampled_video


        # Round YUV values to avoid error acumullation when they get converted to ints
        y = y.round().clip(0, 255)
        u = u.round().clip(0, 255)
        v = v.round().clip(0, 255)

        planes       = [ y, u, v ]

    else:
        pixel_format        = 'RGB3'
        matrix_coefficients = 'null'
        planes              = [ video_bgr[..., 0], video_bgr[..., 1], video_bgr[..., 2] ]


    compressed_planes = encoder.encode_video(planes, 
                                             i_frame_indices, 
                                             compression_factor, 
                                             num_threads, 
                                             pixel_format,
                                             matrix_coefficients, 
                                             quantization_parameter,
                                             block_size,
                                             search_window
                                             )

    return video_bgr, compressed_planes


def decode_pipeline(bytes_data: bytes, device: torch.device) -> torch.Tensor:

    decoded_video = decoder.decode_video(bytes_data, device)

    return decoded_video

