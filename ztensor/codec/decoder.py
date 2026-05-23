import torch
import typing
import zstandard

import numpy as np
import torch.nn.functional as F

from ztensor.effects import chroma, quantization


def decode_video(compressed_bytes: bytes, device: torch.device) -> torch.Tensor:
    planes, i_frame_indices, pixel_format = decompress_video(compressed_bytes, device)

    scene_boundaries = i_frame_indices.tolist()
    scene_boundaries.append(len(planes[0])) # Append the idx of the last frame to the list to use it as the scene_end variable for the last iteration of the loop below

    planes_decoded = []
    for plane in planes:
        for i in range(len(scene_boundaries)-1):
            scene_start = scene_boundaries[i]
            scene_end   = scene_boundaries[i+1]

            plane[scene_start : scene_end] = torch.cumsum(plane[scene_start : scene_end], dim=0)

        planes_decoded.append(plane)
    
    if pixel_format in ['I422', 'I420']: # This means the video is chroma subsampled, so we need to interpolate the U and V channels to be the same dimension as the Y channel
        y_tensor   = planes_decoded[0].unsqueeze(1).float()
        target_res = (y_tensor.shape[2], y_tensor.shape[3])

        u_tensor   = planes_decoded[1].unsqueeze(1).float()
        u_upscaled = F.interpolate(u_tensor, size=(target_res), mode='bilinear', align_corners=False)

        v_tensor   = planes_decoded[2].unsqueeze(1).float()
        v_upscaled = F.interpolate(v_tensor, size=(target_res), mode='bilinear', align_corners=False)

        video = torch.cat([y_tensor, u_upscaled, v_upscaled], dim=1)
        video = video.permute(0, 2, 3, 1)

        video = chroma.yuv2bgr(video)


    elif pixel_format == 'RGB3': # This means the video is RGB, so we just concatenate the channels along the last axis to form the RGB video.
        planes_decoded[0] = planes_decoded[0].unsqueeze(-1)
        planes_decoded[1] = planes_decoded[1].unsqueeze(-1)
        planes_decoded[2] = planes_decoded[2].unsqueeze(-1)

        video = torch.cat([planes_decoded[0], planes_decoded[1], planes_decoded[2]], dim=-1)
    
    else:
        raise ValueError(f"Unsupported format: {pixel_format}")

    video = video.clip(0,255).to(torch.uint8)

    return video


def decompress_video(compressed_bytes: bytes, device: torch.device) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor, str]:
    decompressed_bytes = zstandard.decompress(compressed_bytes)
    current_byte = 0

    pixel_format  = decompressed_bytes[current_byte : current_byte+4].decode('ascii')
    current_byte += 4

    quantization_parameter = int.from_bytes(decompressed_bytes[current_byte : current_byte+1], signed=False)

    datatype_format_np    = np.int8      if quantization_parameter in [1] else np.uint8
    datatype_format_torch = torch.int8   if quantization_parameter in [1] else torch.uint8
    num_bytes_per_pixel   = 1

    current_byte         += 1

    num_planes    = int.from_bytes(decompressed_bytes[current_byte : current_byte+1], signed=False)
    current_byte += 1

    num_i_frames  = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
    current_byte += 4


    i_frame_indices = np.frombuffer(decompressed_bytes[current_byte : current_byte + (num_i_frames*4)], dtype=np.uint32).copy() # we create a copy because torch asks for a writeable copy of the bytearray, not a read-only memory view of it.
    i_frame_indices = torch.as_tensor(i_frame_indices, dtype=torch.uint32, device=device)
    current_byte   += num_i_frames*4

    plane_shapes = []
    for _ in range(num_planes):
        shape       = np.frombuffer(decompressed_bytes[current_byte : current_byte + 12], dtype=np.int32)
        current_byte += 12

        plane_shapes.append(shape)

    planes = []
    for shape in plane_shapes:
        plane_len     = np.prod(shape) * num_bytes_per_pixel # The number of bytes that the current plan has.
        plane         = np.frombuffer(decompressed_bytes[current_byte : current_byte + plane_len], dtype=datatype_format_np).copy()
        plane         = torch.as_tensor(plane, dtype=datatype_format_torch, device=device).reshape(tuple(shape))
        
        if quantization_parameter in [1]:
            plane   = quantization.dequantize(plane, quantization_parameter)

        current_byte += plane_len

        if quantization_parameter not in [1]:
            plane = plane.to(torch.uint8)

        planes.append(plane)

    return planes, i_frame_indices, pixel_format