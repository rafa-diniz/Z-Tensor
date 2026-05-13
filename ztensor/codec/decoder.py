import torch
import typing
import zstandard

import numpy as np
import torch.nn.functional as F

from ztensor.effects import chroma


def decode_video(compressed_bytes: bytes, device: torch.device) -> torch.Tensor:
    video, i_frames = decompress_video(compressed_bytes, device)

    scene_boundaries = i_frames.tolist()
    scene_boundaries.append(len(video))

    for i in range(len(scene_boundaries)-1):
        scene_start = scene_boundaries[i]
        scene_end   = scene_boundaries[i+1]

        video[scene_start : scene_end] = torch.cumsum(video[scene_start : scene_end], dim=0)

    video = video.clip(0,255).to(torch.uint8)

    return video


def decompress_video(compressed_bytes: bytes, device: torch.device) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    decompressed_bytes = zstandard.decompress(compressed_bytes)
    current_byte = 0

    pixel_format  = decompressed_bytes[current_byte : current_byte+4].decode('ascii')
    current_byte += 4

    num_planes    = int.from_bytes(decompressed_bytes[current_byte : current_byte+4])
    current_byte += 4

    num_i_frames  = int.from_bytes(decompressed_bytes[current_byte : current_byte+4])
    current_byte += 4


    i_frames      = np.frombuffer(decompressed_bytes[current_byte : current_byte + (num_i_frames*4)], dtype=np.int32).copy() # we create a copy because torch asks for a writeable copy of the bytearray, not a read-only memory view of it.
    i_frames      = torch.as_tensor(i_frames, dtype=torch.int32, device=device)
    current_byte += num_i_frames*4

    plane_shapes = []
    for _ in range(num_planes):
        shape       = np.frombuffer(decompressed_bytes[current_byte : current_byte + 12], dtype=np.int32)
        current_byte += 12

        plane_shapes.append(shape)

    planes = []
    for shape in plane_shapes:
        plane_len     = np.prod(shape) * 2 # The number of bytes that the current plan has. Multiplied by 2 because the plane is an int16
        plane         = np.frombuffer(decompressed_bytes[current_byte : current_byte + plane_len], dtype=np.int16).copy()
        plane         = torch.as_tensor(plane, dtype=torch.int16, device=device).reshape(tuple(shape))
        
        current_byte += plane_len

        planes.append(plane)

    if pixel_format in ['I420', 'I422']: # This means the video is chroma subsampled, so we need to interpolate the U and V channels to be the same dimension as the Y channel
 
        y_tensor   = planes[0].unsqueeze(1).float()
        target_res = (y_tensor.shape[2], y_tensor.shape[3])

        u_tensor   = planes[1].unsqueeze(1).float()
        u_upscaled = F.interpolate(u_tensor, size=(target_res), mode='bilinear', align_corners=False).to(torch.int16)

        v_tensor   = planes[2].unsqueeze(1).float()
        v_upscaled = F.interpolate(v_tensor, size=(target_res), mode='bilinear', align_corners=False).to(torch.int16)

        y_tensor = y_tensor.to(torch.int16)

        video = torch.cat([y_tensor, u_upscaled, v_upscaled], dim=1)
        video = video.permute(0, 2, 3, 1)

        video = chroma.yuv2bgr(video)


    # This just means the video is full-resolution, so all 3 channels are the same resolution. So, we just have to stack them.
    elif pixel_format == 'RGB3':
        planes[0] = planes[0].unsqueeze(-1)
        planes[1] = planes[1].unsqueeze(-1)
        planes[2] = planes[2].unsqueeze(-1)

        video = torch.cat([planes[0], planes[1], planes[2]], dim=-1)

    else:
        raise ValueError(f"Unsupported format: {pixel_format}")

    return video, i_frames