import torch
import torch.nn.functional as F

from ztensor.effects import chroma
from ztensor.codec import serialization, block_matching


def decode_video(compressed_bytes: bytes, device: torch.device) -> torch.Tensor:
    planes, i_frame_indices, pixel_format, matrix_coefficients = serialization.deserialize_payload(compressed_bytes, device)

    planes_decoded = block_matching.deconstruct_block_matching(planes, i_frame_indices, device)

    if pixel_format in ['I422', 'I420']: # This means the video is chroma subsampled, so we need to interpolate the U and V channels to be the same dimension as the Y channel
        y_tensor   = planes_decoded[0].unsqueeze(1).float()
        target_res = (y_tensor.shape[2], y_tensor.shape[3])

        u_tensor   = planes_decoded[1].unsqueeze(1).float()
        u_upscaled = F.interpolate(u_tensor, size=(target_res), mode='bicubic', align_corners=False, antialias=True)

        v_tensor   = planes_decoded[2].unsqueeze(1).float()
        v_upscaled = F.interpolate(v_tensor, size=(target_res), mode='bicubic', align_corners=False, antialias=True)

        video = torch.cat([y_tensor, u_upscaled, v_upscaled], dim=1)
        video = video.permute(0, 2, 3, 1)

        video = chroma.ycbcr2bgr(video, matrix_coefficients)
        video = video.round().clip(0, 255) # Round and clip U and V channels to integer values


    elif pixel_format == 'RGB3': # This means the video is RGB, so we just concatenate the channels along the last axis to form the RGB video.
        planes_decoded[0] = planes_decoded[0].unsqueeze(-1)
        planes_decoded[1] = planes_decoded[1].unsqueeze(-1)
        planes_decoded[2] = planes_decoded[2].unsqueeze(-1)

        video = torch.cat([planes_decoded[0], planes_decoded[1], planes_decoded[2]], dim=-1)
        video = video.clip(0, 255)
    
    else:
        raise ValueError(f"Unsupported format: {pixel_format}")

    video = video.to(torch.uint8)

    return video
