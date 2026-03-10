import torch
import torch.nn.functional as F


def blur_video(video_grayscale):
    # Define a 3x3 box blur kernel
    blur_kernel   = torch.ones((3, 3), device=video_grayscale.device, dtype=torch.float32) / 9.0
    # Make it 4d because F.conv2d strictly refuses to work with anything less than 4d. Yes, this is dumb.
    blur_kernel   = blur_kernel.view(1, 1, 3, 3) 

    # Make the video 4d by squeezing a fake dimension in it
    video_4d      = video_grayscale.unsqueeze(1)
    
    blurred_4d    = F.conv2d(video_4d, blur_kernel, padding=1)
    blurred_video = blurred_4d.squeeze(1)

    return blurred_video