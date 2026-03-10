import torch
import torch.nn.functional as F


def sobel(video_grayscale):
    sobel_horizontal = torch.as_tensor(
                            [
                                [ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]
                            ], device=video_grayscale.device, dtype=torch.float32)

    sobel_vertical = torch.as_tensor(
                            [
                                [1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]
                            ], device=video_grayscale.device, dtype=torch.float32)


    # Make it 4d because F.conv2d strictly refuses to work with anything less than 4d. Yes, this is dumb.
    sobel_horizontal = sobel_horizontal.view(1, 1, 3, 3) 
    sobel_vertical   = sobel_vertical.view(1, 1, 3, 3)

    # Make the video 4d by squeezing a fake dimension in it
    video_4d      = video_grayscale.unsqueeze(1)
    
    horizontal_gradients = F.conv2d(video_4d, sobel_horizontal, padding=1)
    vertical_gradients   = F.conv2d(video_4d, sobel_vertical, padding=1)
    
    gradient = torch.sqrt((horizontal_gradients**2) + (vertical_gradients**2))
    gradient = gradient.squeeze(1)
    
    return gradient