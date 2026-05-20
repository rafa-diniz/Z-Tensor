import torch
import typing

def bgr2yuv(bgr_video: torch.Tensor) -> torch.Tensor:
    '''
    https://www.computerlanguage.com/results.php?definition=YUV%2FRGB+conversion+formulas

    From RGB to YUV

    Y = 0.299R + 0.587G + 0.114B
    U = 0.492 (B-Y)
    V = 0.877 (R-Y)
    '''

    bgr_weights = torch.Tensor([0.114, 0.587, 0.299]).to(bgr_video.device)
    b = bgr_video[:, :, :, 0].unsqueeze(-1)
    r = bgr_video[:, :, :, 2].unsqueeze(-1)

    y = torch.sum(bgr_video * bgr_weights, dim=-1).unsqueeze(-1)
    u = 128 + (0.492 * (b - y)) 
    v = 128 + (0.877 * (r - y))
    
    yuv_video = torch.cat([y, u, v], dim=-1)

    return yuv_video


def yuv2bgr(yuv_video: torch.Tensor) -> torch.Tensor:
    '''
    https://www.computerlanguage.com/results.php?definition=YUV%2FRGB+conversion+formulas

    From YUV to RGB

    R = Y + (1.140 * V)
    G = Y - (0.395 * U) - (0.581 * V)
    B = Y + (2.032 * U)
    '''
    y = yuv_video[:,:,:, 0]
    u = yuv_video[:,:,:, 1] - 128
    v = yuv_video[:,:,:, 2] - 128

    r = y + (1.140 * v)
    g = y - (0.395 * u) - (0.581 * v)
    b = y + (2.032 * u)

    r = r.unsqueeze(-1)
    g = g.unsqueeze(-1)
    b = b.unsqueeze(-1)

    bgr_video = torch.cat([b, g, r], dim=-1)

    return bgr_video


def subsample_chroma_422(yuv_img: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies 422 chroma subsampling
    """
    y  = yuv_img[..., 0]
    uv = yuv_img[..., 1 : ]

    uv = uv.permute(0, -1, 1, 2) # Convert from (frames, h, w, channel) to (frames, channel, h, w)

    pool2d  = torch.nn.AvgPool2d(kernel_size=(1,2), stride=(1,2), padding=0)
    
    uv_subsampled = pool2d(uv)
    uv_subsampled = uv_subsampled.permute(0, 2, 3, 1) # Convert back to (frames, h, w, channel)

    return y, uv_subsampled[..., 0], uv_subsampled[..., 1]


def subsample_chroma_420(yuv_img: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies 420 chroma subsampling
    """    
    y  = yuv_img[..., 0]
    uv = yuv_img[..., 1 : ]

    uv = uv.permute(0, -1, 1, 2) # Convert from (frames, h, w, channel) to (frames, channel, h, w)

    pool2d  = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    uv_subsampled = pool2d(uv)
    uv_subsampled = uv_subsampled.permute(0, 2, 3, 1) # Convert back to (frames, h, w, channel)

    return y, uv_subsampled[..., 0], uv_subsampled[..., 1]