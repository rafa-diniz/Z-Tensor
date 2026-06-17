import torch
import typing


def bgr2ycbcr(bgr_video: torch.Tensor, matrix_coefficients: str) -> torch.Tensor:
    """Convert BGR to YCbCr according to the BT.709-6 or the BT.601-7 standards.

    References: https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en and https://www.itu.int/rec/R-REC-BT.601-7-201103-I/en


    Args:
        bgr_video (torch.Tensor): The BGR video. Shape = (frames, height, width, channels)
        matrix_coefficients (str): The reference standard. Should be either \'BT.709\' or \'BT.601\'

    Returns:
        torch.Tensor: The YCbCr video. WARNING: Cb and Cr values are shifted to the right so they map to [0,255] instead of [-128, 127]!
        To get the original values back, simply subtract 128. 
        Shape = (frames, height, width, channels)
    """
    bgr_video = bgr_video.float()

    b_channel = bgr_video[:, :, :, 0].unsqueeze(-1)
    r_channel = bgr_video[:, :, :, 2].unsqueeze(-1)

    if matrix_coefficients == "bt709":
        luma_coefficients_bgr = torch.tensor([0.0722, 0.7152, 0.2126], device=bgr_video.device)
        cb_factor = 1.8556
        cr_factor = 1.5748
    elif matrix_coefficients == "bt601":
        luma_coefficients_bgr = torch.tensor([0.114, 0.587, 0.299], device=bgr_video.device)
        cb_factor = 1.772
        cr_factor = 1.402
    else:
        raise NotImplementedError(f"Standard {matrix_coefficients} is not implemented!")

    y  = torch.sum(bgr_video * luma_coefficients_bgr, dim=-1).unsqueeze(-1)
    cb = (b_channel - y) / cb_factor
    cr = (r_channel - y) / cr_factor

    cb += 128
    cr += 128

    ycbcr_video = torch.cat([y, cb, cr], dim=-1)

    return ycbcr_video


def ycbcr2bgr(yuv_video: torch.Tensor, matrix_coefficients: str) -> torch.Tensor:
    """
    Inverse of the conversion above.

    Args:
        yuv_video (torch.Tensor): _description_
        matrix_coefficients(str): _description

    Returns:
        torch.Tensor: _description_
    """
    yuv_video = yuv_video.float()
    
    y  = yuv_video[:, :, :, 0]
    cb = yuv_video[:, :, :, 1] - 128
    cr = yuv_video[:, :, :, 2] - 128

    if matrix_coefficients == "bt709":
        luma_coefficients_bgr = torch.tensor([0.0722, 0.7152, 0.2126], device=yuv_video.device)
        cb_factor = 1.8556
        cr_factor = 1.5748
    elif matrix_coefficients == "bt601":
        luma_coefficients_bgr = torch.tensor([0.114, 0.587, 0.299], device=yuv_video.device)
        cb_factor = 1.772
        cr_factor = 1.402
    else:
        raise NotImplementedError(f"Standard {matrix_coefficients} is not implemented!")

    b = (cb * cb_factor) + y
    r = (cr * cr_factor) + y
    g = ( y - (b * luma_coefficients_bgr[0] + r * luma_coefficients_bgr[2])) / luma_coefficients_bgr[1]

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