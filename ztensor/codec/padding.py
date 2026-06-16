import torch


def pad_plane(plane: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pads the current plane so it is perfectly covered by 
    block_size x block_size blocks. Dimensions that are already
    perfectly divisible by block_size are ignored

    Args:
        plane (torch.Tensor): A (T, H, W) plane
        block_size (int): The block's width. Since it's a perfect square, the height is the same as the width

    Returns:
        torch.Tensor: The padded plane
    """
    _, h, w = plane.shape

    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    # Pad only the right and the bottom of the plane.
    # This makes the cropping easier when decoding. 
    # just do plane = plane[ : H, : W] !
    plane = torch.nn.functional.pad(plane, (0, pad_w, 0, pad_h), 'reflect')

    return plane
