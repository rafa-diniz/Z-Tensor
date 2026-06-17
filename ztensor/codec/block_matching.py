import torch
import typing

def _patchId2Coords(patchIds: torch.Tensor, block_size: int, blocks_in_plane_width: int) -> torch.Tensor:
    """Converts patchIds into (X,Y) coordinates.

    SHAPE WARNING: 'num_blocks' changes depending if this function was called for blocks_plane0 or blocks_plane1, since these 2 have different strides!

    
    Args:
        patchId (torch.Tensor): Tensor containing the block IDs. Shape = (num_blocks).
        block_size (int): The size of each block. 
        blocks_in_plane_width (int): The number of blocks across the plane's horizontal axis.

    Returns:
        torch.Tensor: A Tensor containing the corresponding (X,Y) coords of each block. Shape = (num_blocks, 2)
    """
    return torch.stack([patchIds % blocks_in_plane_width * block_size, 
                           patchIds // blocks_in_plane_width * block_size
                           ], dim=-1)



def _coords2PatchId(coords: torch.Tensor, block_size: int, blocks_in_plane_width: int) -> torch.Tensor:
    """Converts coordinates to patchIds taking into account how many blocks each row in the frame supports. 

    SHAPE WARNING: 'num_blocks' changes depending if this function was called for blocks_plane0 or blocks_plane1, since these 2 have different strides! Originally this function was only called for getting the coords of the candidate patches,
        so 'num_blocks' should be the number of blocks in plane0, NOT plane1. But beware, because if this function gets called for the coords in block0, the shape might change.


    Args:
        coords (torch.Tensor): Tensor containing the coordinates of each block. Shape = (num_blocks, num_candidates, 2).
        block_size (int): The size of each block.
        blocks_in_plane_width (int): The number of blocks across the plane's horizontal axis.

    Returns:
        torch.Tensor: A Tensor containing the corresponding block IDs. Shape = (num_blocks, num_candidates)
    """
    x, y = coords[..., 0], coords[..., 1]

    return (y // block_size) * blocks_in_plane_width + (x//block_size)



def _sad(block_a: torch.Tensor, block_b: torch.Tensor) -> torch.Tensor:
    """Sum of absolute differences

    SHAPE WARNING: 'num_blocks' changes depending if this function was called for blocks_plane0 or blocks_plane1, since these 2 have different strides!
    
    Args:
        block_a (torch.Tensor): the reference block.  Shape = (num_blocks,              1, block_size * block_size)
        block_b (torch.Tensor): the candidate blocks. Shape = (num_blocks, num_candidates, block_size * block_size)

    Returns:
        torch.Tensor: The SAD score of each candidate block. Shape = (num_blocks, num_candidates)
    """
    scores = (block_a - block_b).abs().sum(dim=-1)
    
    return scores



def _get_coords_of_candidate_patches(current_coords: torch.Tensor, search_radius: int) -> torch.Tensor:
    """
    For each patch, we explore the <block_size> neighborhood and store the patchIds of each neighboring patch.
    Note that these patchIds refer to the patches with stride 1! Not the ones with stride <block_size>!
    The idea is to slide the window centered in the current coordinates in all neighboring directions moving 1 pixel at a time,
    and then store the IDs of these neighboring patches. They will later be compared to the original patch centered in the current
    coordinates to find the neighboring 8x8 patch that is the most similar to the current one.

    SHAPE WARNING: 'num_blocks' changes depending if this function was called for blocks_plane0 or blocks_plane1, since these 2 have different strides!
    

    Args:
        current_coords (torch.Tensor): The (X,Y) candidates for each block. Shape = (num_blocks, 2). 
        search_radius (int): The size of the search window.

    Returns:
        torch.Tensor: The offset coordinates of each candidate patch. Shape = (num_blocks, 2 * search_window + 1, 2). 
        TERMINOLOGY ALERT : 2 * search_window + 1 is hard to write and read. From now, it will be called 'num_candidates', as it refers to the number of candidate patches for each block.
    """    
    offsets = torch.arange(-search_radius, search_radius+1).to(current_coords.device)
    offsets = torch.cartesian_prod(offsets, offsets) # shape: (number_of_offsets, 2)

    # Shapes:        (number_of_blocks, 1, 2)     +  (1, number_of_offsets, 2) = (number_of_blocks, 2, number_of_offsets)
    offset_coords =   current_coords[:, None, :]  +  offsets[None, :, :] 

    return offset_coords


def _get_invalid_coords(coords: torch.Tensor, block_size: int, h: int, w: int) -> torch.Tensor:
    """X

    SHAPE WARNING: 'num_blocks' changes depending if this function was called for blocks_plane0 or blocks_plane1, since these 2 have different strides!

    Args:
        coords (torch.Tensor): The (X, Y) coordinates of each candidate block. Shape = (num_blocks, num_candidates, 2)
        block_size (int): The size of each block.
        h (int): The plane's height
        w (int): The plane's width

    Returns:
        torch.Tensor: A mask with the invalid candidates. Shape = (num_blocks, num_candidates)
    """
    invalid_x       = (coords[..., 0] < 0) | (coords[..., 0] + block_size > w)
    invalid_y       = (coords[..., 1] < 0) | (coords[..., 1] + block_size > h)
    
    mask_invalid    = invalid_x | invalid_y

    return mask_invalid


def block_matching(plane: torch.Tensor, block_size: int, search_radius: int, i_frame_indices: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Runs Block-Matching Motion Estimation (https://en.wikipedia.org/wiki/Block-matching_algorithm) on the plane to find matching macroblocks
    inside the video. This function implements block matching on PyTorch tensors and runs fully tensorized.

    Args:
        plane (torch.Tensor): The plane where block matching will run on. Shape = (num_frames, height, width,)
        block_size (int): The size of each block. This is simply the argument 'args.block_size'.
        search_radius (int): The size of the search window. This is simply the argument 'args.search_window'.
        i_frame_indices (torch.Tensor): The indices of the I-frames. 

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: A tuple containing the motion vectors and residuals for each block.
        Shape motion vectors: (num_frames, num_blocks, 2)
        Shape residuals:      (num_frames, num_blocks, block_size * block_size)
    """
    DEBUG = False

    device                 = plane.device

    plane                  = plane.to(torch.float32)
    num_frames, h, w       = plane.shape
    blocks_in_plane_height = h // block_size
    blocks_in_plane_width  = w // block_size

    # These two are different. Both divide the plane into patches with dimensions block_size x block_size, but
    # <unfold_window> has a stride of block_size because it is referencing all blocks that cover the image without overlapping, while
    # unfold_stride_1 is used for finding candidate patches in the previous frame, and must have overlaps to maximize coverage.
    # Because the planes are always padded if, it is guaranteed that the image can be perfectly divided by patches with dims block_size x block_size 
    unfold_window   = torch.nn.Unfold(kernel_size=(block_size,block_size), stride=block_size)
    unfold_stride_1 = torch.nn.Unfold(kernel_size=(block_size,block_size), stride=1)

    
    residue_size         = block_size * block_size

    # Make a fake unfold on a 1-frame plane to get the number of blocks per frame that will be created.
    # This is necessary to pre-allocate the motion_vectors and block_residuals tensors on the GPU.
    mock_plane_unfold    = unfold_window(torch.empty_like(plane[0].unsqueeze(0).unsqueeze(0)))
    num_blocks_per_frame = mock_plane_unfold.shape[-1]
    
    # Pre-allocate these two into memory. 
    motion_vectors  = torch.zeros((num_frames, num_blocks_per_frame, 2),            dtype=torch.int8).to(device)
    block_residuals = torch.zeros((num_frames, num_blocks_per_frame, residue_size), dtype=torch.int16).to(device)

    for frame_idx in range(1, num_frames):
        if frame_idx in i_frame_indices:
            continue

        if DEBUG:
            print(f"Processing: Frame{frame_idx}")

        # Stores the dx and dy motion vectors and the residuals that will be used for reconstruction.
        # This is what gets returned by the function and will be serialized.

        plane0 = plane[frame_idx-1]
        plane1 = plane[frame_idx]

        blocks_plane0 = unfold_stride_1(plane0.unsqueeze(0).unsqueeze(0)) # (1, 1 * ∏(kernel_size), num_blocks_plane0). Since we're dealing with individual planes, C=1. And B = 1 too.
        blocks_plane0 = blocks_plane0.squeeze(0).permute(1, 0)            # (num_blocks_plane0, ∏(kernel_size))
        
        blocks_plane1 = unfold_window(plane1.unsqueeze(0).unsqueeze(0))   # (1, 1 * ∏(kernel_size), num_blocks_plane1). Again, same dimensions. The number of blocks now is different because it uses a larger stride of <block_size>
        blocks_plane1 = blocks_plane1.squeeze(0).permute(1, 0)            # (num_blocks_plane1, ∏(kernel_size))
        
        if DEBUG:
            print(f"Plane shape: {plane.shape}, Blocks Plane0 shape: {blocks_plane0.shape}, Blocks Plane1 shape: {blocks_plane1.shape}")
            print(f"Blocks in plane width: {blocks_in_plane_width}, Blocks in plane height: {blocks_in_plane_height}")
        
        # the coordinates for all block_size * block_size blocks in plane1
        # shape: (num_blocks_plane1, 2) 
        coords_of_plane1_patches = _patchId2Coords(torch.arange(len(blocks_plane1)), block_size, blocks_in_plane_width).to(device) 
        
        # the coordinates of the candidate patches. These are in (x,y)
        # shape: (num_blocks_plane1, num_candidates, 2)
        coords_of_candidate_patches = _get_coords_of_candidate_patches(coords_of_plane1_patches, search_radius)

        # Not all coords of candidate patches are valid. Some can contain coordinates lower than 0,
        # higher than the image's width, etc. This function gives a mask with the invalid coords.
        # shape: (num_blocks_plane1, num_candidates)
        mask_invalid_coords  = _get_invalid_coords(coords_of_candidate_patches, block_size, h, w)

        # The patchIds of each candidate patch in plane0's Ids. Notice that these are using plane0's blockIds,
        # not plane1's!
        # shape: (num_blocks_plane1, num_candidates)
        patchIds_of_candidates_in_plane0 = _coords2PatchId(coords_of_candidate_patches, 1, w - block_size + 1) 
        
        # Turn the patchIds of the invalid patches to 0. This is not sufficient by itself, because these 
        # invalid patches might still have the best SAD score when using patch0. Later I'll also
        # turn the SAD scores of these patches to 0 to infinity to force SAD to ignore these patches.
        patchIds_of_candidates_in_plane0[mask_invalid_coords] = 0
        
        # The sad scores for each candidate patch in plane0. 
        # shape: (num_blocks_plane1, num_candidates)
        sad_scores_for_candidates_in_plane0 = _sad(blocks_plane1[:, None, :], blocks_plane0[patchIds_of_candidates_in_plane0])

        # Make the scores for the invalid patches into infinity so they don't get accidentaly picked as the 
        # lowest sad scores
        sad_scores_for_candidates_in_plane0[mask_invalid_coords] = float('inf')
        
        
        # Get the Id of the patch with lowest sad score
        # shape: (num_blocks_plane1,)
        patchIds_with_lowest_sad_in_plane0 = patchIds_of_candidates_in_plane0[torch.arange(len(blocks_plane1)), torch.argmin(sad_scores_for_candidates_in_plane0, dim=-1)]

        # and convert it to coordinates
        # shape: (num_blocks_plane1, 2)
        coords_of_patches_with_lowest_sad = _patchId2Coords(patchIds_with_lowest_sad_in_plane0, 1, w-block_size+1).to(device)

        # The motion vectors for each frame are the coordinates of the patch with lowest SAD score - the coordinates of the patches in plane1
        motion_vectors[frame_idx]  = coords_of_patches_with_lowest_sad - coords_of_plane1_patches

        # And the residuals are just the blocks in frame1 - the best candidate block in frame0
        block_residuals[frame_idx] = blocks_plane1 - blocks_plane0[patchIds_with_lowest_sad_in_plane0]

    return motion_vectors, block_residuals



def deconstruct_block_matching(planes: typing.List[typing.Dict], i_frame_indices: typing.Set, device: torch.Device) -> typing.List[torch.Tensor]:
    """Uses the deserialized payload to reconstruct the video using the motion vectors, residuals and all other info that went into the serialized file.

    Args:
        planes (typing.List[typing.Dict]): The dict containing the per-plane information. It contains the i_frames, the original height/width, padded height/width, the motion vectors, residuals, etc. Check the code below to see everything in it.
        i_frame_indices (typing.Set): A set containing the indices for the i_frames
        device (torch.Device): The device to store the reconstructed frames in

    Returns:
        typing.List[torch.Tensor]: A list of the the reconstructed planes. Shape = (num_frames, height, width)
    """
    planes_decoded = []
    for plane_id in range(len(planes)):
        frames                                         = planes[plane_id]['frames'].to(device)
        motion_vectors                                 = planes[plane_id]['motion_vectors'].to(device)
        residual_blocks                                = planes[plane_id]['residual_blocks'].to(device)
        original_height                                = planes[plane_id]['original_h']
        original_width                                 = planes[plane_id]['original_w']
        padded_height                                  = planes[plane_id]['padded_h']
        padded_width                                   = planes[plane_id]['padded_w']
        block_size                                     = planes[plane_id]['block_size']
        num_frames, num_blocks, num_elements_per_block = planes[plane_id]['residual_blocks'].shape
        blocks_in_plane_width                          = padded_width // block_size
        
        is_lossy = residual_blocks.dtype == torch.int16
        if is_lossy:
            frames = frames.to(torch.int16)
        else:
            frames = frames.to(torch.uint8)
        
        for frame_idx in range(1, num_frames):
            if frame_idx in i_frame_indices:
                continue    
            
            # The patch ids for each block
            # shape: (num_blocks_in_plane1,)
            patch_ids_for_patches_in_plane1 = torch.arange(num_blocks).to(device)

            # The coordinate version of those patches
            # shape: (num_blocks_in_plane1, 2)
            coords_for_patches_in_plane1    = _patchId2Coords(patch_ids_for_patches_in_plane1, block_size, blocks_in_plane_width)

            y_coords_for_patches_in_plane1  = coords_for_patches_in_plane1[..., 1]
            x_coords_for_patches_in_plane1  = coords_for_patches_in_plane1[..., 0]
            coords_offsets                  = torch.arange(block_size).to(device)

            # The coordinates of the patches in the height axis
            # shape: (num_blocks_in_plane1, block_size, 1)
            y_patches = y_coords_for_patches_in_plane1[:, None, None] + coords_offsets[None, :, None]

            # The coordinates of the patches in the width axis
            # shape: (num_blocks_in_plane1, 1, block_size)
            x_patches = x_coords_for_patches_in_plane1[:, None, None] + coords_offsets[None, None, :]

            # The motion vectors
            # shapes: (num_blocks_in_plane1, 1, 1)
            dy = motion_vectors[frame_idx, :, 1][:, None, None]
            dx = motion_vectors[frame_idx, :, 0][:, None, None]
            
            residue = residual_blocks[frame_idx].reshape(num_blocks, block_size, block_size).to(device)
            
            # Clamping prevents modulo problems when adding these up
            if is_lossy:
                frames[frame_idx, y_patches, x_patches] = (frames[frame_idx-1, y_patches+dy, x_patches+dx] + residue).clamp(0, 255)
            else:
                frames[frame_idx, y_patches, x_patches] = frames[frame_idx-1, y_patches+dy, x_patches+dx] + residue

        frames = frames[:, : original_height, : original_width]
        frames = frames.to(torch.uint8)
        planes_decoded.append(frames)

    return planes_decoded