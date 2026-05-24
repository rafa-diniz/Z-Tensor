import torch

# Converts patchIds into pixel coordinates.
def patchId2Coords(patchId: torch.Tensor, block_width, blocks_in_plane_width):
    return torch.stack([patchId % blocks_in_plane_width * block_width, 
                           patchId // blocks_in_plane_width * block_width
                           ], dim=-1)


# Converts coordinates to patchIds taking into account how many blocks each row in the frame supports. 
def coords2PatchId(coords, block_width, blocks_in_plane_width):
    x, y = coords[..., 0], coords[..., 1]
    return (y // block_width) * blocks_in_plane_width + (x//block_width)



def sad(block_a, block_b):
    """Sum of absolute differences

    Args:
        block_a (_type_): _description_
        block_b (_type_): _description_

    Returns:
        _type_: _description_
    """
    scores = (block_a - block_b).abs().sum(dim=-1)
    
    return scores



def get_coords_of_candidate_patches(current_coords, search_radius):
    # For each patch, we explore the <block_width> neighborhood and store the patchIds of each neighboring patch.
    # Note that these patchIds refer to the patches with stride 1! Not the ones with stride <block_width>!
    # The idea is to slide the window centered in the current coordinates in all neighboring directions moving 1 pixel at a time,
    # and then store the IDs of these neighboring patches. They will later be compared to the original patch centered in the current
    # coordinates to find the neighboring 8x8 patch that is the most similar to the current one.
    
    offsets = torch.arange(-search_radius, search_radius+1).to(current_coords.device)
    offsets = torch.cartesian_prod(offsets, offsets) # shape: (number_of_offsets, 2)

    #               (number_of_blocks, 1, 2)      +  (1, number_of_offsets, 2) = (number_of_blocks, 2, number_of_offsets)
    offset_coords =   current_coords[:, None, :]  +  offsets[None, :, :] 

    return offset_coords


def get_invalid_coords(coords, block_width, h, w):
    invalid_x       = (coords[..., 0] < 0) | (coords[..., 0] + block_width >= w)
    invalid_y       = (coords[..., 1] < 0) | (coords[..., 1] + block_width >= h)
    
    mask_invalid    = invalid_x | invalid_y

    return mask_invalid


def block_matching(plane, block_width, search_radius, i_frame_indices):
    DEBUG = False

    device                 = plane.device

    plane                  = plane.to(torch.float32)
    num_frames, h, w       = plane.shape
    blocks_in_plane_height = h // block_width
    blocks_in_plane_width  = w // block_width

    # These two are different. Both divide the plane into patches with dimensions block_width x block_width, but
    # <unfold_window> has a stride of block_width because it is referencing all blocks that cover the image without overlapping, while
    # unfold_stride_1 is used for finding candidate patches in the previous frame, and must have overlaps to maximize coverage.
    # Because the planes are always padded if, it is guaranteed that the image can be perfectly divided by patches with dims block_width x block_width 
    unfold_window   = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=block_width)
    unfold_stride_1 = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=1)

    
    residue_size         = block_width * block_width

    # Make a fake unfold on a 1-frame plane to get the number of blocks per frame that will be created.
    # This is necessary to pre-allocate the motion_vectors and block_residuals tensors on the GPU.
    mock_plane_unfold    = unfold_window(torch.empty_like(plane[0].unsqueeze(0).unsqueeze(0)))
    num_blocks_per_frame = mock_plane_unfold.shape[-1]
    
    # Pre-allocate these two into memory. 
    motion_vectors  = torch.zeros((num_frames, num_blocks_per_frame, 2),            dtype=torch.int8).to(device)
    block_residuals = torch.zeros((num_frames, num_blocks_per_frame, residue_size), dtype=torch.uint8).to(device)

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
        
        blocks_plane1 = unfold_window(plane1.unsqueeze(0).unsqueeze(0))   # (1, 1 * ∏(kernel_size), num_blocks_plane1). Again, same dimensions. The number of blocks now is different because it uses a larger stride of <block_width>
        blocks_plane1 = blocks_plane1.squeeze(0).permute(1, 0)            # (num_blocks_plane1, ∏(kernel_size))

        if DEBUG:
            print(f"Plane shape: {plane.shape}, Blocks Plane0 shape: {blocks_plane0.shape}, Blocks Plane1 shape: {blocks_plane1.shape}")
            print(f"Blocks in plane width: {blocks_in_plane_width}, Blocks in plane height: {blocks_in_plane_height}")
        
        # the coordinates for all block_width * block_width blocks in plane1
        # shape: (num_blocks_plane1, 2) 
        coords_of_plane1_patches = patchId2Coords(torch.arange(len(blocks_plane1)), block_width, blocks_in_plane_width).to(device) 
        
        # the coordinates of the candidate patches. These are in (x,y)
        # shape: (num_blocks_plane1, num_candidates, 2)
        coords_of_candidate_patches = get_coords_of_candidate_patches(coords_of_plane1_patches, search_radius)

        # Not all coords of candidate patches are valid. Some can contain coordinates lower than 0,
        # higher than the image's width, etc. This function gives a mask with the invalid coords.
        # shape: (num_blocks_plane1, num_candidates)
        mask_invalid_coords  = get_invalid_coords(coords_of_candidate_patches, block_width, h, w)

        # The patchIds of each candidate patch in plane0's Ids. Notice that these are using plane0's blockIds,
        # not plane1's!
        # shape: (num_blocks_plane1, num_candidates)
        patchIds_of_candidates_in_plane0 = coords2PatchId(coords_of_candidate_patches, 1, w - block_width + 1) 
        
        # Turn the patchIds of the invalid patches to 0. This is not sufficient by itself, because these 
        # invalid patches might still have the best SAD score when using patch0. Later I'll also
        # turn the SAD scores of these patches to 0 to infinity to force SAD to ignore these patches.
        patchIds_of_candidates_in_plane0[mask_invalid_coords] = 0
        
        # The sad scores for each candidate patch in plane0. 
        # shape: (num_blocks_plane1, num_candidates)
        sad_scores_for_candidates_in_plane0 = sad(blocks_plane1[:, None, :], blocks_plane0[patchIds_of_candidates_in_plane0])

        # Make the scores for the invalid patches into infinity so they don't get accidentaly picked as the 
        # lowest sad scores
        sad_scores_for_candidates_in_plane0[mask_invalid_coords] = float('inf')
        
        
        # Get the Id of the patch with lowest sad score
        # shape: (num_blocks_plane1,)
        patchIds_with_lowest_sad_in_plane0 = patchIds_of_candidates_in_plane0[torch.arange(len(blocks_plane1)), torch.argmin(sad_scores_for_candidates_in_plane0, dim=-1)]

        # and convert it to coordinates
        # shape: (num_blocks_plane1, 2)
        coords_of_patches_with_lowest_sad = patchId2Coords(patchIds_with_lowest_sad_in_plane0, 1, w-block_width+1).to(device)

        # The motion vectors for each frame are the coordinates of the patch with lowest SAD score - the coordinates of the patches in plane1
        motion_vectors[frame_idx]  = coords_of_patches_with_lowest_sad - coords_of_plane1_patches

        # And the residuals are just the blocks in frame1 - the best candidate block in frame0
        block_residuals[frame_idx] = blocks_plane1 - blocks_plane0[patchIds_with_lowest_sad_in_plane0]

    return motion_vectors, block_residuals



def deconstruct_block_matching(planes, i_frame_indices):
    planes_decoded = []
    for plane_id in range(len(planes)):
        frames                      = planes[plane_id]['frames'].cuda()
        motion_vectors              = planes[plane_id]['motion_vectors'].cuda()
        residual_blocks             = planes[plane_id]['residual_blocks'].cuda()
        original_height             = planes[plane_id]['original_h']
        original_width              = planes[plane_id]['original_w']
        padded_height               = planes[plane_id]['padded_h']
        padded_width                = planes[plane_id]['padded_w']
        block_width                 = planes[plane_id]['block_width']
        num_frames, num_blocks, num_elements_per_block = planes[plane_id]['residual_blocks'].shape
        blocks_in_plane_width  = padded_width // block_width
        
        
        for frame_idx in range(1, num_frames):
            if frame_idx in i_frame_indices:
                continue    
            
            # The patch ids for each block
            # shape: (num_blocks_in_plane1,)
            patch_ids_for_patches_in_plane1 = torch.arange(num_blocks).cuda()

            # The coordinate version of those patches
            # shape: (num_blocks_in_plane1, 2)
            coords_for_patches_in_plane1    = patchId2Coords(patch_ids_for_patches_in_plane1, block_width, blocks_in_plane_width).cuda()

            y_coords_for_patches_in_plane1  = coords_for_patches_in_plane1[..., 1]
            x_coords_for_patches_in_plane1  = coords_for_patches_in_plane1[..., 0]
            coords_offsets                  = torch.arange(block_width).cuda()

            # The coordinates of the patches in the height axis
            # shape: (num_blocks_in_plane1, block_width, 1)
            y_patches = y_coords_for_patches_in_plane1[:, None, None] + coords_offsets[None, :, None].cuda()

            # The coordinates of the patches in the width axis
            # shape: (num_blocks_in_plane1, 1, block_width)
            x_patches = x_coords_for_patches_in_plane1[:, None, None] + coords_offsets[None, None, :].cuda()

            # The motion vectors
            # shapes: (num_blocks_in_plane1, 1, 1)
            dy = motion_vectors[frame_idx, :, 1][:, None, None].cuda()
            dx = motion_vectors[frame_idx, :, 0][:, None, None].cuda()
            
            residue = residual_blocks[frame_idx].reshape(num_blocks, block_width, block_width).cuda()

            frames[frame_idx, y_patches, x_patches] = frames[frame_idx-1, y_patches+dy, x_patches+dx] + residue


        frames = frames[:, : original_height, : original_width]
        frames = frames.to(torch.uint8)
        planes_decoded.append(frames)

    return planes_decoded