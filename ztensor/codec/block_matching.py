import torch

# Converts patchIds into pixel coordinates.
def patchId2Coords(patchId, block_width, blocks_in_plane_width):
    return torch.as_tensor([patchId % blocks_in_plane_width * block_width, 
                           patchId // blocks_in_plane_width * block_width
                           ], dtype=torch.int32)


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



def get_candidate_patches(current_coords, search_radius, block_width, w, h):
    # For each patch, we explore the <block_width> neighborhood and store the patchIds of each neighboring patch.
    # Note that these patchIds refer to the patches with stride 1! Not the ones with stride <block_width>!
    # The idea is to slide the window centered in the current coordinates in all neighboring directions moving 1 pixel at a time,
    # and then store the IDs of these neighboring patches. They will later be compared to the original patch centered in the current
    # coordinates to find the neighboring 8x8 patch that is the most similar to the current one.

    offsets = [ [0, 0] ]
    for s in ["+", "-"]:
        for i in range(search_radius+1):
            if s == "-":
                i = -i
            for j in range(search_radius+1):
                if s == "-":
                    j = -j
                if (i != 0) or (j != 0):
                    offsets.append([i, j])
    
    offset_coords = current_coords + torch.as_tensor(offsets, device=current_coords.device)
    mask_negative = torch.any(offset_coords < 0, dim=-1)
    mask_within_max_width  = offset_coords[..., 0] + block_width < w
    mask_within_max_height = offset_coords[..., 1] + block_width < h
    
    offset_coords = offset_coords[~mask_negative & mask_within_max_width & mask_within_max_height]

    patchIds_compare_to_in_prev_frame = coords2PatchId(offset_coords, 1, w-block_width+1)
    patchIds_compare_to_in_prev_frame = torch.as_tensor(patchIds_compare_to_in_prev_frame, dtype=torch.int32)

    return patchIds_compare_to_in_prev_frame


def block_matching(plane, block_width, search_radius):
    DEBUG = False

    device                 = plane.device

    plane                  = plane.to(torch.float32)
    num_frames, h, w       = plane.shape
    blocks_in_plane_height = h // block_width
    blocks_in_plane_width  = w // block_width

    # These two are different. Both divide the plane into patches with dimensions block_width x block_width, but the
    # first one has a stride of block_width because it is referencing all block_width x block_width patches that cover the image perfectly, while
    # the second is using for finding candidate patches in the previous frame.
    # Because the image is padded earlier, it is guaranteed that the image can be divided into block_width x block_width patches
    unfold_window   = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=block_width)
    unfold_stride_1 = torch.nn.Unfold(kernel_size=(block_width,block_width), stride=1)

    
    residue_size         = block_width * block_width
    mock_plane_unfold    = unfold_window(torch.empty_like(plane[0].unsqueeze(0).unsqueeze(0)))
    num_blocks_per_frame = mock_plane_unfold.shape[-1]
    
    # Pre-allocate these two
    motion_vectors  = torch.zeros((num_frames, num_blocks_per_frame, 2),            dtype=torch.int8).to(device)
    block_residuals = torch.zeros((num_frames, num_blocks_per_frame, residue_size), dtype=torch.uint8).to(device)

    for frame_idx in range(1, num_frames):
        if True:
            print(f"Processing: Frame{frame_idx}")

        # Stores the dx and dy motion vectors and the residuals that will be used for reconstruction.
        # This is what gets returned by the function and will be serialized.

        plane0 = plane[frame_idx-1]
        plane1 = plane[frame_idx]

        blocks_plane0 = unfold_stride_1(plane0.unsqueeze(0).unsqueeze(0)) # (1, 1 * ∏(kernel_size), totalNumberOfBlocks). Since we're dealing with individual planes, C=1. And B = 1 too.
        blocks_plane0 = blocks_plane0.squeeze(0).permute(1, 0)            # (B, C * ∏(kernel_size), totalNumberOfBlocks) -> (C * ∏(kernel_size), totalNumberOfBlocks) -> (totalNumberOfBlocks, ∏(kernel_size))
        
        blocks_plane1 = unfold_window(plane1.unsqueeze(0).unsqueeze(0))   # (1, 1 * ∏(kernel_size), totalNumberOfBlocks). Again, same dimensions. The number of blocks now is different because it uses a larger stride of <block_width>
        blocks_plane1 = blocks_plane1.squeeze(0).permute(1, 0)            # (totalNumberOfBlocks, ∏(kernel_size))

        if DEBUG:
            print(f"Plane shape: {plane.shape}, Blocks Plane0 shape: {blocks_plane0.shape}, Blocks Plane1 shape: {blocks_plane1.shape}")
            print(f"Blocks in plane width: {blocks_in_plane_width}, Blocks in plane height: {blocks_in_plane_height}")
        
        # Loop over every block_width x block_width patch with stride block_width
        for patchId in range(len(blocks_plane1)):
            # pixel coordinates are always the TOP LEFT CORNER pixel! It's a simple 2-digit tuple because ince all blocks have width <block_width> 
            # and also height <block_width>, we can always get the full patch by adding + <block_width> to x and y.
            current_coords                     = patchId2Coords(patchId, block_width, blocks_in_plane_width).to(device)
            current_x, current_y               = current_coords


            if DEBUG:
                print(f"Current coords: {current_coords}, Patch: {patchId}")
            
            candidate_patches = get_candidate_patches(current_coords, search_radius, block_width, w, h)
            candidate_patches = candidate_patches.to(device)

            if DEBUG:
                print(f"Patch Ids to compare: {candidate_patches}")
            
            sad_scores    = sad(blocks_plane1[patchId], blocks_plane0[candidate_patches])

            # Get coords of the candidate patch with lowest SAD score, which is the best patch (maybe change variable name to candidate_patches?).
            best_patchid            = candidate_patches[torch.argmin(sad_scores)] 
            coords_patch_lowest_sad = patchId2Coords(best_patchid, 1, w-block_width+1).to(device)

            # these are the motion vectors that tell us how to get to the candidate patch starting from the current patch.
            # since the block sizes are all equal, we can use the motion vectors to tell us how to reconstruct the current frame's patch
            # using just relative motion vectors from the previous frame.
            dx, dy  = coords_patch_lowest_sad - current_coords

            # prev are the best patch's pixel values in the previous frame
            prev    = plane0[current_y + dy: current_y + dy + block_width, current_x + dx: current_x + dx + block_width].flatten()

            if DEBUG:
                print(f"Patch Id lowest SAD: {best_patchid}")
                print(current_y + dy, current_y + dy + block_width, current_x + dx, current_x + dx + block_width, prev.shape)

            # Get the residue between the current patch and the best patch in the previous frame and
            # store the motion vector and the residue
            residue = blocks_plane1[patchId] - prev
            
            motion_vectors[frame_idx][patchId] = torch.as_tensor([dx, dy], dtype=torch.int8, device=device)
            block_residuals[frame_idx][patchId] = residue.to(torch.uint8)
            

    return motion_vectors, block_residuals