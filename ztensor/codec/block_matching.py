import torch

# Converts patchIds into pixel coordinates.
def patchId2Coords(patchId, block_width, blocks_in_plane_width):
    return torch.as_tensor([patchId % blocks_in_plane_width * block_width, 
                           patchId // blocks_in_plane_width * block_width
                           ], dtype=torch.int32)

# Converts coordinates to patchIds taking into account how many blocks each row in the frame supports. 
def coords2PatchId(coords, block_width, blocks_in_plane_width):
    x, y = coords
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
    patchIds_compare_to_in_prev_frame  = []
    
    current_x, current_y               = current_coords

    #TODO This is begging for a refactor. It's just too slow and should run on the GPU.
    for i in range(search_radius+1):
        for j in range(search_radius+1):
            # The rightmost pixel index has to be less than w, and the bottommost one has to be less than h.
            # < instead of <= because we're using indexes, and if the rightmost index == w, this will cause an out of bounds error
            if current_x+i+block_width < w: 
                if  current_y+j+block_width < h:
                    patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x+i, current_y+j], 1, w-block_width+1)) # This -(block_width) in w-(block_width) is necessary because we currently don't support
                                                                                                                            # videos with dimensions that aren't perfectly divisible by block_width x block_width
                

                # Skip patches where i + j = 0, because that one was already added in the line above. It's a base one that always gets added.
                if  current_y - j >= 0 and (i +j != 0):
                    patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x+i, current_y-j], 1, w-block_width+1))
            
            if current_x - i >= 0:
                if current_y+j+block_width < h and (i +j != 0):
                    patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x-i, current_y+j], 1, w-block_width+1))

                if current_y - j >= 0 and (i +j != 0):
                    patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_x-i, current_y-j], 1, w-block_width+1))


    return torch.as_tensor(patchIds_compare_to_in_prev_frame, dtype=torch.int32)


def block_matching(plane, block_width, search_radius):
    DEBUG = False

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


    motion_vectors_patches = {}
    for idx in range(1, num_frames):
        if DEBUG:
            print(f"Processing: Frame{idx}")

        # Stores the dx and dy motion vectors and the residuals that will be used for reconstruction.
        # This is what gets returned by the function and will be serialized.
        motion_vectors_patches[idx] = []

        plane0 = plane[idx-1]
        plane1 = plane[idx]

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
            current_coords                     = patchId2Coords(patchId, block_width, blocks_in_plane_width)
            current_x, current_y               = current_coords


            if DEBUG:
                print(f"Current coords: {current_coords}, Patch: {patchId}")
            
            candidate_patches = get_candidate_patches(current_coords, search_radius, block_width, w, h)
            candidate_patches = candidate_patches.to(plane.device)
            
            if DEBUG:
                print(f"Patch Ids to compare: {candidate_patches}")
            
            sad_scores    = sad(blocks_plane1[patchId], blocks_plane0[candidate_patches])

            # Get coords of the candidate patch with lowest SAD score, which is the best patch (maybe change variable name to candidate_patches?).
            best_patchid            = candidate_patches[torch.argmin(sad_scores)] 
            coords_patch_lowest_sad = patchId2Coords(best_patchid, 1, w-block_width+1)

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
            residue = residue
            motion_vectors_patches[idx].append([dx, dy, residue])

            
    return motion_vectors_patches