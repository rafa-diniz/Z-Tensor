import torch

def patchId2Coords(patchId, width, search_window):
    return torch.as_tensor([patchId // (width-(search_window-1)), patchId % (width-(search_window-1))], dtype=torch.int32)

def coords2PatchId(coords, width, search_window):
    return coords[0] * (width-(search_window-1)) + coords[1]

def sad(block_a, block_b):
    print(block_a - block_b)
    scores = (block_a - block_b).abs().sum(dim=-1)
    print(scores)
    return scores

def block_matching(plane, search_window):
    search_window = 8
    plane         = plane.to(torch.float32)
    plane0        = plane[0]
    plane1        = plane[1]
    h, w          = plane[0].shape


    unfold  = torch.nn.Unfold(kernel_size=(search_window,search_window), stride=search_window) # unfold the plane into all possible 2x2 blocks without padding 
    fold    = torch.nn.Fold(plane1.shape, kernel_size=(search_window,search_window), stride=search_window)


    blocks_plane0 = unfold(plane0.unsqueeze(0).unsqueeze(0)) # (1, 1 * ∏(kernel_size), totalNumberOfBlocks). Since we're dealing with individual planes, C=1. And B = 1 too.
    blocks_plane0 = blocks_plane0.squeeze(0).permute(1, 0) #  (B, C * ∏(kernel_size), totalNumberOfBlocks) -> (C * ∏(kernel_size), totalNumberOfBlocks) -> (totalNumberOfBlocks, ∏(kernel_size))
    
    blocks_plane1 = unfold(plane1.unsqueeze(0).unsqueeze(0))
    blocks_plane1 = blocks_plane1.squeeze(0).permute(1, 0)

    print(f"Plane shape: {plane0.shape}, Blocks shape: {blocks_plane1.shape}")
    
    
    #print(blocks_plane1.shape, fold(blocks_plane1.permute(1, 0)).reshape(plane0.shape).shape)

    for patchId in range(len(blocks_plane1)):
        current_coords                    = patchId2Coords(patchId, w, search_window)
        patchIds_compare_to_in_prev_frame = []

        for i in range(search_window):
            for j in range(search_window):
                patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_coords[0]+i, current_coords[1]+j], w, search_window))

                if (current_coords[0] - i > 0 and current_coords[1]-j > 0) and (i + j != 0):
                    patchIds_compare_to_in_prev_frame.append(coords2PatchId([current_coords[0]-i, current_coords[1]-j], w, search_window))

        print(patchIds_compare_to_in_prev_frame)
        patchIds_compare_to_in_prev_frame = torch.as_tensor(patchIds_compare_to_in_prev_frame, device=plane0.device)
        
        sad_scores = sad(blocks_plane1[patchId], blocks_plane0[patchIds_compare_to_in_prev_frame])
        best_patch = patchIds_compare_to_in_prev_frame[torch.argmin(sad_scores)]
        print(best_patch)

        motion_vector = current_coords - patchId2Coords(best_patch, w, search_window)
        print(motion_vector)
        raise Exception
        #print(torch.equal(blocks[patchId], 
        #                  plane1[coords[0]: coords[0]+search_window, coords[1]:coords[1]+search_window].flatten()
        #                  )
        #    )

    
    raise NotImplementedError