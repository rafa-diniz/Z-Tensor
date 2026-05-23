import torch
import numpy as np


def serialize_header(pixel_format: str, 
                    quantization_parameter: int,
                    i_frame_indices: torch.Tensor,
                    block_width: int,
                    num_frames: int,
                    num_planes: int
                     ) -> bytes:
    """Constructs the header for the .ztensor file format. 

    Args:
        pixel_format (str): The pixel format. Either RGB3, I422 or I420
        quantization_parameter (int): The parameter that stores which quantization option was used for the video
        i_frame_indices (torch.Tensor): The indices of the i_frames
        block_width (int): The with of the motion block. They're square, so no need to pass the height as it is the same as the width. 
        num_frames (int): The number of frames in the video
        num_planes (int): The number of planes in the video

    Returns:
        bytes: The header
    """
    header  = bytes()

    header += pixel_format.encode('ascii')
    header += quantization_parameter.to_bytes(1, signed=False)            # uint8  value for the quantization parameter. 1 = Linear (less aggresive)
    header += len(i_frame_indices).to_bytes(4, signed=False)              # uint32 the number of i-frames in the video
    header += i_frame_indices.cpu().numpy().astype(np.uint32).tobytes()   # uint32 indices of the i-frames
    header += block_width.to_bytes(4, signed=False)                       # uint32  the size of the motion blocks
    
    header += num_planes.to_bytes(4,  signed=False)                       # uint32 the number of planes in the video.
    header += num_frames.to_bytes(4,  signed=False)                       # uint32 the number of frames

    return header


def serialize_payload(num_motion_blocks_per_frame: int, 
                      motion_blocks: dict, 
                      plane: torch.Tensor, 
                      i_frame_indices: torch.Tensor, 
                      original_plane_h: int, 
                      original_plane_w: int) -> bytes:
    """Serialize the payload of the encoded video

    Args:
        num_motion_blocks_per_frame (int): The number of motion blocks in each frame #TODO maybe remove? I think this is always the same value as just doing len(motion_blocks[1]). 
        motion_blocks (dict): The motion blocks. Stores the motion vectors and the residue for each block
        plane (torch.Tensor): The unprocessed frame. Will be used to store the i-frames as themselves instead of processed blocks.
        i_frame_indices (torch.Tensor): The indices of the i-frames
        original_plane_h (int): The original height of the plane.
        original_plane_w (int): The original width of the plane.

    Returns:
        bytes: _description_
    """
    payload = bytes()

    payload += original_plane_h.to_bytes(4,  signed=False)             # uint32 the height of the video
    payload += original_plane_w.to_bytes(4,  signed=False)             # uint32 the width of the video
    payload += num_motion_blocks_per_frame.to_bytes(4, signed=False)   # uint32 the number of motion blocks in each video frame


    # TODO this block is killing performance. I thought about serializing dx, dy and residue right inside block_matching, but that was also killing performance
    # because it was sending each block to the GPU sequentially, which is too slow. Maybe look into a way of writing the serialized dx, dy and residue values
    # into the GPU itself and then send it all back to the CPU at once. Not even as a byte array, maybe just concat the tensors on top of each other and then
    # send everything to the CPU at once instead of individually. This can be done inside block_matching.py
    for frame_idx, frame in enumerate(plane):
        print(f"Serializing frame {frame_idx}")
        # If frame is an I-frame, store it as-is
        if frame_idx in i_frame_indices:
            payload += frame.to(torch.uint8).cpu().numpy().tobytes()

        # If not, store its motion blocks
        else:
            block_movements = motion_blocks[frame_idx]
            for block in block_movements:
                dx, dy, residue = block
                
                # int8 is fine for storing the motion vectors because the search_window parameter is small (usually < 10).
                # So this value is always in the [-10, 10] interval.
                # A 127 search window is also unrealistic because it would take years to encode the video
                # even on the best codecs. So int8 it is.
                payload += int(dx).to_bytes(1, signed=True)  # int8 the block's horizontal motion. 

                payload += int(dy).to_bytes(1, signed=True)  # uint8 the block's vertical motion.
                payload += residue.to(torch.uint8).cpu().numpy().tobytes() # uint8 the stored residuals

    return payload
