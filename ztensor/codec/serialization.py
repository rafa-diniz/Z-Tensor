import torch
import zstandard
import numpy as np

from typing import Tuple, List, Dict, Set


from ztensor.effects import quantization

def serialize_header(pixel_format: str, 
                    quantization_parameter: int,
                    i_frame_indices: torch.Tensor,
                    block_size: int,
                    num_frames: int,
                    num_planes: int
                     ) -> bytes:
    """Constructs the header for the .ztensor file format. 

    Args:
        pixel_format (str): The pixel format. Either RGB3, I422 or I420
        quantization_parameter (int): The parameter that stores which quantization option was used for the video
        i_frame_indices (torch.Tensor): The indices of the i_frames
        block_size (int): The with of the motion block. They're square, so no need to pass the height as it is the same as the width. 
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
    header += block_size.to_bytes(4, signed=False)                       # uint32  the size of the motion blocks
    
    header += num_planes.to_bytes(4,  signed=False)                       # uint32 the number of planes in the video.
    header += num_frames.to_bytes(4,  signed=False)                       # uint32 the number of frames

    return header


def serialize_payload(motion_vectors: torch.Tensor, 
                      block_residuals: torch.Tensor, 
                      plane: torch.Tensor, 
                      i_frame_indices: torch.Tensor, 
                      original_plane_h: int, 
                      original_plane_w: int,
                      padded_plane_h: int,
                      padded_plane_w: int,
                      ) -> bytes:
    """Serialize the payload of the encoded video

    Args:
        motion_vectors (torch.Tensor): The tensor storing the motion vectors for each block. Must be shape (T, L, 2), where L is the number of blocks in each frame
        block_residuals (torch.Tensor): The tensor storing the residuals for each block. Must be shape (T, L, block_size * block_size), where L is the number of blocks in each frame
        plane (torch.Tensor): The unprocessed frame. Will be used to store the i-frames as themselves instead of processed blocks.
        i_frame_indices (torch.Tensor): The indices of the i-frames
        original_plane_h (int): The original height of the plane.
        original_plane_w (int): The original width of the plane.

    Returns:
        bytes: _description_
    """

    # Stores pointers to the numpy arrays containing motion vectors and block residuals.
    payload = []

    # This payload = [] and the payload = b"".join(payload) line at the end might seem strange at first, but it's a neat trick I learned in this project. 
    # Using a bytes() object and appending bytes to it with += causes python to allocate a brand new array in memory with
    # the exact size of the new array and then copy everything to it, which is slow.

    # Using a bytearray() and calling .extend() on it is much better, because bytearrays over-allocate
    # on purpuse and can avoid the slow copies by just using the empty over-allocated segments until
    # they're full. But they still copy the entire data when the bytearray gets full and need to
    # allocate a new one.

    # But if I use a list and call .append() on the bytes() objects created with .tobytes(), so python allocates each of those in memory,
    # but append their memory addresses to the list, not their data itself! So it's not actually copying
    # entire frames worth of data, but a bunch of 8-byte memory addresses into the list, which requires many less memory copies!

    # Then, when I call "".join(payload) at the end, it pre-allocates the EXACT number of bytes needed in memory
    # and finally copies all those byte sequences the pointers point to into the b"" bytes object.

    payload.append(original_plane_h.to_bytes(4,  signed=False))             # uint32 the height of the video
    payload.append(original_plane_w.to_bytes(4,  signed=False))             # uint32 the width of the video
    
    payload.append(padded_plane_h.to_bytes(4,  signed=False))             # uint32 the height of the video
    payload.append(padded_plane_w.to_bytes(4,  signed=False))             # uint32 the width of the video


    num_motion_blocks_per_frame = block_residuals.shape[1]
    payload.append(num_motion_blocks_per_frame.to_bytes(4, signed=False))   # uint32 the number of motion blocks in each video frame

    motion_vectors_np = motion_vectors.cpu().numpy()
    residue_blocks_np = block_residuals.cpu().numpy()

    for frame_idx, frame in enumerate(plane):
        # If the frame is an I-frame, store it as-is
        if frame_idx in i_frame_indices:
            payload.append(frame.to(torch.uint8).cpu().numpy().tobytes())

        # If not, store its motion blocks
        else:
            frame_motion_vectors = motion_vectors_np[frame_idx]
            frame_residue_blocks = residue_blocks_np[frame_idx]

            for blockId in range(num_motion_blocks_per_frame):
                dx, dy  = frame_motion_vectors[blockId]
                residue = frame_residue_blocks[blockId]

                # If there is no movement (dx == 0, dy == 0 and all residues are 0), skip this block. 
                # If there is movement, save it to the payload!
                if (dx == 0) and (dy == 0) and ((residue != 0).any() == False):
                    payload.append(int(1).to_bytes(1, signed=False)) # uint8 signaling to skip the block during decode
                
                else:
                    payload.append(int(0).to_bytes(1, signed=False)) # uint8 signaling to NOT skip the block
                    payload.append(frame_motion_vectors[blockId].tobytes())
                    payload.append(frame_residue_blocks[blockId].tobytes())

    # pre-allocate the exact number of bytes needed and write the full payload to it.
    payload = b"".join(payload)
    return payload





def deserialize_payload(compressed_bytes: bytes, device: torch.device) -> Tuple[List[Dict], Set, str]:
    """Decodes the video's entire byte sequence

    Args:
        compressed_bytes (bytes): The compressed bytes
        device (torch.device): Where to move the motion vectors, block residuals and frames to

    Returns:
        _type_: _description_
    """
    decompressed_bytes = zstandard.decompress(compressed_bytes)
    current_byte = 0

    pixel_format  = decompressed_bytes[current_byte : current_byte+4].decode('ascii')
    current_byte += 4

    quantization_parameter = int.from_bytes(decompressed_bytes[current_byte : current_byte+1], signed=False)
    current_byte          += 1

    datatype_residues_np    = np.int8      if quantization_parameter in [1] else np.uint8
    datatype_residues_torch = torch.int16  if quantization_parameter in [1] else torch.uint8
    num_bytes_per_pixel   = 1

    num_i_frames  = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
    current_byte += 4

    i_frame_set     = set(np.frombuffer(decompressed_bytes[current_byte : current_byte + (num_i_frames*4)], dtype=np.uint32).tolist())
    current_byte   += num_i_frames*4

    block_size   = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
    current_byte += 4

    num_planes    = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
    current_byte += 4

    num_frames    = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
    current_byte += 4

    all_planes_data = []
    for _ in range(num_planes):
        original_plane_h = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
        current_byte    += 4
        original_plane_w = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
        current_byte     += 4

        padded_plane_h = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
        current_byte  += 4
        padded_plane_w = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
        current_byte  += 4

        num_motion_blocks_per_frame = int.from_bytes(decompressed_bytes[current_byte : current_byte+4], signed=False)
        current_byte  += 4

        num_elements_per_motion_block = block_size * block_size

        bytes_per_frame        = padded_plane_h * padded_plane_w * num_bytes_per_pixel
        bytes_per_motion_block = num_elements_per_motion_block * 1 # 1 because 
        
        frames          = torch.zeros((num_frames, padded_plane_h, padded_plane_w),                             dtype=torch.int16)
        residual_blocks = torch.zeros((num_frames, num_motion_blocks_per_frame, num_elements_per_motion_block), dtype=datatype_residues_torch)
        motion_vectors  = torch.zeros((num_frames, num_motion_blocks_per_frame, 2),                             dtype=torch.int8)

        for frame_idx in range(num_frames):
            if frame_idx in i_frame_set:
                reconstructed_i_frame = np.frombuffer(decompressed_bytes[current_byte : current_byte + bytes_per_frame], dtype=np.uint8).copy().reshape((padded_plane_h, padded_plane_w))
                reconstructed_i_frame = torch.as_tensor(reconstructed_i_frame, device=frames.device)
                frames[frame_idx]     = reconstructed_i_frame
                current_byte         += bytes_per_frame

            else:
                for block_id in range(num_motion_blocks_per_frame):
                    skip_flag     = int.from_bytes(decompressed_bytes[current_byte : current_byte + 1], signed=False)
                    current_byte += 1

                    if skip_flag == 0:
                        dx            = int.from_bytes(decompressed_bytes[current_byte : current_byte + 1], signed=True)
                        current_byte += 1
                        dy            = int.from_bytes(decompressed_bytes[current_byte : current_byte + 1], signed=True)
                        current_byte += 1

                        residue       = np.frombuffer(decompressed_bytes[current_byte : current_byte + bytes_per_motion_block], dtype=datatype_residues_np).copy()
                        current_byte += bytes_per_motion_block

                        motion_vectors[frame_idx][block_id]  = torch.as_tensor([dx, dy], dtype=torch.int8)
                        residual_blocks[frame_idx][block_id] = torch.as_tensor(residue,  dtype=datatype_residues_torch)
                    else:
                        motion_vectors[frame_idx][block_id]  = torch.as_tensor([0, 0], dtype=torch.int8)
                        residual_blocks[frame_idx][block_id] = torch.as_tensor([ 0 ] * num_elements_per_motion_block,  dtype=datatype_residues_torch)


        if quantization_parameter in [1]:
            residual_blocks = quantization.dequantize(residual_blocks, quantization_parameter)
        
        frames          = frames.to(device)
        residual_blocks = residual_blocks.to(device)
        motion_vectors  = motion_vectors.to(device)

        all_planes_data.append({
            'frames': frames,                 # I-frame slots filled, P-frame slots zero
            'motion_vectors': motion_vectors,
            'residual_blocks': residual_blocks,
            'original_h': original_plane_h,
            'original_w': original_plane_w,
            'padded_h': padded_plane_h,
            'padded_w': padded_plane_w,
            'block_size': block_size
        })
    
    return all_planes_data, i_frame_set, pixel_format