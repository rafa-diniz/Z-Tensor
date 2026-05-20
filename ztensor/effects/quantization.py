import torch

def quantize(plane: torch.Tensor, quantization_parameter: int):
    plane = plane.float()

    # Linear quantization
    if quantization_parameter == 1:
        plane = plane / 2
        plane = plane.to(torch.int8)
    

    return plane


def dequantize(plane: torch.Tensor, quantization_parameter: int):
    plane = plane.float()
    
    # Linear quantization
    if quantization_parameter == 1:
        plane = plane * 2
        plane = plane.to(torch.int16)


    return plane