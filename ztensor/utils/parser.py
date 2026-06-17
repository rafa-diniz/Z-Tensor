import os
import psutil
import torch

from argparse import ArgumentParser, Namespace

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Define the program parameters')

    parser.add_argument('-i', '--input-video', type=str,
                        help="The input video")
    
    parser.add_argument('-n', '--name', type=str,
                        help="The name of the processed video.")
    
    parser.add_argument('-e', '--encode', action='store_true', default=False,
                        help="Encode the input video")
    
    parser.add_argument('-d', '--decode', action='store_true', default=False,
                        help="Decode the input video back into uncompressed video")
    
    parser.add_argument('--test', action='store_true', default=False,
                        help="Run encode/decode and calculate PSNR and SSIM metrics.")
    
    parser.add_argument('-cf', '--compression-factor', type=int, default=16,
                        help="The compression factor for zstandard. Higher values lead to better compression, but increase encode time. Accepted values go from 1 to 20. Default = 16")
    
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help="The number of threads zstandard is allowed to use for compression. Default = 4.")
    
    parser.add_argument('-c', '--chroma', type=str, default='quarter', choices=['full', 'half-width', 'quarter'], help="The level for chroma subsampling. \'full\' is 4:4:4/No chroma subsampling, \'half-width\' is 4:2:2, \'quarter\' is 4:2:0. Default = half-width")

    parser.add_argument('-qp', '--quantization-parameter', type=int, default=0, help="Quantizes the residuals to improve compression ratios by mapping multiple residuals into the same bins. 0 = No Quantization (lossless residuals), 1 = Linear (lossy). Default = 0")

    parser.add_argument('-b', '--block-size', type=int, default=8, help="The size (width and height) of the blocks used for motion estimation. Larger values can make encoding faster, but may slightly increase file size. Default = 8.")

    parser.add_argument('-s', '--search-window', type=int, default=12, help="The size block matching's search window. Larger values can improve compression gains at the cost of more compute and memory. Default = 12.")

    parser.add_argument('-mem', type=str, default='4G',
                        help="The amount of memory that the motion estimation algorithm is allowed to use. Use G for GB and M for MB. The codec respects this memory limit regardless of the value for \'-device\'. Default = 4G")

    parser.add_argument('-device', type=str, default='0',
                        help="Use \"cpu\" to run on the CPU and numbers to select which GPU to use. Default = cuda:0")
    

    return parser


def validate_args(args: Namespace) -> None:

    if sum([args.encode, args.decode, args.test]) != 1:
        raise ValueError(f"Choose exactly one operation: --encode, --decode, or --test.")
    
    if not (1 <= args.compression_factor <= 20):
        raise ValueError(f"The compression factor has to be between 1 and 20.")

    if args.encode or args.decode:
        if not args.input_video:
            raise ValueError(f"Argument -i/--input-video is required for encoding or decoding")
        if not args.name:
            raise ValueError(f"Argument -n/--name is required for encoding or decoding")
    
    if args.input_video and not os.path.isfile(args.input_video):
        raise ValueError(f"Input video \'{args.input_video}\' does not exist!")

    if args.threads > os.cpu_count():
        args.threads = -1 # -1 means use all threads.
    elif args.threads < 1:
        args.threads = 1


    args.device = check_device(args.device)
    max_mem     = get_max_mem(args.device)

    mem_str = args.mem.strip().upper()
    try:
        if mem_str.endswith('G'):
            mem_bytes = float(mem_str[ : -1]) * (1024**3) 
        elif mem_str.endswith('M'):
            mem_bytes = float(mem_str[ : -1]) * (1024**2)
        else:
            raise ValueError(f"Invalid format: \'{args.mem}\'. Use \'G\' or \'M\' (example: \'2G\', \'500M\')")
    except ValueError:
        raise ValueError(f"Could not parse memory amount from \'{args.mem}\'")
    
    if mem_bytes > 0.7 * max_mem:
        print(f"Requested memory amount leaves almost no headroom for your PC. Capping the usage at 70% of the maximum memory available")
        mem_bytes = 0.7 * max_mem

    args.mem = int(mem_bytes)


def check_device(device: str) -> torch.device:
    try:
        validated_device = torch.device(device) if not device.isdigit() else torch.device(f"cuda:{device}")
    
        if validated_device.type == 'cuda' and not torch.cuda.is_available():
            print(f"No CUDA-enabled GPU detect. Falling back to CPU...")
            validated_device = torch.device('cpu')
        
    except Exception:
        print(f"Error trying to access device \'{device}\'. Falling back to CPU...")
        validated_device = torch.device('cpu')

    return validated_device


def get_max_mem(device: torch.device) -> int:
    if device == torch.device('cpu'):
        total_memory_bytes = psutil.virtual_memory().total
    else:
        total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    
    return int(total_memory_bytes)