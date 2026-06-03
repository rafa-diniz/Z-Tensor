import os
import torch
import numpy as np

from skimage.metrics import structural_similarity as ssim

from ztensor.pipeline import pipeline

def run_fidelity_check(video_path, device, memory_budget, compression_factor, num_threads, chroma_subsampling, quantization_parameter):
    original_video, encoded_video = pipeline.encode_pipeline(   video_path, 
                                                                device, 
                                                                memory_budget, 
                                                                compression_factor, 
                                                                num_threads,
                                                                chroma_subsampling,
                                                                quantization_parameter
                                                                )
    

    original_video = original_video.cpu().numpy().astype(np.uint8)

    # Wait until the original video goes to the CPU.
    with torch.cuda.device(device):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    decoded_video = pipeline.decode_pipeline(encoded_video, device)
    
    decoded_video  = decoded_video.cpu().numpy().astype(np.uint8)

    mse = np.mean((original_video.astype(np.float64) - decoded_video.astype(np.float64)) ** 2)
    avg_psnr = float('inf') if mse == 0 else 10 * np.log10(255**2 / mse)

    ssim_values = []
    for i in range(len(original_video)):
        s = ssim(original_video[i], decoded_video[i], channel_axis=2, data_range=255)
        ssim_values.append(s)

    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim, os.path.getsize(video_path) / (1024 * 1024), len(encoded_video) / (1024 * 1024)


def test_codec_fidelity(args):

    if args.input_video:
        videos = [ args.input_video ]
    else:
        test_dir = "./test_videos/"
        videos   = sorted([ os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.avi'))])


    print(f"{'Video Source':<30} | {'PSNR score (dB)':<20} | {'SSIM score':<20} | {'Size Original (MB)':<20} | {'Size Z-Tensor Encoded (MB)':<20}")
    print("-" * 130)


    for video_path in videos:

        avg_psnr, avg_ssim, size_orig_mb, size_ztensor_mb = run_fidelity_check(video_path, 
                                                                               args.device, 
                                                                               args.mem, 
                                                                               args.compression_factor, 
                                                                               args.threads,
                                                                               args.chroma,
                                                                               args.quantization_parameter
                                                                               )

        if np.isinf(avg_psnr):
            print(f"{os.path.basename(video_path):<30} | {'Lossless':<20}  | {avg_ssim:<20.2f} | {size_orig_mb:<20.1f} | {size_ztensor_mb:<20.1f}")
        else:
            print(f"{os.path.basename(video_path):<30} | {avg_psnr:<20.2f} | {avg_ssim:<20.2f} | {size_orig_mb:<20.1f} | {size_ztensor_mb:<20.1f}")
