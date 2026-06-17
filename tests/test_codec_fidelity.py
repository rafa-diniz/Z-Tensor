import os
import time
import torch
import numpy as np

from skimage.metrics import structural_similarity as ssim

from ztensor.pipeline import pipeline

def run_fidelity_check(video_path, device, memory_budget, compression_factor, num_threads, chroma_subsampling, quantization_parameter, block_size, search_window):
    
    start = time.perf_counter()
    original_video, encoded_video = pipeline.encode_pipeline(   video_path, 
                                                                device, 
                                                                memory_budget, 
                                                                compression_factor, 
                                                                num_threads,
                                                                chroma_subsampling,
                                                                quantization_parameter,
                                                                block_size,
                                                                search_window
                                                                )
    encode_time = time.perf_counter() - start

    original_video = original_video.cpu().numpy().astype(np.uint8)

    # Wait until the original video goes to the CPU.
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    start          = time.perf_counter()
    decoded_video  = pipeline.decode_pipeline(encoded_video, device)
    decode_time    = time.perf_counter() - start

    decoded_video  = decoded_video.cpu().numpy().astype(np.uint8)

    # Check if the decoded frames are pixel-exact matches to the original video
    if np.array_equal(decoded_video, original_video): 
        avg_psnr = "Lossless"
        avg_ssim = "Lossless"
    else:
        mse      = np.mean((original_video.astype(np.float64) - decoded_video.astype(np.float64)) ** 2)
        avg_psnr = float('inf') if mse == 0 else 10 * np.log10(255**2 / mse)

        ssim_values = []
        for i in range(len(original_video)):
            s = ssim(original_video[i], decoded_video[i], channel_axis=2, data_range=255)
            ssim_values.append(s)

        avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim, os.path.getsize(video_path) / (1024 * 1024), len(encoded_video) / (1024 * 1024), encode_time, decode_time


def test_codec_fidelity(args):

    if args.input_video:
        videos = [ args.input_video ]
    else:
        test_dir = "./test_videos/"
        videos   = sorted([ os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.avi'))])


    print("Calculating PSNR/SSIM scores")
    print("-"*50)
    print("PSNR: 0.0 to inf, with infinite being a perfect score.\nSSIM: 0.0 to 1.0, with 1.0 being a perfect score", end="\n\n")
    print("Quality Reference:\nPSNR >= 40 dB, SSIM >= 0.95: Excelent Fidelity\nPSNR >= 30 dB, SSIM >= 0.90: Good Fidelity\n")

    print(f"{'Video Source':<20} | {'PSNR score (dB)':<20} | {'SSIM score':<20} | {'Size Raw Video (MB)':<20} | {'Size Z-Tensor Encoded (MB)':<30} | {'Encode Time (s)':<20} | {'Decode Time (s)':<20}")
    print("-" * 170)


    for video_path in videos:
        avg_psnr, avg_ssim, size_orig_mb, size_ztensor_mb, encode_time, decode_time = run_fidelity_check(video_path, 
                                                                               args.device, 
                                                                               args.mem, 
                                                                               args.compression_factor, 
                                                                               args.threads,
                                                                               args.chroma,
                                                                               args.quantization_parameter,
                                                                               args.block_size,
                                                                               args.search_window
                                                                               )

        if avg_psnr == "Lossless" and avg_ssim == "Lossless":
            print(f"{os.path.basename(video_path):<20} | {'Lossless':<20} | {'Lossless':<20}  | {size_orig_mb:<20.1f} | {size_ztensor_mb:<30.1f} | {encode_time:20.1f} | {decode_time:20.1f}")
        else:
            print(f"{os.path.basename(video_path):<20} | {avg_psnr:<20.3f} | {avg_ssim:<20.3f} | {size_orig_mb:<20.1f} | {size_ztensor_mb:<30.1f} | {encode_time:20.1f} | {decode_time:20.1f}")
