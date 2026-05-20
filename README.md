# Z-Tensor

## My Custom-Made Hardware-Accelerated Video Codec!

Z-Tensor is a GPU-accelerated video codec that encodes raw videos into a compact `.ztensor` file using scene-aware keyframes, chroma subsampling, frame differencing, optional quantization, and Zstandard entropy coding. And then decodes it back! Lossless and lossy modes are both supported, and the user can choose between using the CPU or GPU.

The library gets its name from its two main components: **Z** comes from Zstandard, which is used as the entropy coder, and **Tensor** because every pixel-level operation in the pipeline runs as a PyTorch tensor op on the GPU.

Z-Tensor is not a wrapper around FFmpeg or any other existing codec :) Every step in both the encoder and decoder are implemented from scratch using just tensor operations.

---

## What are the features?

- **End-to-end GPU pipeline.** BGR→YUV, blur, Sobel, chroma subsampling, frame differencing, and histogram computation are all PyTorch ops. The CPU only handles the final Zstandard pass, since Zstandard doesn't run on GPU. This means the encoding runs fast on the GPU and only uses the CPU when needed.
- **Scene-aware keyframes.** Instead of placing an I-frame every N frames, Z-Tensor finds scene boundaries by looking at how much the grayscale histogram changes between consecutive frames. Within each scene, it picks the frame with the highest edge content as the keyframe, so the I-frame is the sharpest possible reference for the P-frames that follow.
- **Frame differencing.** P-frames are stored as the deltas against the previous frame, and the decoder reconstructs them with a cumulative sum within each scene!
- **Chroma subsampling.** All three canonical modes are implemented: full (4:4:4), half-width (4:2:2), and quarter (4:2:0). The U and V planes are downsampled with 2D average pooling on the GPU.
- **Optional residual quantization.** A linear mode divides the frame deltas by 2 and stores them as `int8`, trading a very small fidelity hit for a measurable reduction in file size.
- **VRAM budget.** You tell Z-Tensor how much memory it can use and the histogram stage batches automatically to stay under that limit.
- **CPU or GPU.** Same code path: pass `-device cpu` or a CUDA index and it will use that device!

---

## How does the encode pipeline work?

1. Read the video into a `(frames, H, W, 3)` BGR tensor on the chosen device.
2. Convert to grayscale and apply a 3×3 box blur to make the histograms more robust to noise.
3. Compute a 256-bin intensity histogram per frame, batched to respect the VRAM budget.
4. Find scene boundaries: compare consecutive histograms and find frames where the histogram change is both a local peak and exceeds `mean + 1σ` of all deltas to find sudden scene changes and treat them as cuts in the video.
5. Within each scene selected above, run Sobel edge detection and pick the frame with the highest edge variance as the I-frame.
6. Optionally convert to YUV for chroma subsampling.
7. For every non-I-frame, replace it with `frame - previous_frame`.
8. Optionally quantize the frame deltas.
9. Pack a small header (pixel format, quantization mode, plane count, I-frame indices, plane shapes) followed by the plane bytes, and compress the whole byte array with Zstandard.

The decoder reverses this: Zstandard decompress → parse header → per-scene cumulative sum to reconstruct P-frames → interpolate the subsampled chroma planes if chroma subsampling was used, → YUV→BGR → clip to `uint8` -> save as a watchable video!

---

## Installation

```bash
git clone https://github.com/RafaelAmauri/Z-Tensor.git
cd Z-Tensor
pip install -r requirements.txt
```

---

## Usage

### Encode

```bash
# Balanced Lossy: 4:2:2 chroma + no quantization. Visually indistinguishable from the original.
python main.py -i test_videos/bowing_cif.avi -n bowing_out -e --chroma half-width -qp 0 -device 0

# Aggressive Lossy: 4:2:0 chroma + no quantization. Slightly lower quality than Balanced Lossy, still excellent fidelity.
python main.py -i test_videos/bowing_cif.avi -n bowing_out -e --chroma quarter -qp 0 -device 0

# Most Aggressive Lossy: 4:2:0 chroma + linear quantization. Smallest file, some fidelity loss
python main.py -i test_videos/bus_cif.avi -n bus_out -e --chroma quarter -qp 1 -device 0

# Lossless: no chroma subsampling + no quantization.
python main.py -i test_videos/bowing_cif.avi -n bowing_out -e --chroma full -qp 0 -device 0

# For low-VRAM GPUs: uses 1GB of RAM
python main.py -i test_videos/bowing_cif.avi -n bowing_out -e -mem 1G -device 0

# CPU mode: 16 threads, 3 GB RAM
python main.py -i test_videos/carphone_qcif.avi -n carphone_out -e -cf 18 -device cpu --threads 16 -mem 3G
```

### Decode

```bash
# The file header in the .ztensor format automatically detects the settings that should be used. Just run:
python main.py -i out.ztensor -n decoded -d
```

### Quality test

Runs the automatic quality test on every `.avi` in `./test_videos/` and prints PSNR, SSIM, and file size before/after:

```bash
# with default config (uses Balanced Lossy)
python main.py --test

# with Aggressive Lossy 
python main.py --test --chroma quarter -qp 1

# with Lossless 
python main.py --test --chroma full -qp 0
```

## Results

#### Some quick tests on standard CIF/QCIF benchmark sequences

#### Uses Balanced Lossy mode: `--chroma half-width -qp 0`

| Video | PSNR (dB) | SSIM | Original Filesize | Z-Tensor Filesize | Compression Ratio |
| --- | --- | --- | --- | --- | --- |
| carphone_qcif.avi | 42.96 | 1.00 | 29 MB | 7 MB | 4.1× |
| bus_cif.avi | 41.29 | 1.00 | 45 MB | 16 MB | 2.8× |
| bowing_cif.avi | 45.63 | 1.00 | 91 MB | 17 MB | 5.4× |

#### Lossless mode (`--chroma full -qp 0`)

| Video |PSNR (dB) | SSIM | Original Filesize | Z-Tensor Filesize | Compression Ratio |
| --- | --- | --- | --- | --- | --- |
| carphone_qcif.avi | Lossless | 1.00 | 29 MB | 17 MB | 1.7× |
| bus_cif.avi | Lossless | 1.00 | 45 MB | 37 MB | 1.2× |
| bowing_cif.avi | Lossless | 1.00 | 91 MB | 41 MB | 2.2× |

Quality reference:

- PSNR ≥ 40 dB / SSIM ≥ 0.95: excellent fidelity, visually indistinguishable
- PSNR ≥ 30 dB / SSIM ≥ 0.90: good fidelity

---

## Flags

| Flag | What it does |
| --- | --- |
| `-i / --input-video` | Path to the input video |
| `-n / --name` | Output name (no extension) |
| `-e / --encode` | Encode mode |
| `-d / --decode` | Decode mode |
| `--test` | Run PSNR/SSIM against the test set |
| `-cf` | Zstandard compression level, 1–20 (default 16) |
| `-t / --threads` | Threads for Zstandard (default 4) |
| `-c / --chroma` | `full`, `half-width`, or `quarter` (default `quarter`) |
| `-qp` | `0` lossless residuals, `1` linear quantization |
| `-mem` | Memory budget, e.g. `2G`, `500M` (default `2G`) |
| `-device` | `cpu` or a CUDA index like `0` (default `0`) |

---

## Some neat things that I might implement in the future:

* Motion estimation via block-matching
* Discrete Cosine Transforms
* Adaptive Quantization