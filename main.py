from ztensor.utils import parser, video
from ztensor.pipeline import pipeline

from tests.test_codec_fidelity import test_codec_fidelity


if __name__ == '__main__':

    args = parser.make_parser().parse_args()
    parser.validate_args(args)

    if args.encode:
        print("Encoding video...")
        _, encoded_video = pipeline.encode_pipeline( args.input_video, 
                                            args.device, 
                                            args.mem, 
                                            args.compression_factor, 
                                            args.threads,
                                            args.chroma,
                                            args.quantization_parameter
                                            )
        
        # Writing encoded video
        with open(f"{args.name}.ztensor", "wb") as f:
            f.write(encoded_video)

        print(f"Encoded video saved as {args.name}.ztensor")


    elif args.decode:
        print("Decoding video...")

        with open(args.input_video, "rb") as f:
            bytes_data = f.read()

        decoded_video = pipeline.decode_pipeline(bytes_data, args.device)
        decoded_video = decoded_video.cpu().numpy()

        video.write_video(decoded_video, args.name)

        print(f"Decoded file saved as {args.name}.avi!")


    elif args.test:
        print("Calculating PSNR/SSIM scores")
        print("-"*50)
        print("PSNR: 0.0 to inf, with infinite being a perfect score.\nSSIM: 0.0 to 1.0, with 1.0 being a perfect score", end="\n\n")
        print("Quality Reference:\nPSNR >= 40 dB, SSIM >= 0.95: Unnoticeable/Excelent Fidelity\nPSNR >= 30 dB, SSIM >= 0.90: Good Fidelity\n")

        test_codec_fidelity(args)