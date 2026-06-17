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
                                            args.quantization_parameter,
                                            args.block_size,
                                            args.search_window
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
        test_codec_fidelity(args)