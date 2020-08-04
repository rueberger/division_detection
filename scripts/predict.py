""" A script for running predictions
"""
import argparse
import logging
import sys

from division_detection.predict import predict_from_inbox


def main():
    logger = logging.getLogger("division_detection")
    logger.setLevel(logging.DEBUG)

    # define file handler for module
    file_handler = logging.FileHandler('/var/log/division_detection.log')
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.ERROR)

    # create formatter and add to handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    parser = argparse.ArgumentParser(description="Runs division detector")

    data_path_help = ("Absolute path to directory containing data klbs\n"
                      "Example: '/data/ds1' where '/data/ds1/' contains a directory "
                      "named 'klb' containing the raw images")
    parser.add_argument('data_dir', help=data_path_help)


    model_name_help = ("Name of model to use. "
                       "Unless you've trained your own model, leave it set to the default value. ")
    parser.add_argument('--model_name', help=model_name_help, default='pretrained')

    chunk_size_help = ("Size of chunk to use when processing the images. \n"
                       "Due to GPU memory limitations, images are processed in a chunk-wise fashion"
                       "Chunk size should be set to the largest chunk that will fit in GPU memory")
    # TODO: set default to largest for 11GB GPUs
    parser.add_argument('--chunk_size', help=chunk_size_help, type=int, nargs=3,
                        default=[200, 100, 100])

    allowed_gpus_help = ("The ids of the GPUs to use for this job. \n"
                         "Same numbering scheme as used for the CUDA_VISIBLE_DEVICES"
                         "environment variable. \n"
                         "Job will be parallelized across available GPUs"
                         "Defaults to 0")
    parser.add_argument('--allowed_gpus', help=allowed_gpus_help, type=int, nargs='+', default=[0])

    args = parser.parse_args()

    predict_from_inbox(args.model_name, args.data_dir, args.chunk_size, args.allowed_gpus)


if __name__ == '__main__':
    main()
