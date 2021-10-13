import os
import sys
import argparse
import h5py
import numpy as np

master_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(1, master_dir)
from src.utils.user_interface import saved_data_render


def show_files_performance(files):
    for file_path in files:
        f = h5py.File(file_path, 'r')
        if ('images' not in f.keys()):
            print(f"No images in file {file_path}")
        else:
            images = np.asarray(f['images'])
            saved_data_render(images)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--file', default=[], type=str, nargs="*",
                             help="path to files of data saved in order to present")
    parsed_args = args_parser.parse_args()
    show_files_performance(parsed_args.file)
