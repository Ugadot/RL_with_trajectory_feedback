import glob
from PIL import Image
import h5py
import numpy as np
import argparse

def main(args):
    images_h5_path = args.images_path
    out_gif_path = args.output
    win_size = int(args.win_size)
    upsample_factor = args.upsample
    curr_images = []
    try:
      data = h5py.File(images_h5_path, 'r')
    except Execption as e:
      print (repr(e))
      exit(1)

    images = data['images']
    
    print("Found " + str(len(images)) + " images")
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    for image in images:
      PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
      PIL_image = PIL_image.resize((upsample_factor * PIL_image.size[0], upsample_factor * PIL_image.size[1]), Image.NEAREST)
      curr_images.append(PIL_image)
    img = curr_images[0]
    img.save(fp=out_gif_path, format='GIF', append_images=curr_images[1:],
    save_all=True, duration=200, loop=0)
    print("Saved GIF")
    exit(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_path', default='', help='Path to h5 file with images')
    parser.add_argument(
        '--output', default='', help='Path output gif file')
    parser.add_argument(
        '--win_size', default=10, help='Windows size that used for creating this data')
    parser.add_argument(
        '--upsample', default=10, help='Upsample factoer (default=10)')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    main(args)