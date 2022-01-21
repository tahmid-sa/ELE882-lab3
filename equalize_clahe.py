import argparse

from skimage import io
from skimage.exposure import equalize_adapthist


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument("--kernel", type=int, help="The kernel size.")
    parser.add_argument("--clip", type=float, help="The clip limit.")

    return parser.parse_args()

def perform_clahe(input_image, output_image, kernel, clip):
   
    img = io.imread(input_image, as_gray=True)
    img = equalize_adapthist(img, kernel, clip, 256)
    io.imsave(output_image, img)

def main():

    kernel = 3
    clip = 0.01
    args = get_args()

    print(f'Input filename: {args.input_image}')
    input_image = f'{args.input_image}'

    print(f'Output filename: {args.output_image}')
    output_image = f'{args.output_image}'

    if args.kernel:
        print(f'Kernel size: {args.kernel}')
    if args.clip:
        print(f'Clip limit: {args.clip}')

    perform_clahe(input_image, output_image, kernel, clip)

if __name__ == "__main__":
    main()