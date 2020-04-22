import torch
import argparse
from system import CTCInference, CEInference
from PIL import Image

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device = {}'.format(device))
    # inferencer = CTCInference(args.checkpoint, device)
    inferencer = CEInference(args.checkpoint, device)
    while True:
        path = input('Enter image path: ')
        if path == '':
            print('Exit')
            break
        image = Image.open(path).convert('L')
        
        # NOTE: pad to a large enough image
        padded_image = Image.new('L',
                        (max(image.size[0], 650), image.size[1]),   # W, H
                        (255,))  # White
        padded_image.paste(image, image.getbbox())  # Not centered, top-left corner
        strings = inferencer.inference([padded_image]) # because we just test 1 image
        print(strings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint of model')
    parser.add_argument('--beamsearch', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
