import os
import argparse
import csv
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type='text')
    parser.add_argument('output_csv', type='text', default=None)
    args = parser.parse_args()

    images = os.listdir(args.dir)
    with open(args.output_csv, 'wt') as f:
        writer = csv.writer(f) if f else None
        writer.writerow(['path', 'w', 'h'])
        for image in images:
            image = os.path.join(path, image)
            w, h = Image.open(image).size
            writer.writerow([image, w, h])

    print('Done')
