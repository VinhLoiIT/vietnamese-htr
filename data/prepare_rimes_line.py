import pandas as pd
import os
import argparse
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from PIL import Image


def convert_df(xmlfile: str):
    results = []
    with open(xmlfile, 'r') as f:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        for page in root.findall('.//SinglePage'):
            filename = os.path.join(os.path.dirname(xmlfile), page.attrib['FileName'])
            lines = page.findall('.//Line')
            results.extend([{
                'filename': filename,
                'bottom': int(line.attrib['Bottom']),
                'top': int(line.attrib['Top']),
                'right': int(line.attrib['Right']),
                'left': int(line.attrib['Left']),
                'label': line.attrib['Value'],
            } for line in lines])
    df = pd.DataFrame(results)
    return df

def extract_image(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    new_df = []
    for idx, row in df.iterrows():
        filename = row['filename']
        base_name, ext = os.path.splitext(os.path.basename(filename))
        base_name = os.path.join(output_dir, base_name)

        image: Image.Image = Image.open(filename)
        bbox = (row['left'], row['top'], row['right'], row['bottom'])
        crop_path = os.path.join(f'{base_name}_{idx}{ext}')
        crop_image = image.crop(bbox)
        crop_image.save(crop_path)
        new_df.append((os.path.basename(crop_path), row['label']))
    return pd.DataFrame(new_df, columns=['filename', 'label'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='RIMES folder')
    args = parser.parse_args()

    train_df = convert_df(os.path.join(args.dir, 'training_2011.patched.xml'))
    test_df = convert_df(os.path.join(args.dir, 'eval_2011_annotated.xml'))
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=0)

    print('Extracting train partition...')
    train_df = extract_image(train_df, os.path.join(args.dir, 'train'))
    print('Extracting validation partition...')
    val_df = extract_image(val_df, os.path.join(args.dir, 'val'))
    print('Extracting test partition...')
    test_df = extract_image(test_df, os.path.join(args.dir, 'test'))

    print('Writing csv files...')
    train_df.to_csv(os.path.join(args.dir, 'train.csv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(args.dir, 'val.csv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(args.dir, 'test.csv'), sep='\t', index=False)
    print('Done')
