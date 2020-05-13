import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='VNOnDB folder')
    parser.add_argument('level', type=str, choices=['word', 'line'], help='VNOnDB level')
    args = parser.parse_args()

    for partition in ['train', 'validation', 'test']:
        csv_path = os.path.join(args.dir, f'{partition}_{args.level}.csv')
        df = pd.read_csv(csv_path,
                         sep='\t',
                         index_col=0,
                         keep_default_na=False)
        df['id'] = df['id'].apply(lambda id: id+'.png' if os.path.splitext(id)[-1] != '.png' else id)
        df.to_csv(os.path.join(args.dir, f'{partition}_{args.level}_new.csv'), sep='\t', index=False)
    print('Done')
