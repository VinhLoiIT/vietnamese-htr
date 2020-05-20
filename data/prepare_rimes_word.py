import pandas as pd
import os

def read_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=' ', header=None, names=['path', 'label'])


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RIMES')
print(data_path)

train_df = read_df(os.path.join(data_path, 'groundtruth_training_icdar2011.txt'))
train_df = train_df.drop(index=0) # file not found

print('Export train.csv')
train_df.to_csv(os.path.join(data_path, 'train.csv'), sep='\t', index=None)
print('Done')

val_df = read_df(os.path.join(data_path, 'ground_truth_validation_icdar2011.txt'))
print('Export validation.csv')
val_df.to_csv(os.path.join(data_path, 'validation.csv'), sep='\t', index=None)
print('Done')

test_df = read_df(os.path.join(data_path, 'grount_truth_test_icdar2011.txt'))
print('Export test.csv')
test_df.to_csv(os.path.join(data_path, 'test.csv'), sep='\t', index=None)
print('Done')
