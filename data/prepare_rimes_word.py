import pandas as pd
import os

def read_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=' ', header=None, names=['path', 'label'])


train_df = read_df('RIMES/groundtruth_training_icdar2011.txt')
train_df = train_df.drop(index=0) # file not found

print('Export train.csv')
train_df.to_csv('RIMES/train.csv', sep='\t', index=None)
print('Done')

val_df = read_df('RIMES/ground_truth_validation_icdar2011.txt')
print('Export validation.csv')
val_df.to_csv('RIMES/validation.csv', sep='\t', index=None)
print('Done')

test_df = read_df('RIMES/grount_truth_test_icdar2011.txt')
print('Export test.csv')
test_df.to_csv('RIMES/test.csv', sep='\t', index=None)
print('Done')
