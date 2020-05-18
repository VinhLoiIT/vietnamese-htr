import pandas as pd
from .vocab import Vocab


class RIMESVocab(Vocab):
    def __init__(self, train_csv: str, add_blank: bool):
        self.__train_csv = train_csv
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        train_df = pd.read_csv(self.__train_csv, sep='\t')
        return train_df['label'].astype(str)
