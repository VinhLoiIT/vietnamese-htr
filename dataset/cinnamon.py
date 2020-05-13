from .vocab import Vocab
import pandas as pd

class CinnamonVocab(Vocab):
    def __init__(self, train_csv: str, add_blank):
        self._train_csv = train_csv
        super().__init__(add_blank)

    def load_labels(self) -> pd.Series:
        '''
        Load labels from train partition
        '''
        df = pd.read_csv(self._train_csv, sep='\t', keep_default_na=False)
        labels = df['label']
        return labels
