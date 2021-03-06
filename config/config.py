import os
from pprint import pformat
from typing import Dict, Union

from torch import optim
from model.feature_extractor import *
from dataset import IAM, RIMESVocab, VNOnDBVocab, VNOnDBVocabFlatten, CinnamonVocab, HTRDataset

MAPPING_NAME = {

    # cnn
    'densenet': DenseNetFE,
    'squeezenet': SqueezeNetFE,
    'efficientnet': EfficientNetFE,
    'custom': CustomFE,
    'resnet': ResnetFE,
    'resnext': ResnextFE,
    'deformresnet': DeformResnetFE,
    'vgg': VGGFE,

    # dataset
    'vnondb': HTRDataset,
    'vnondb_line': HTRDataset,
    'rimes': HTRDataset,
    'rimes_line': HTRDataset,
    'cinnamon': HTRDataset,

    # vocab
    'vnondb_vocab': VNOnDBVocab,
    'vnondb_vocab_flatten': VNOnDBVocabFlatten,
    'vnondb_line_vocab': VNOnDBVocab,
    'vnondb_line_vocab_flatten': VNOnDBVocabFlatten,
    'cinnamon_vocab': CinnamonVocab,
    'rimes_vocab': RIMESVocab,
    'rimes_line_vocab': RIMESVocab,
}

def initialize(config: Dict, *args, **kwargs):
    assert config.get('name', None) in MAPPING_NAME.keys()
    config_args = config.get('args', {}) or {}
    params = {**config_args, **kwargs }
    obj = MAPPING_NAME[config['name']](*args, **params)
    return obj


class Config(object):
    def __init__(self, config: Union[str, Dict], config_dir: str = None, **kwargs):
        if isinstance(config, Dict):
            self.config = config
        elif isinstance(config, str):
            self.config_dir = config_dir
            if self.config_dir is None:
                self.config_dir = os.path.dirname(config)
            defaults = self.from_yaml(config)['defaults']
            override = kwargs
            self._recursive_load_config(defaults)
            self.config = {**defaults, **override}
        else:
            raise ValueError('config should be str or dict')

    def from_yaml(self, yaml_file: str):
        import yaml
        with open(yaml_file, 'r') as stream:
            config = yaml.safe_load(stream)
            return config

    def _recursive_load_config(self, config: Dict):
        for key, value in config.items():
            sub_config_dir = os.path.join(self.config_dir, key)
            if os.path.exists(sub_config_dir):
                sub_config_file = os.path.join(sub_config_dir, value)
                try:
                    sub_config = self.from_yaml(f'{sub_config_file}.yaml')
                    config.update(sub_config)
                except FileNotFoundError as e:
                    print(e)
                    exit(-1)

    def save(self, path:str):
        with open(path, 'w') as file:
            yaml.dump(path, file)

    def get(self, key, default_val):
        if key in self.config.keys():
            return self.config[key]
        else:
            return default_val

    def __getitem__(self, key):
        return self.config[key]

    def __str__(self):
        return pformat(self.config, width=-1, indent=2)
