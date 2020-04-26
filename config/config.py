import os
from pprint import pformat
from typing import Dict, Union

from torch import optim
from model import *
from dataset import IAM, VNOnDB, RIMES, RIMESLine, Cinnamon

MAPPING_NAME = {
    # optimizers
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,

    # cnn
    'densenet': DenseNetFE,
    'squeezenet': SqueezeNetFE,
    'efficientnet': EfficientNetFE,
    'custom': CustomFE,
    'resnet': ResnetFE,
    'resnext': ResnextFE,
    'deformresnet': DeformResnetFE,

    # lr_scheduler:
    'plateau': optim.lr_scheduler.ReduceLROnPlateau,

    # model
    'tf': ModelTF,
    'tf_a2d': ModelTFA2D,
    'rnn': ModelRNN,

    # model
    'ctc_tf_encoder': CTCModelTFEncoder,
    'ctc_tf_full': CTCModelTF,
    'ctc_rnn': CTCModelRNN,

    # dataset
    'vnondb': VNOnDB,
    'vnondb_line': VNOnDB,
    'rimes': RIMES,
    'rimes_line': RIMESLine,
    'cinnamon': Cinnamon,

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

    def __getitem__(self, key):
        return self.config[key]

    def __str__(self):
        return pformat(self.config, width=-1, indent=2)
