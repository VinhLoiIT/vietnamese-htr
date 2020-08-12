# vietnamese-htr
Vietnamese handwritten text recognition system

# Install Library
```
conda env create -f environment.yml
```

# Run
## Train

**Note**: Default `--max_epochs` of Pytorch Lightning is `100` and **not use GPU**, then 2 following option might always be used

To train Transformer (see config params at `model/model_tf.py`):
```bash
python train.py tf config/base.yaml --gpus -1 --max_epochs 50 --deterministic True
```

To train RNN (see config params at `model/model_rnn.py`):
```bash
python train.py rnn config/base.yaml --gpus -1 --max_epochs 50 --deterministic True
```

Example train TF
```
python train.py tf config/base.yaml --gpus -1 --max_epochs 50 --deterministic True --attn_size 512 --dim_feedforward 4096 --encoder_nlayers 2 --decoder_nlayers 2 --seed 9498 --decoder_nlayers 2 --stn --pe_text --pe_image
```


See Pytorch Lightning Trainer config at: [Pytorch Lightning Doc](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags). Some useful flags:
```
--fast_dev_run True                             # Run 1 batch on train, 1 batch on val to debug
--profiler True                                 # Log time (might slow)
--max_epochs 50                                 # Train for 50 epochs
--resume_from_checkpoint [PATH_TO_CKPT_FILE]    # Resume training from checkpoint
--deterministic True                            # Reproducible
```

## Test
```
python test.py {tf, rnn} CKPT_FILE
```

## Visualization

Run jupyter:
```
jupyter lab
```

Open respective notebooks for further visualization


# Code references
- [Show, Attend, and Tell | a PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
- [Image Captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
- [DeepSpeech](https://github.com/SeanNaren/deepspeech.pytorch)
- [Seq2Seq-Pytorch](https://github.com/b-etienne/Seq2seq-PyTorch)
