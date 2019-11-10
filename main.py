import argparse
import torch
import json
import os

from models.model import Model
from train import train, evaluate

from torch.utils.data import DataLoader
from dataset import VNOnDB, to_batch
from utils import get_vietnamese_alphabets

default_config = {
  "decoder": "Bahdanau",
  "n_channels": 4,
  "decoder_layers": 1,
  "decoder_dropout": 0.1,
  "decoder_size": 256,
  "batch_size": 64,
  "sampling_prob": 0.0,
  "attention_score": "bahdanau",
  "n_epochs_decrease_lr": 15,
  "start_learning_rate": 0.00000001,
  "end_learning_rate": 0.00000000001,
  "gpu": True,
  "loss": "cross_entropy",
  "dense_depth": 4,
  "n_blocks": 3,
  "growth_rate": 96,
}

def run():

    config_path = os.path.join("models", args.config)
    config = default_config
    
    with open(config_path, "r") as f:
        config = json.load(f)
    

    config["gpu"] = torch.cuda.is_available()
    batch_size = config["batch_size"]

    train_data = VNOnDB('./data/VNOnDB/word_train', './data/VNOnDB/train_word.csv')
    validation_data = VNOnDB('./data/VNOnDB/word_val', './data/VNOnDB/validation_word.csv')
    test_data = VNOnDB('./data/VNOnDB/word_test', './data/VNOnDB/test_word.csv')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=to_batch)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=to_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=to_batch)

    # Models
    model = Model(config)

    if config["gpu"]:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    print("=" * 60)
    print(model)
    print("=" * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(" (" + k + ") : " + str(v))
    print()
    print("=" * 60)

    # print("\nInitializing weights...")
    # for name, param in model.named_parameters():
    #     if 'bias' in name:
    #         torch.nn.init.constant_(param, 0.0)
    #     elif 'weight' in name:
    #         torch.nn.init.xavier_normal_(param)

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer = train(model, optimizer, train_loader, run_state)
        evaluate(model, eval_loader)

        # TODO implement save models function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    args, _ = parser.parse_known_args()
    run()



