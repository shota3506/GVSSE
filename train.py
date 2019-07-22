import argparse
import os
import configparser
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import datasets
import models
from build_vocab import Vocab


config = configparser.ConfigParser()


def maharanobis_distance(mean, var, target):
    distance = torch.sum((target - mean) * (target - mean) / var, 1)
    return distance


class MaharanobisLoss(nn.Module):
    def __init__(self):
        super(MaharanobisLoss, self).__init__()

    def forward(self, mean, var, target):
        distance = maharanobis_distance(mean, var, target)
        loss = torch.mean(distance)
        return loss


def train(encoder, dataloader, device, lr, weight_decay, n_epochs, grad_clip,
          save_path, name, print_every, save_every):
    optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = MaharanobisLoss()

    # Train
    print("[info] Training ...")
    for epoch in range(n_epochs):
        pbar = tqdm(dataloader)
        running_loss = 0.0

        for i, (images, src_seq, src_pos, _, _) in enumerate(pbar):
            pbar.set_description('epoch %3d / %d' % (epoch + 1, n_epochs))

            images = images.to(device)
            src_seq = src_seq.to(device)
            src_pos = src_pos.to(device)
            img_embedded = images
            mean, var = encoder(src_seq, src_pos)

            optimizer.zero_grad()

            loss = criterion(mean, var, img_embedded)
            loss.backward()

            nn.utils.clip_grad_value_(encoder.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % print_every == 0:
                pbar.set_postfix(loss=running_loss / print_every)
                running_loss = 0

        if (epoch + 1) % save_every == 0:
            save_dir = os.path.join(save_path, name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            sen_dict = encoder.state_dict()
            sen_dict.pop('embed.weight')
            torch.save(sen_dict, os.path.join(
                save_dir, 'sentence_encoder-{}.pth'.format(epoch + 1)))


def main(args):
    gpu = args.gpu
    config_path = args.config
    vocab_path = args.vocab
    img2vec_path = args.img2vec
    train_json_path = args.train_json
    name = args.name
    save_path = args.save

    print("[args] gpu=%d" % gpu)
    print("[args] config_path=%s" % config_path)
    print("[args] word2vec_path=%s" % vocab_path)
    print("[args] img2vec_path=%s" % img2vec_path)
    print("[args] train_json_path=%s" % train_json_path)
    print("[args] name=%s" % name)
    print("[args] save_path=%s" % save_path)

    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

    config.read(config_path)

    # Model parameters
    modelparams = config["modelparams"]
    sentence_encoder_name = modelparams.get("sentence_encoder")
    metric = modelparams.get("metric", "maharanobis")
    n_layers = modelparams.getint("n_layers")
    d_model = modelparams.getint("d_model")

    # Hyper parameters
    hyperparams = config["hyperparams"]
    weight_decay = hyperparams.getfloat("weight_decay")
    grad_clip = hyperparams.getfloat("grad_clip")
    lr = hyperparams.getfloat("lr")
    batch_size = hyperparams.getint("batch_size")
    n_epochs = hyperparams.getint("n_epochs")

    # Data preparation
    print("[info] Loading vocabulary ...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataloader_train = datasets.coco.get_loader(img2vec_path, train_json_path, vocab, batch_size)

    # Model preparation
    encoder = models.SentenceEncoder(vocab, sentence_encoder_name, d_model,
                                     n_layers, variance=(metric == "maharanobis")).to(device)
    train(encoder, dataloader_train, device, lr, weight_decay, n_epochs, grad_clip,
          save_path, name, args.print_every, args.save_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--img2vec", type=str, default=None)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--print_every", type=int, default=300)
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()
    main(args)
