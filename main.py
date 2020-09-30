# coding: utf-8

from train import CDCGAN

if __name__ == '__main__':
    gan = CDCGAN(batch_size=64)

    gan.train()
