# coding: utf-8
import torch
import torchvision
import tqdm
import utils
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from model import Discriminator, Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CDCGAN:
    def __init__(self, epochs=50, batch_size=256):
        self.epochs = epochs
        self.epoch = 0
        self.batch_size = batch_size

    def train_step(self, gen, dis, train_loader, g_loss_func, d_loss_func, g_optim, d_optim):
        gen.train()
        dis.train()

        g_loss_val = 0
        d_loss_val = 0
        noise = torch.randn(train_loader.batch_size, 100, 1, 1).to(device)
        num_of_batch = len(train_loader)
        desc = f'Epoch {self.epoch + 1}/{self.epochs}, Step'
        data_loader_itr = iter(train_loader)
        for _ in tqdm.trange(num_of_batch, desc=desc, total=num_of_batch):
            batch_img, batch_label = next(data_loader_itr)
            real_img = batch_img.unsqueeze(1).to(device)
            one_hot_label = torch.nn.functional.one_hot(batch_label).view([batch_img.size(0), 10, 1, 1]).to(device)

            real_label_img = torch.cat([real_img, torch.ones_like(real_img) * one_hot_label], dim=1).to(device)
            noise_label_img = torch.cat([noise[:batch_img.size(0)], one_hot_label], dim=1).to(device)

            fake_img = gen(noise_label_img)
            fake_label_img = torch.cat([fake_img, torch.ones_like(fake_img) * one_hot_label], dim=1)

            # train D
            d_optim.zero_grad()
            real_logits = dis(real_label_img)
            fake_logits = dis(fake_label_img)
            d_loss = d_loss_func(real_logits, fake_logits)
            d_loss_val += d_loss.cpu().item()
            d_loss.backward(retain_graph=True)
            d_optim.step()

            # train G
            g_optim.zero_grad()
            fake_label_img = torch.cat([fake_img, torch.ones_like(fake_img) * one_hot_label], dim=1)
            fake_logits = dis(fake_label_img)
            g_loss = g_loss_func(fake_logits)
            g_loss_val += g_loss.cpu().item()
            g_loss.backward(retain_graph=False)
            g_optim.step()

        return g_loss_val, d_loss_val

    def test_step(self, gen, noise):
        gen.eval()
        with torch.no_grad():
            label = torch.tensor([i // 10 for i in range(100)])
            label = torch.nn.functional.one_hot(label).view([100, 10, 1, 1]).to(device)
            noise_label = torch.cat([noise, label], dim=1)
            fake_img = gen(noise_label)
            plt.figure(figsize=(10, 10))
            for i in range(10):
                for j in range(10):
                    plt.subplot(10, 10, i * 10 + j + 1)
                    plt.imshow(fake_img[i * 10 + j, 0, :, :].cpu().numpy() * 127.5 + 127.5, cmap='gray')
                    plt.axis(False)
            plt.savefig(utils.path.join(utils.FAKE_MNIST_DIR, str(self.epoch) + '.png'))
            plt.tight_layout()
            plt.close()

    def train(self):
        train_data = torchvision.datasets.MNIST('../data', download=False, train=True)
        test_data = torchvision.datasets.MNIST('../data', download=False, train=False)

        data = torch.cat([train_data.data, test_data.data]) / 127.5 - 1
        target = torch.cat([train_data.targets, test_data.targets])

        dataset = TensorDataset(data, target)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        gen = Generator().to(device)
        dis = Discriminator().to(device)

        def d_loss_func(real, fake):
            return torch.mean(torch.square(real - 1.0)) + torch.mean(torch.square(fake))

        def g_loss_func(fake):
            return torch.mean(torch.square(fake - 1.0))

        d_optim = torch.optim.Adam(dis.parameters(), lr=0.00005)
        g_optim = torch.optim.Adam(gen.parameters(), lr=0.00015)
        test_noise = torch.randn(100, 100, 1, 1).to(device)
        for _ in range(self.epochs):
            g_loss, d_loss = self.train_step(gen, dis, data_loader, g_loss_func, d_loss_func, g_optim, d_optim)
            print('Epoch {}/{}, g_loss = {:.4f}, d_loss = {:.4f}'.format(self.epoch + 1, self.epochs, g_loss, d_loss))
            self.test_step(gen, test_noise)
            self.epoch += 1
