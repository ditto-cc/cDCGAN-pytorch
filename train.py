# coding: utf-8
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_cat_label(batch, label):
    w, h = batch.size(2), batch.size(3)
    batch_label = []
    for i in range(0, batch.size(0)):
        one_hot = torch.zeros([10])[label[i]] = 1
        cat_label = torch.ones([10, w, h]) * one_hot
        batch_label.append(cat_label)
    cat_label = torch.stack(batch_label).to(device)
    return torch.cat([batch, cat_label], 1)


def train_step(gen, dis, train_loader, g_loss_func, d_loss_func, g_optim, d_optim):
    gen.train()
    dis.train()

    g_loss_val = 0
    d_loss_val = 0
    noise = torch.randn(train_loader.batch_size, 100, 1, 1).to(device)
    for batch_img, batch_label in iter(train_loader):
        real_img = batch_img.unsqueeze(1).to(device)
        label = batch_label.to(device)
        real_label_img = img_cat_label(real_img, batch_label)

        noise_label = img_cat_label(noise, label)
        fake_img = gen(noise_label)
        fake_label_img = img_cat_label(fake_img.detach(), label)

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
        fake_label_img = img_cat_label(fake_img, label)
        fake_logits = dis(fake_label_img)
        g_loss = g_loss_func(fake_logits)
        g_loss_val += g_loss.cpu().item()
        g_loss.backward(retain_graph=False)
        g_optim.step()

    return g_loss_val, d_loss_val


def test_step(gen, epoch):
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(100, 100, 1, 1).to(device)
        noise_label = img_cat_label(noise, [i // 10 for i in range(100)])
        fake_img = gen(noise_label)
        plt.figure(figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                plt.imshow(fake_img[i * 10 + j, 0, :, :].cpu().numpy() * 127.5 + 127.5, cmap='gray')
                plt.axis(False)
        plt.savefig('gen_img/' + str(epoch) + '.png')
        plt.tight_layout()
        plt.close()


def train(epochs, batch_size):
    train_data = torchvision.datasets.MNIST('../data', download=True, train=True)
    test_data = torchvision.datasets.MNIST('../data', download=False, train=False)

    data = torch.cat([train_data.data, test_data.data]) / 127.5 - 1
    target = torch.cat([train_data.targets, test_data.targets])

    dataset = TensorDataset(data, target)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    gen = Generator().to(device)
    dis = Discriminator().to(device)

    def d_loss_func(real, fake):
        return torch.mean(torch.square(real - 1.0)) + torch.mean(torch.square(fake))

    def g_loss_func(fake):
        return torch.mean(torch.square(fake - 1.0))

    d_optim = torch.optim.Adam(dis.parameters(), lr=0.00005)
    g_optim = torch.optim.Adam(gen.parameters(), lr=0.00015)
    for epoch in range(epochs):
        g_loss, d_loss = train_step(gen, dis, data_loader, g_loss_func, d_loss_func, g_optim, d_optim)

        print('Epoch {}/{}, g_loss = {:.4f}, d_loss = {:.4f}'.format(epoch + 1, epochs, g_loss, d_loss))
        test_step(gen, epoch + 1)


if __name__ == '__main__':
    import datetime

    print(datetime.datetime.now())
    train(50, 500)
    print(datetime.datetime.now())
