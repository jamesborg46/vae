import torch
import torch.nn as nn
from torch import optim
from torch.nn import init

from torchvision import datasets
from torchvision import transforms

import datetime
import argparse

import wandb


class Encoder(nn.Module):

    def __init__(self, units=500, z_dim=32):
        super(Encoder, self).__init__()

        self.dense = nn.Linear(28*28, units)
        self.mean = nn.Linear(units, z_dim)
        self.log_var = nn.Linear(units, z_dim)

    def forward(self, x):
        x = torch.tanh(self.dense(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        var = torch.exp(log_var)
        std = torch.sqrt(var)

        self.kl_loss = -(0.5 * torch.sum(1 + log_var - mean**2 - var))
        return mean, std


class GaussianDecoder(nn.Module):
    def __init__(self, z_dim=32, units=500, sigmoidal_mean=False):
        super(GaussianDecoder, self).__init__()
        self.dense = nn.Linear(z_dim, units)
        self.mean = nn.Linear(units, 28*28)
        self.log_var = nn.Linear(units, 28*28)
        self.sigmoidal_mean = sigmoidal_mean

    def forward(self, z):
        z = torch.tanh(self.dense(z))
        mean = self.mean(z)
        if self.sigmoidal_mean:
            mean = torch.sigmoid(mean)
        log_var = self.log_var(z)
        var = torch.exp(log_var)
        std = torch.sqrt(var)
        return mean, std


class BenroulliDecoder(nn.Module):
    def __init__(self, z_dim=32, units=500):
        super(BenroulliDecoder, self).__init__()
        self.dense = nn.Linear(z_dim, units)
        self.out = nn.Linear(units, 28*28)

    def forward(self, z):
        z = torch.tanh(self.dense(z))
        out = torch.sigmoid(self.out(z))
        return out


class VAE(nn.Module):
    def __init__(self,
                 enoder_units=500,
                 decoder_units=500,
                 latent_dim=32,
                 sigmoidal_mean=False):

        super(VAE, self).__init__()
        self.encoder = Encoder(units=encoder_units, z_dim=latent_dim)
        self.decoder = GaussianDecoder(z_dim=latent_dim,
                                       units=decoder_units,
                                       sigmoidal_mean=sigmoidal_mean)

    def forward(self, x, num_samples=1):
        batch_size = x.shape[0]
        x = torch.flatten(x, 1)
        z_params = self.encoder(x)
        z_samples = (torch.distributions
                          .Normal(*z_params)
                          .rsample((num_samples,)))
        x_params = self.decoder(z_samples)
        reconstructed_x = torch.distributions.Normal(*x_params)
        self.reconstruction_loss = (
            -(1/num_samples) * torch.sum(reconstructed_x.log_prob(x))
        )
        self.loss = self.encoder.kl_loss + self.reconstruction_loss
        reconstructed_x = torch.reshape(reconstructed_x.sample(),
                                        (num_samples, batch_size, 1, 28, 28))

        return reconstructed_x


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model(data)
        loss = (1/128) * model.loss
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            wandb.log({
                "training loss": model.loss,
                "training kl loss": model.encoder.kl_loss / 128,
                "training reconstruction loss": model.reconstruction_loss / 128
            })


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    kl_loss = 0
    reconstruction_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model(data)
            test_loss += ((1/10000) * model.loss).item()  # sum up batch loss
            kl_loss += ((1/10000) * model.encoder.kl_loss).item()
            reconstruction_loss += (
                (1/10000) * model.reconstruction_loss).item()

    wandb.log({
        "test loss": test_loss,
        "test kl loss": kl_loss,
        "test reconstruction loss": reconstruction_loss
    })

    if epoch % 10 == 0:
        z = torch.distributions.Normal(
            loc=torch.zeros((5, 32,)),
            scale=torch.ones((5, 32,))
        )
        z_params = model.decoder(z.sample().to(device))
        samples = torch.reshape(
            torch.distributions.Normal(*z_params).sample(),
            (5, 28, 28)
        ).cpu().numpy()

        wandb.log({"examples": [wandb.Image(i) for i in samples]})

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    parser = argparse.ArgumentParser(description='PyTorch VAE')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--encoder-units', type=int, default=500)
    parser.add_argument('--decoder-units', type=int, default=500)
    parser.add_argument('--decoder-weight-decay', type=float, default=0.1)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--weight-init-std', type=float, default=0.01)
    parser.add_argument('--bias-init-std', type=float, default=0.0)
    parser.add_argument('--pre-normalization',
                        action='store_true', default=False)
    parser.add_argument('--sigmoidal-mean',
                        action='store_true', default=False)

    args = parser.parse_args()

    exp_name = datetime.datetime.now('%Y-%m-%d-%H:%M:%S') + '_' + args.name

    wandb.init(
        name=exp_name,
        project="vae-project",
        config=args,
    )

    device = torch.device('cuda:0')

    if args.pre_normalization:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_data = datasets.MNIST('./mnist/', transform=transform)
    validation_data = datasets.MNIST('./mnist/', train=False,
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=8,
    )

    model = VAE(enoder_units=args.enoder_units,
                decoder_units=args.decoder_units,
                latent_dim=args.latent_dim,
                sigmoidal_mean=args.sigmoidal_mean).to(device)

    optimizer = optim.Adagrad([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters(),
         'weight_decay': args.decoder_weight_decay}
    ], lr=args.lr)

    wandb.watch(model, log="all")

    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, 0.0, args.weight_init_std)
            init.normal_(m.bias, 0.0, args.bias_init_std)

    model.apply(weights_init)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)


if __name__ == '__main__':
    main()
