import torch
import torch.nn as nn
from torch import optim
from torch.nn import init

from torchvision import datasets
from torchvision import transforms

import datetime
import argparse
import math

import logging

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):

    def __init__(self, units=500, z_dim=32, prior_std=1):
        super(Encoder, self).__init__()

        self.dense = nn.Linear(28*28, units)
        self.mean = nn.Linear(units, z_dim)
        self.log_var = nn.Linear(units, z_dim)
        self.prior_std = prior_std

    def forward(self, x):
        x = torch.tanh(self.dense(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        var = torch.exp(log_var)
        std = torch.sqrt(var)

        self.kl_loss = -0.5 * (torch.sum(
            1 + log_var -
            (mean**2 + var) / self.prior_std**2 -
            math.log(self.prior_std**2)
        ))

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
                 encoder_units=500,
                 decoder_units=500,
                 latent_dim=32,
                 sigmoidal_mean=False):

        super(VAE, self).__init__()
        self.encoder = Encoder(units=encoder_units, z_dim=latent_dim)
        self.decoder = GaussianDecoder(z_dim=latent_dim,
                                       units=decoder_units,
                                       sigmoidal_mean=sigmoidal_mean)
        self.cumulative_losses = {
            "total_loss": 0,
            "kl_loss": 0,
            "reconstruction_loss": 0,
        }

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
        self.accumulate_losses()
        reconstructed_x = torch.reshape(reconstructed_x.sample(),
                                        (num_samples, batch_size, 1, 28, 28))

        return reconstructed_x

    def accumulate_losses(self):
        self.cumulative_losses['total_loss'] += self.loss
        self.cumulative_losses['kl_loss'] += self.encoder.kl_loss
        self.cumulative_losses['reconstruction_loss'] += self.reconstruction_loss

    def get_cumulative_losses(self, size=60000):
        mode = "training_" if self.training else "testing_"
        losses = {
            mode + key: value / size
            for key, value in self.cumulative_losses.items()
        }

        self.cumulative_losses = {
            "total_loss": 0,
            "kl_loss": 0,
            "reconstruction_loss": 0,
        }

        return losses


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    num_training_steps = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model(data)
        loss = (1/batch_size) * model.loss
        loss.backward()
        optimizer.step()

        num_training_steps += batch_size

        if batch_idx % 200 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    return model.get_cumulative_losses()


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model(data)
            test_loss += ((1/len(test_loader.dataset)) * model.loss).item()  # sum up batch loss

    logger.info('Test set: Average loss: {:.4f}\n'.format(test_loss))
    return model.get_cumulative_losses(size=10000)


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
    parser.add_argument('--custom-init', action='store_true', default=False)
    parser.add_argument('--weight-init-std', type=float, default=0.01)
    parser.add_argument('--bias-init-std', type=float, default=0.0)
    parser.add_argument('--pre-normalization',
                        action='store_true', default=False)
    parser.add_argument('--sigmoidal-mean',
                        action='store_true', default=False)

    args = parser.parse_args()

    logger.info(args)

    exp_name = (datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                + '_' + args.name)

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

    logger.info("Training set size: {}".format(len(train_loader.dataset)))
    logger.info("Test set size: {}".format(len(test_loader.dataset)))

    model = VAE(encoder_units=args.encoder_units,
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
            if args.bias_init_std == 0.0:
                init.zeros_(m.bias)
            else:
                init.normal_(m.bias, 0.0, args.bias_init_std)

    if args.custom_init:
        model.apply(weights_init)

    for epoch in range(args.epochs):
        train_losses = train(model, device, train_loader, optimizer, epoch)
        test_losses = test(model, device, test_loader, epoch)

        with torch.no_grad():
            z = torch.distributions.Normal(
                loc=torch.zeros((5, 32,)),
                scale=torch.ones((5, 32,))
            )

            x_params = model.decoder(z.sample().to(device))
            x_mean, x_std = x_params
            samples = torch.reshape(
                torch.distributions.Normal(*x_params).sample(),
                (5, 28, 28)
            ).cpu().numpy()

            wandb.log(
                {**train_losses,
                 **test_losses,
                 "examples": [wandb.Image(i) for i in samples],
                 "output_mean": wandb.Histogram(x_mean.cpu().numpy()),
                 "output_std": wandb.Histogram(x_std.cpu().numpy()),
                 "epoch": epoch+1
                 })


if __name__ == '__main__':
    main()
