import torch
import torch.nn as nn
from torch import optim
from torch.nn import init
import torch.nn.functional as F

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

    def __init__(self, units=500, z_dim=32, z_std_prior=1.):
        super(Encoder, self).__init__()

        self.dense = nn.Linear(28*28, units)
        self.mean = nn.Linear(units, z_dim)
        self.log_var = nn.Linear(units, z_dim)
        self.z_std_prior = z_std_prior

    def forward(self, x):
        x = torch.tanh(self.dense(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        var = torch.exp(log_var)
        std = torch.sqrt(var)

        self.kl_loss = -0.5 * (torch.sum(
            1 + log_var -
            (mean**2 + var) / self.z_std_prior**2 -
            math.log(self.z_std_prior**2)
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


class BernoulliDecoder(nn.Module):
    def __init__(self, z_dim=32, units=500):
        super(BernoulliDecoder, self).__init__()
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
                 z_std_prior=1.,
                 decoder_type='bernoulli',
                 sigmoidal_mean=False):

        super(VAE, self).__init__()
        self.encoder = Encoder(units=encoder_units,
                               z_dim=latent_dim,
                               z_std_prior=z_std_prior)

        if decoder_type == 'bernoulli':
            self.decoder = BernoulliDecoder(z_dim=latent_dim,
                                            units=decoder_units)
        elif decoder_type == 'gaussian':
            self.decoder = GaussianDecoder(z_dim=latent_dim,
                                           units=decoder_units,
                                           sigmoidal_mean=sigmoidal_mean)
        else:
            raise ValueError('Must select decoder_type'
                             'of bernoulli or gaussian')

        self.decoder_type = decoder_type

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

        reconstructed_x_params = self.decoder(z_samples)

        self.reconstruction_loss = self.get_reconstruction_loss(
            reconstructed_x_params,
            x
        )

        self.loss = self.encoder.kl_loss + self.reconstruction_loss

        self.accumulate_losses()

        if self.decoder_type == 'bernoulli':
            reconstructed_x = torch.reshape(
                reconstructed_x_params,
                (num_samples, batch_size, 1, 28, 28)
            )
        elif self.decoder_type == 'gaussian':
            reconstructed_x = torch.reshape(
                torch.distributions.Normal(*reconstructed_x_params).rsample(),
                (num_samples, batch_size, 1, 28, 28)
            )

        return reconstructed_x

    def get_reconstruction_loss(self, reconstructed_x_params, target):
        if self.decoder_type == 'bernoulli':
            samples, batch_size, dim = reconstructed_x_params.shape
            assert batch_size, dim == target.shape
            target = target.repeat(samples, 1, 1)
            reconstruction_loss = F.binary_cross_entropy(
                reconstructed_x_params,
                target,
                reduction='sum'
            )

        elif self.decoder_type == 'gaussian':
            mu, std = reconstructed_x_params
            samples, batch_size, dim = mu.shape
            dist = torch.distributions.Normal(mu, std)
            reconstruction_loss = -torch.sum(dist.log_prob(target))

        return (1/samples) * reconstruction_loss

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
        loss = (len(train_loader.dataset)/batch_size) * model.loss
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
    parser.add_argument('--decoder-type', type=str, default='bernoulli')
    parser.add_argument('--decoder-weight-decay', type=float, default=0.1)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--custom-init', action='store_true', default=False)
    parser.add_argument('--weight-init-std', type=float, default=0.01)
    parser.add_argument('--bias-init-std', type=float, default=0.0)
    parser.add_argument('--z-std-prior', type=float, default=1.0)
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

    visualizer_test_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    visualizer_test = iter(visualizer_test_loader)

    visualizer_train_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=5,
        shuffle=True,
        num_workers=1,
    )

    visualizer_train = iter(visualizer_train_loader)

    logger.info("Training set size: {}".format(len(train_loader.dataset)))
    logger.info("Test set size: {}".format(len(test_loader.dataset)))

    if args.sigmoidal_mean:
        logger.info("SIGMOIDAL_MEAN")

    model = VAE(encoder_units=args.encoder_units,
                decoder_units=args.decoder_units,
                latent_dim=args.latent_dim,
                z_std_prior=args.z_std_prior,
                decoder_type=args.decoder_type,
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
            train_input_samples, _ = visualizer_train.next()
            reconstructed_train_samples = model(
                train_input_samples.to(device)
            )[0]

            test_input_samples, _ = visualizer_test.next()
            reconstructed_test_samples = model(
                test_input_samples.to(device)
            )[0]

            z = torch.distributions.Normal(
                loc=torch.zeros((5, args.latent_dim,)),
                scale=args.z_std_prior*torch.ones((5, args.latent_dim,))
            )

            if args.decoder_type == 'bernoulli':
                x = model.decoder(z.sample().to(device))
                generated_samples = torch.reshape(x, (5, 28, 28)).cpu().numpy()

                outputs = {
                    'output': wandb.Histogram(x.cpu().numpy())
                }

            elif args.decoder_type == 'gaussian':
                x_params = model.decoder(z.sample().to(device))
                x_mean, x_std = x_params
                generated_samples = torch.reshape(
                    torch.distributions.Normal(*x_params).sample(),
                    (5, 28, 28)
                ).cpu().numpy()

                outputs = {
                    "output_mean": wandb.Histogram(x_mean.cpu().numpy()),
                    "output_std": wandb.Histogram(x_std.cpu().numpy()),
                }

            wandb.log(
                {**train_losses,
                 **test_losses,
                 **outputs,
                 "input_train_samples":
                    [wandb.Image(i) for i in train_input_samples],
                 "reconstructed_train_samples":
                    [wandb.Image(i) for i in reconstructed_train_samples],
                 "input_test_samples":
                    [wandb.Image(i) for i in test_input_samples],
                 "reconstructed_test_samples":
                    [wandb.Image(i) for i in reconstructed_test_samples],
                 "generated_samples":
                    [wandb.Image(i) for i in generated_samples],
                 "epoch": epoch+1
                 })


if __name__ == '__main__':
    main()
