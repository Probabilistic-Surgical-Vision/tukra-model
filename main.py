import argparse
import os

from datetime import datetime

import torch
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
import json

from loaders import DaVinciDataset, SCAREDDataset
from model import RandomlyConnectedModel, RandomDiscriminator

import train
from train import train_model
from train.loss import TukraLoss
import train.utils as u

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str,
                    help='The config file path to build the model from.')
parser.add_argument('dataset', choices=['da-vinci', 'scared'],
                    help='The dataset to use for training (must be'
                    'either "da-vinci" or "scared").')
parser.add_argument('--epochs', '-e', default=200, type=int,
                    help='The number of epochs to train the model for.')
parser.add_argument('--learning-rate', '-lr', default=1e-4, type=float,
                    help='The initial learning rate for training.')
parser.add_argument('--batch-size', '-b', default=8, type=int,
                    help='The batch size to train/evaluate the model with.')
parser.add_argument('--adversarial', action='store_true', default=False,
                    help='Train the model with a discriminator.')
parser.add_argument('--finetune-from', default=None, type=str,
                    help='The path to the model to finetune.')
parser.add_argument('--training-size', default=None, nargs='?', type=int,
                    help='The number of samples to train with.')
parser.add_argument('--validation-size', default=None, nargs='?', type=int,
                    help='The number of samples to evaluate with.')
parser.add_argument('--workers', '-w', default=8, type=int,
                    help='The number of workers to use for the dataloader.')
parser.add_argument('--save-model-to', default=None, type=str,
                    help='The path to save models to.')
parser.add_argument('--save-results-to', default=None, type=str,
                    help='The path to save results and images to.')
parser.add_argument('--save-model-every', default=10, type=int,
                    help='The number of epochs between saving the model.')
parser.add_argument('--evaluate-every', default=10, type=int,
                    help='The number of epochs between evaluations.')
parser.add_argument('--no-pbar', action='store_true', default=False,
                    help='Prevent program from printing the progress bar.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Prevent program from training model using cuda.')
parser.add_argument('--no-augment', action='store_true', default=False,
                    help='Prevent program from augmenting training images.')
parser.add_argument('--home', default=os.environ['HOME'], type=str,
                    help='Override the home directory (to find datasets).')


def main(args: argparse.Namespace) -> None:
    print("Arguments passed:")
    for key, value in vars(args).items():
        print(f'\t- {key}: {value}')

    dataset_path = os.path.join(args.home, 'datasets', args.dataset)
    dataset_class = DaVinciDataset if args.dataset == 'da-vinci' \
        else SCAREDDataset

    device = torch.device('cuda') if torch.cuda.is_available() \
        and not args.no_cuda else torch.device('cpu')

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    augment_transform = transforms.Compose([
        train.transforms.ResizeImage((256, 512)),
        train.transforms.RandomFlip(0.5),
        train.transforms.ToTensor(),
        train.transforms.RandomAugment(0.5, gamma=(0.8, 1.2),
                                       brightness=(0.5, 2.0),
                                       colour=(0.8, 1.2))])

    no_augment_transform = transforms.Compose([
        train.transforms.ResizeImage((256, 512)),
        train.transforms.ToTensor()])

    train_transform = no_augment_transform \
        if args.no_augment else augment_transform

    train_dataset = dataset_class(dataset_path, 'train',
                                  train_transform, args.training_size)
    val_dataset = dataset_class(dataset_path, 'test',
                                no_augment_transform, args.validation_size)

    print(f'Dataset size:'
          f'\n\tTrain: {len(train_dataset):,} images.'
          f'\n\tTest: {len(val_dataset):,} images.')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    model = RandomlyConnectedModel(**config['model']).to(device)
    loss_function = TukraLoss(**config['loss']).to(device)

    model_parameters = sum(p.numel() for p in model.parameters())
    print(f'Model has {model_parameters:,} learnable parameters.'
          f'\n\tUsing CUDA? {next(model.parameters()).is_cuda}')

    if args.adversarial:
        disc = RandomDiscriminator(**config['discriminator']).to(device)
        disc_loss_function = BCELoss().to(device)

        disc_parameters = sum(p.numel() for p in disc.parameters())
        print(f'Discriminator has {disc_parameters:,} learnable parameters.'
              f'\n\tUsing CUDA? {next(disc.parameters()).is_cuda}')

    else:
        disc = None
        disc_loss_function = None

    if args.finetune_from is not None:
        state_dict = torch.load(args.finetune_from, map_location=device)

        if disc is not None:
            model_state = u.prepare_state_dict(state_dict['model'])
            disc_state = u.prepare_state_dict(state_dict['disc'])

            disc.load_state_dict(disc_state)
        else:
            model_state = u.prepare_state_dict(state_dict)

        model.load_state_dict(model_state)

    date = datetime.now().strftime('%Y%m%d%H%M%S')
    folder = f'model_{date}'

    if args.save_model_to is not None:
        model_directory = os.path.join(args.save_model_to, folder)
        os.makedirs(model_directory, exist_ok=True)
    else:
        model_directory = None

    if args.save_results_to is not None:
        results_directory = os.path.join(args.save_results_to, folder)
        os.makedirs(results_directory, exist_ok=True)
    else:
        results_directory = None

    loss = train_model(model, train_loader, loss_function, args.epochs,
                       args.learning_rate, disc, disc_loss_function,
                       val_loader=val_loader, save_model_to=model_directory,
                       save_evaluation_to=results_directory,
                       save_every=args.save_model_every,
                       evaluate_every=args.evaluate_every,
                       finetune=(args.finetune_from is not None),
                       device=device, no_pbar=args.no_pbar)

    training_losses, validation_metrics = loss

    if results_directory is not None:
        losses_filepath = os.path.join(results_directory, 'results.json')

        model_train_losses, disc_train_losses = zip(*training_losses)
        disc_train_losses = disc_train_losses if args.adversarial else None

        results_dict = {
            'arguments': vars(args),
            'config': config,
            'losses': {
                'training': {
                    'model': model_train_losses,
                    'discriminator': disc_train_losses
                }
            }
        }

        if len(validation_metrics) > 0:
            left_ssim, right_ssim = zip(*validation_metrics)

            results_dict['losses'].update({
                'validation': {
                    'ssim': {
                        'left': left_ssim,
                        'right': right_ssim
                    }
                }
            })

        print(f'Saving args and losses to:\n\t{losses_filepath}')
        with open(losses_filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
