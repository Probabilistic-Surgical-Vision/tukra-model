import argparse
import os
import psutil

from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import BCELoss, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
import json

from loaders import DaVinciDataset, SCAREDDataset
from model import RandomlyConnectedModel, RandomDiscriminator

import train
from train import train_model
from train.loss import TukraLoss


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
parser.add_argument('--workers', '-w', default=8, type=int,
                    help='The number of workers to use for the dataloader.')
parser.add_argument('--training-size', default=None, nargs='?', type=int,
                    help='The number of samples to train with.')
parser.add_argument('--validation-size', default=None, nargs='?', type=int,
                    help='The number of samples to evaluate with.')
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

# Distributed Data Parallel arguments
parser.add_argument('--number-of-nodes', default=1, type=int,
                    help='The number of nodes available.')
parser.add_argument('--number-of-gpus', default=1, type=int,
                    help='The number of GPUs available.')
parser.add_argument('--global-rank', default=0, type=int,
                    help='The global rank of the node running this program.')
parser.add_argument('--master-address', default='localhost', type=str,
                    help='The IP address to spawn processes from.')
parser.add_argument('--master-port', default=3000, type=int,
                    help='The port for processes to communicate through.')
parser.add_argument('--init-seed', default=0, type=int,
                    help='Set the manual seed for initialising models.')
parser.add_argument('--debug-distributed', action='store_true', default=False,
                    help='Set torch.distributed logging to "DETAILED".')


def main(gpu_index: int, args: argparse.Namespace) -> None:
    rank = (args.global_rank * args.number_of_gpus) + gpu_index
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=rank)

    torch.manual_seed(args.init_seed)

    if rank == 0:
        print('Arguments passed:')
        for key, value in vars(args).items():
            print(f'\t- {key}: {value}')

        print('Live Python Processes:')
        for p in psutil.process_iter():
            if 'python' not in p.name():
                continue

            created = datetime.fromtimestamp(p.create_time()) \
                .strftime('%d-%m-%Y %H:%M:%S')

            print(f'\t- {p.name()} ({p.pid}) created {created}.')

    dataset_path = os.path.join(args.home, 'datasets', args.dataset)
    dataset_class = DaVinciDataset if args.dataset == 'da-vinci' \
        else SCAREDDataset

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

    if rank == 0:
        print(f'Dataset size:'
              f'\n\tTrain: {len(train_dataset):,} images.'
              f'\n\tTest: {len(val_dataset):,} images.')

    train_sampler = DistributedSampler(train_dataset, rank=rank,
                                       num_replicas=args.world_size)

    val_sampler = DistributedSampler(val_dataset, rank=rank,
                                     num_replicas=args.world_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.workers,
                              sampler=train_sampler)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.workers,
                            sampler=val_sampler)

    device = torch.device(f'cuda:{gpu_index}') \
        if torch.cuda.is_available() and not args.no_cuda \
        else torch.device('cpu')

    model = RandomlyConnectedModel(**config['model']).to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[gpu_index])

    loss_function = TukraLoss(**config['loss']).to(device)

    if rank == 0:
        model_parameters = sum(p.numel() for p in model.parameters())
        print(f'Model has {model_parameters:,} learnable parameters.'
              f'\n\tUsing CUDA? {next(model.parameters()).is_cuda}')

    if args.adversarial:
        disc = RandomDiscriminator(**config['discriminator']).to(device)
        disc = SyncBatchNorm.convert_sync_batchnorm(disc)
        disc = DistributedDataParallel(disc, device_ids=[gpu_index])

        disc_loss_function = BCELoss().to(device)

        if rank == 0:
            disc_parameters = sum(p.numel() for p in disc.parameters())
            print(f'Disc has {disc_parameters:,} learnable parameters.'
                  f'\n\tUsing CUDA? {next(disc.parameters()).is_cuda}')

    else:
        disc = None
        disc_loss_function = None

    if args.finetune_from is not None:
        state_dict = torch.load(args.finetune_from, map_location=device)

        if disc is not None:
            model_state = state_dict['model']
            disc_state = state_dict['disc']

            disc.load_state_dict(disc_state)
        else:
            model_state = state_dict

        model.load_state_dict(model_state)

    date = datetime.now().strftime('%Y%m%d%H%M%S')
    folder = f'model_{date}'

    if args.save_model_to is not None and rank == 0:
        model_directory = os.path.join(args.save_model_to, folder)
        os.makedirs(model_directory, exist_ok=True)
    else:
        model_directory = None

    if args.save_results_to is not None and rank == 0:
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
                       device=device, no_pbar=args.no_pbar, rank=rank)

    training_losses, validation_metrics = loss

    if results_directory is not None and rank == 0:
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
    if args.number_of_nodes > 1:
        raise ValueError('Running more than one node is not supported.')

    args.world_size = args.number_of_nodes * args.number_of_gpus

    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = str(args.master_port)

    if args.debug_distributed:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    mp.spawn(main, nprocs=args.number_of_gpus, args=(args,))
