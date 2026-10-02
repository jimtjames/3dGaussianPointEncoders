import os, sys
from typing import List
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from models.encoders import GaussianEncoder
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler
from .soap import SOAP

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config, distill=False):
    if distill:
        opti_config = config.distill_optimizer
        sche_config = config.distill_scheduler
    else:
        opti_config = config.optimizer
        sche_config = config.scheduler
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizers = [optim.AdamW(param_groups, **opti_config.kwargs)]
    elif opti_config.type == 'Adam':
        optimizers = [optim.Adam(base_model.parameters(), **opti_config.kwargs)]
    elif opti_config.type == 'SGD':
        optimizers = [optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)]
    elif opti_config.type == 'SOAP':
        optimizers = [SOAP(base_model.parameters(), **opti_config.kwargs)]
    elif opti_config.type == 'mixed':
        def generate_param_groups(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            sgd_param_group = []
            adamw_param_group = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if name.endswith("mu"):
                    sgd_param_group.append({'params': [param], 'lr': opti_config.mean_lr, 'weight_decay': opti_config.mean_decay})
                elif name.endswith("l_triangs"):
                    sgd_param_group.append({'params': [param], 'lr': opti_config.cov_lr, 'weight_decay': opti_config.cov_decay})
                elif len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            adamw_param_group.append({'params': no_decay, 'weight_decay': 0.})
            adamw_param_group.append({'params': decay, 'weight_decay': weight_decay})
            return sgd_param_group, adamw_param_group

        sgd_param_group, adamw_param_group = generate_param_groups(base_model, weight_decay=opti_config.kwargs.weight_decay)
        adamw = optim.AdamW(adamw_param_group, **opti_config.kwargs)
        sgd = optim.SGD(sgd_param_group, momentum=opti_config.momentum, **opti_config.kwargs)
        optimizers = [adamw, sgd]
    else:
        raise NotImplementedError()
    
    schedulers = []
    if sche_config.type == 'LambdaLR':
        for optimizer in optimizers:
            schedulers.append(build_lambda_sche(optimizer, sche_config.kwargs))
    elif sche_config.type == 'CosLR':
        for optimizer in optimizers:
            scheduler = CosineLRScheduler(optimizer,
                    t_initial=sche_config.kwargs.epochs,
                    cycle_mul=1,
                    lr_min=1e-6,
                    cycle_decay=0.1,
                    warmup_lr_init=1e-6,
                    warmup_t=sche_config.kwargs.initial_epochs,
                    cycle_limit=1,
                    t_in_epochs=True)
            schedulers.append(scheduler)
    elif sche_config.type == 'StepLR':
        for optimizer in optimizers:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
            schedulers.append(scheduler)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
            schedulers.append(bnscheduler)
    
    return optimizers, schedulers

def resume_model(base_model, args, logger = None, given_path=None):
    if given_path is None:
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    else:
        ckpt_path = given_path
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # parameter resume of base model
    # if args.local_rank == 0:
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizers, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    if isinstance(optimizers, List):
        if not isinstance(state_dict['optimizer'], List):
            optimizers[-1].load_state_dict(state_dict['optimizer'])
        else:
            for i, optimizer in enumerate(optimizers):
                optimizer.load_state_dict(state_dict['optimizer'][i])
    else:
        optimizers.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : [optimizer.state_dict() for optimizer in optimizers] if isinstance(optimizers, List) else optimizers.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 
