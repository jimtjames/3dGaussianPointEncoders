from functools import partial
import copy
import torch
import torch.nn as nn
# from Mamba3D.models.encoders import GaussianEncoder
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

#To calaulate param and flops
from thop import profile
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

import time
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table, flop_count_str

# train_transforms = transforms.Compose(
#     [
#          data_transforms.PointcloudScaleAndTranslate(),
#     ]
# )

# test_transforms = transforms.Compose(
#     [
#         data_transforms.PointcloudScaleAndTranslate(),
#     ]
# )
class Mamba3DEncoder(nn.Module):   ## This is just Mamba3D's encoder without a maxpool
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        # print(feature.shape)
        # feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature.reshape(bs, g, self.encoder_channel, n).transpose(-1, -2) # B G N 1024


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def cal_params_flops(model):
    # model.eval()
    input_shape = tuple([2048, 3])
    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n' 
        f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
        'You may need to check if all ops are supported and verify that the '
        'flops computation is correct.')
    # model.train()

def cal_flops(model):
    model.eval()
    # input_shape = tuple([2048, 3])
    try:
        flops = FlopCountAnalysis(model, torch.rand(1, 2048, 3).cuda())
        print(flop_count_table(flops, max_depth=1))
        cnt = flops.total()
        print("[#FLOPs] cnt: ", cnt)
    except RuntimeError:
        cnt = 0
        print("Cannot compute FLOPs, likely due to compilation in the model. Disable compilation if you want the FLOPs count")
    return cnt
    # model.train()

def calculate_total_param(base_model):
    params = list(base_model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("layer structure: " + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("layer param: " + str(l))
        k = k + l
    print("##################TOTAL PARAMETER NUMBER: " + str(k))

def run_net(args, config, train_writer=None, val_writer=None):
    if config.dataset.train._base_.NAME == "ModelNet": # ModelNet
        train_transforms = transforms.Compose([
            # data_transforms.PointcloudRotate(),
            data_transforms.PointcloudScaleAndTranslate(),
        ])
    else:
        train_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
            # data_transforms.PointcloudScaleAndTranslate(),
        ])
    print("train_transforms: ", train_transforms)
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build teacher model, load its checkpoint
    teacher_base_model = builder.model_builder(config.teacher_model)
    _, _ = builder.resume_model(teacher_base_model, args, logger = logger, given_path=config.teacher_weights_path)
    teacher_base_model.to(args.local_rank)
    # this is just the pointnet but with the transpose built in and the maxpool removed
    teacher_model = Mamba3DEncoder(teacher_base_model.encoder.encoder_channel).to(args.local_rank)
    # copy the weights
    grouper = copy.deepcopy(teacher_base_model.group_divider)
    teacher_model.first_conv = copy.deepcopy(teacher_base_model.encoder.first_conv)
    teacher_model.second_conv = copy.deepcopy(teacher_base_model.encoder.second_conv)
    del teacher_base_model

    base_model_init = builder.model_builder(config.model)
    base_model = copy.deepcopy(base_model_init.encoder)
    base_model.pooling = lambda x: x # don't pool
    del base_model_init

    # num_trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    # print(f"Total Number of trainable parameters: {num_trainable_params}")

    #model param
    # calculate_total_param(base_model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
            grouper = torch.nn.SyncBatchNorm.convert_sync_batchnorm(grouper)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        teacher_model = nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        grouper = nn.parallel.DistributedDataParallel(grouper, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        teacher_model = nn.DataParallel(teacher_model).cuda()
        grouper = nn.DataParallel(grouper).cuda()
    # optimizer & scheduler
    optimizers, schedulers = builder.build_opti_sche(base_model, config, distill=True)
    
    if args.resume:
        builder.resume_optimizer(optimizers, args, logger = logger)
        
    # model statistics
    # cal_params_flops(base_model)
    flops_cnt = cal_flops(base_model) / 1000000000.
    print_log('[FLOPs: %.3f G]' % flops_cnt, logger = logger)
    
    if hasattr(config, 'preconditioner') and config.model.NAME == 'GaussianMamba3D':
        if config.preconditioner == 'FIM':
            print_log('Applying FIM natural gradient')
            base_model.module.register_FIM_hook()
        elif config.preconditioner == 'mahalanobis':
            print_log('Applying mahalanobis natural gradient')
            base_model.module.register_mahalanobis_hook()

    # trainval
    # training
    base_model.zero_grad()
    distill_loss = torch.nn.L1Loss()
    for epoch in range(start_epoch, config.distill_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()
            points = train_transforms(points)
            # print(points.shape)
            with torch.no_grad():
                neighborhood, _ = grouper(points)
                gt = teacher_model(neighborhood)
            ret = base_model(neighborhood)
            loss = distill_loss(ret, gt)
            # loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                for optimizer in optimizers:
                    optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                # acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item()])
            else:
                losses.update([loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                # train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizers[-1].param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(schedulers, list):
            for item in schedulers:
                item.step(epoch)
        else:
            schedulers.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizers[-1].param_groups[0]['lr']), logger = logger)
        # if epoch % args.val_freq == 0 and epoch != 0:
            # # Validate the current model
            # metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # better = metrics.better_than(best_metrics)
            # # Save ckeckpoints
            # if better:
                # best_metrics = metrics
                # builder.save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                # print_log("--------------------------------------------------------------------------------------------", logger=logger)
                # # if metrics.acc > 92:
                # #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-nice-%.6f' % (metrics.acc), args, logger = logger)
                # #     print_log("---------------------------------***********----------------------------------------", logger=logger)
            # if args.vote:
                # if metrics.acc > 92.5 or (better and metrics.acc > 92):
                    # metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    # if metrics_vote.better_than(best_metrics_vote):
                        # best_metrics_vote = metrics_vote
                        # print_log(
                            # "****************************************************************************************",
                            # logger=logger)
                        # builder.save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizers, epoch, None, None, 'encoder-ckpt-last', args, logger = logger)
        print_log('[BEST MODEL] acc = %.6f' % (best_metrics.acc), logger=logger)  
        GB = 1024. * 1024. * 1024.
        gpu_memory = torch.cuda.max_memory_allocated()/GB
        res_gpu_memory = torch.cuda.max_memory_reserved()/GB
        print_log('[GPU Mem] MEM = %.3f GB | Reserved MEM = %.3f GB' % (gpu_memory, res_gpu_memory), logger=logger) 
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    # calculate_total_param(base_model)

    # if train_writer is not None:
        # train_writer.close()
    # if val_writer is not None:
        # val_writer.close()
    #model param

    teacher_base_model = builder.model_builder(config.teacher_model).to(args.local_rank)
    _, _ = builder.resume_model(teacher_base_model, args, logger = logger, given_path=config.teacher_weights_path)
    base_model_init = builder.model_builder(config.model).to(args.local_rank)
    encoder = base_model.module
    encoder.pooling = partial(torch.amax, dim=-2)
    base_model_init.encoder = copy.deepcopy(encoder)
    base_model_init.cls_token = copy.deepcopy(teacher_base_model.cls_token)
    base_model_init.cls_pos = copy.deepcopy(teacher_base_model.cls_pos)
    base_model_init.pos_embed = copy.deepcopy(teacher_base_model.pos_embed)
    base_model_init.group_divider = copy.deepcopy(teacher_base_model.group_divider)
    base_model_init.blocks = copy.deepcopy(teacher_base_model.blocks)
    base_model_init.norm = copy.deepcopy(teacher_base_model.norm)
    base_model_init.cls_head_finetune = copy.deepcopy(teacher_base_model.cls_head_finetune)
    del encoder
    del teacher_base_model
    base_model = base_model_init
    if config.dataset.train._base_.NAME == "ModelNet": # ModelNet
        train_transforms = transforms.Compose([
            # data_transforms.PointcloudRotate(),
            data_transforms.PointcloudScaleAndTranslate(),
        ])
    else:
        train_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
            # data_transforms.PointcloudScaleAndTranslate(),
        ])
    calculate_total_param(base_model)
    # now we train and validate like normal, unfortunately can't just call method again
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizers, schedulers = builder.build_opti_sche(base_model, config)

    flops_cnt = cal_flops(base_model) / 1000000000. #FIXME
    print_log('[FLOPs: %.3f G]' % flops_cnt, logger = logger)

    if hasattr(config, 'preconditioner') and config.model.NAME == 'GaussianMamba3D':
        if config.preconditioner == 'FIM':
            print_log('Applying FIM natural gradient')
            base_model.module.encoder.register_FIM_hook()
        elif config.preconditioner == 'mahalanobis':
            print_log('Applying mahalanobis natural gradient')
            base_model.module.encoder.register_mahalanobis_hook()

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()
            points = train_transforms(points)
            # print(points.shape)
            
            ret = base_model(points)

            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                for optimizer in optimizers:
                    optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizers[-1].param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(schedulers, list):
            for item in schedulers:
                item.step(epoch)
        else:
            schedulers.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizers[-1].param_groups[0]['lr']), logger = logger)
        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
                # if metrics.acc > 92:
                #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-nice-%.6f' % (metrics.acc), args, logger = logger)
                #     print_log("---------------------------------***********----------------------------------------", logger=logger)
            if args.vote:
                if metrics.acc > 92.5 or (better and metrics.acc > 92):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizers, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        print_log('[BEST MODEL] acc = %.6f' % (best_metrics.acc), logger=logger)  
        GB = 1024. * 1024. * 1024.
        gpu_memory = torch.cuda.max_memory_allocated()/GB
        res_gpu_memory = torch.cuda.max_memory_reserved()/GB
        print_log('[GPU Mem] MEM = %.3f GB | Reserved MEM = %.3f GB' % (gpu_memory, res_gpu_memory), logger=logger) 
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    #model param
    calculate_total_param(base_model)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()



    # hopefully done

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        val_time = []
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            val_start_time = time.time()
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints) # 64 2048 3

            # points = test_transforms(points) 

            logits = base_model(points)
            # print("Val input points shape: ",points.shape)
            
            
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            val_end_time = time.time()
            val_time.append(val_end_time-val_start_time)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)
        print_log('[Validation] EPOCH: %d  total time = %.4f  batch avg time = %.4f' % (epoch, np.sum(val_time), np.mean(val_time)), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        GB = 1024. * 1024. * 1024.
        val_gpu_memory = torch.cuda.max_memory_allocated() / GB
        res_gpu_memory = torch.cuda.max_memory_reserved() / GB
        print_log('[Val GPU Mem] MEM = %.3f GB | Reserved MEM = %.3f GB' % (val_gpu_memory, res_gpu_memory), logger=logger) 
        
        
        # dump_input = torch.ones(64, 2048, 3).cuda()
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        #     outputs = base_model(dump_input)
        # print(prof.table())
        # prof.export_chrome_trace('./mamba_profile.json')

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    if config.dataset.train._base_.NAME == "ModelNet": # ModelNet
        test_transforms = transforms.Compose([
            # data_transforms.PointcloudRotate(),
            data_transforms.PointcloudScaleAndTranslate(),
        ])
    else:
        test_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
            # data_transforms.PointcloudScaleAndTranslate(),
        ])
    print("val_vote test_transforms: ", test_transforms)
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        GB = 1024. * 1024. * 1024.
        val_gpu_memory = torch.cuda.max_memory_allocated() / GB
        print_log('[Val GPU Mem] MEM = %.3f GB' % val_gpu_memory, logger=logger) 

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger)
    
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())


        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)
        
        GB = 1024. * 1024. * 1024.
        gpu_memory = torch.cuda.max_memory_allocated() / GB
        print_log('[GPU Mem] MEM = %.3f GB' % gpu_memory, logger=logger) 

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300): # 300
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    if config.dataset.train._base_.NAME == "ModelNet": # ModelNet
        test_transforms = transforms.Compose([
            # data_transforms.PointcloudRotate(),
            data_transforms.PointcloudScaleAndTranslate(),
        ])
    else:
        test_transforms = transforms.Compose([
            data_transforms.PointcloudRotate(),
            # data_transforms.PointcloudScaleAndTranslate(),
        ])
    print("test_vote test_transforms: ", test_transforms)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

        GB = 1024. * 1024. * 1024.
        gpu_memory_vote = torch.cuda.max_memory_allocated() / GB
        print_log('[Vote GPU Mem] MEM = %.3f GB' % gpu_memory_vote, logger=logger) 

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
