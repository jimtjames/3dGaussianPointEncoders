from functools import partial
import os
import datetime
import mlflow
from numpy.random import sample
import torch
import numpy as np
import sys
import pandas as pd
import provider
import tempfile
import torch_optimizer as optim
from soap import SOAP
# from scene import Scene, GaussianModel
from models import *#GaussianBasisPointNet, HalfCosBasisPointNet, RaisedCosBasisPointNet, SincBasisPointNet
from datasets import DistillationDataLoader, ModelNetDataLoader, ScanObjectNN, ModelNet40C, im_to_ptcloud
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from vectorization import unvech

from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def end_to_end(params: dict):
    if 'driver_cwd' in params:
        os.chdir(params['driver_cwd'])
    use_mlflow = params.get('use_mlflow', False)
    run = mlflow.active_run() if use_mlflow else None
    args_dict = { 'num_point': 1024, 'use_uniform_sample': True, 'use_normals': False, 'num_category': 40 }
    ds_args = Namespace(**args_dict)
    if params['dataset'] == 'modelnet':
        test_data = ModelNetDataLoader('./data/modelnet40_normal_resampled/', args=ds_args, split='test', process_data=True)
        train_data = ModelNetDataLoader('./data/modelnet40_normal_resampled/', args=ds_args, split='train', process_data=True)
        valid_data = ModelNetDataLoader('./data/modelnet40_normal_resampled/', args=ds_args, split='valid', process_data=True)
        cloud_range = torch.tensor([[-1, 1], [-1, 1], [-1, 1]])
        k = 40
        dim = 3
    elif params['dataset'] == 'mnist':
        transform = partial(im_to_ptcloud, n_points=256)
        test_data = MNIST(root='data/MNIST', train=False, transform=transform, download=True)
        valid_data = MNIST(root='data/MNIST', train=False, transform=transform, download=False)
        train_data = MNIST(root='data/MNIST', train=True, transform=transform, download=False)
        dim = 2
        cloud_range = torch.tensor([[-1, 1], [-1, 1]])
        k = 10
    else:
        version = 'objectdataset_augmentedrot_scale75' if 'hard' in params['dataset'] else 'objectdataset'
        test_data = ScanObjectNN(subset='test', version=version)
        valid_data = ScanObjectNN(subset='valid', version=version, split_train=True)
        train_data = ScanObjectNN(subset='train', version=version, split_train=True)
        cloud_range = torch.tensor([[-5, 5], [-5, 5], [-5, 5]])
        k = 15
        dim = 3
    # train_data = DistillationDataLoader(4096)
    batch_size = params['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    device = 'cuda'
    constructor = GaussianBasisPointNet
    model = constructor(n_primitives=params['n_primitives'],
                                        in_dim=dim,
                                        cloud_range=cloud_range,
                                        init_mode=params['init_mode'],
                                        global_dim=params['global_dim'],
                                        k=k,
                                        normalized=params['normalized'],
                                        use_tnet=params['use_tnet'],
                                        isotropic=params['isotropic'],
                                        diagonal=params['diagonal'],
                                        ).to(device)
    if params['hook'] == 'FIM':
        model.feat.register_FIM_hook()
        model.tnet.feat.register_FIM_hook()
    elif params['hook'] == 'mahalanobis':
        model.feat.register_mahalanobis_hook()
        model.tnet.feat.register_mahalanobis_hook()
    # saveddir = 'saved/%s' % exp_name
    # os.makedirs(saveddir, exist_ok=True)
    if run is not None:
        run_id = str(run.info.run_id)
        run_name = str(run.info.run_name)
    else:
        run_id = "local"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_name = f"{params.get('primitive', 'model')}_{params.get('dataset', 'dataset')}_{timestamp}"
    saved_dirs = os.path.join('saved', run_name, 'checkpoints')# % exp_name
    gaussian_dirs = os.path.join('saved', run_name, 'gaussian_params')# % exp_name
    os.makedirs(saved_dirs, exist_ok=True)
    os.makedirs(gaussian_dirs, exist_ok=True)
    # model.freeze_encoder()
    # model.classifier.requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()
    first_iter = 0
    write_every = 100
    l = []
    l_sgd = []
    l_mean = []
    l_cov = []
    mean_params = {'params': [model.feat.mu], 'lr': params['mean_lr'], "name": "feat.mean"}
    cov_params = {'params': [model.feat.l_triangs], 'lr': params['cov_lr'], "name": "feat.cov"}
    classifier_modules = [model.fc1, model.fc2, model.bn1, model.bn2]
    for module in classifier_modules:
        l.append({'params': module.parameters(), 'lr': params['cls_lr']})
    optimizer = params['optimizer']
    dual_optim = params['dual_optim']
    coord_desc = params['coord_desc']
    if dual_optim:
        l_sgd.append(mean_params)
        l_sgd.append(cov_params)
    elif coord_desc:
        l_mean.append(mean_params)
        l_cov.append(cov_params)
    else:
        l.append(mean_params)
        l.append(cov_params)
    wd_weighting = 0
    wd_cls = 0
    l.append({'params': model.feat.weighting.parameters(), 'lr': params['weighting_lr'], 'weight_decay': wd_weighting, "name": "weighting"})
    l.append({'params': model.fc3.parameters(), 'lr': params['cls_lr'], 'weight_decay': wd_cls, "name": "cls"})
    mean_params = {'params': [model.tnet.feat.mu], 'lr': params['mean_lr'], "name": "tnet.mean"}
    cov_params = {'params': [model.tnet.feat.l_triangs], 'lr': params['cov_lr'], "name": "tnet.cov"}
    classifier_modules = [model.tnet.fc1, model.tnet.fc2, model.tnet.fc3, model.tnet.bn1, model.tnet.bn2]
    for module in classifier_modules:
        l.append({'params': module.parameters(), 'lr': params['cls_lr']})
    if dual_optim:
        l_sgd.append(mean_params)
        l_sgd.append(cov_params)
    elif coord_desc:
        l_mean.append(mean_params)
        l_cov.append(cov_params)
    else:
        l.append(mean_params)
        l.append(cov_params)
    l.append({'params': model.tnet.feat.weighting.parameters(), 'lr': params['weighting_lr'], "name": "tnet.weighting"})
    # l.append(opacity)
    if optimizer == 'shampoo':
        opt = optim.Shampoo(l, lr=1e-3)
    elif optimizer == 'soap':
        opt = SOAP(l, lr=1e-3)
    else:
        opt = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
    if dual_optim:
        opt_sgd = torch.optim.SGD(l_sgd, lr=0.0)
    elif coord_desc:
        opt_mean = torch.optim.AdamW(l_mean, lr=0.0, eps=1e-15)
        opt_cov = torch.optim.AdamW(l_cov, lr=0.0, eps=1e-15)
    # opt = torch.optim.Adam(l, lr=0.0, weight_decay=1e-5, eps=1e-15)
    # self.xyz_scheduler_args = gaussian.get_expon_lr_func(lr_init=position_lr_init,
                # lr_final=position_lr_final,
                # lr_delay_mult=position_lr_delay_mult,
                # max_steps=position_lr_max_steps)
    # model.training_setup(1e-4, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8)
    # model.training_setup(1e-2, 0, 0, 0, 0, 0, 0, 0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)
    if dual_optim:
        scheduler_sgd = torch.optim.lr_scheduler.StepLR(opt_sgd, step_size=20, gamma=0.7)
    elif coord_desc:
        scheduler_mean = torch.optim.lr_scheduler.StepLR(opt_mean, step_size=20, gamma=0.7)
        scheduler_cov = torch.optim.lr_scheduler.StepLR(opt_cov, step_size=20, gamma=0.7)
    best_acc = 0
    for epoch in range(params['epochs']):
        first_iter = 0
        model.train()
        # progress_bar = tqdm(range(first_iter, len(train_loader)), desc="Training progress")
        first_iter += 1
        acc_num = 0
        acc_denom = 0
        loss_num = 0
        loss_denom = 0
        for iteration, (x, y) in enumerate(train_loader):
            points = provider.random_point_dropout(x.data.numpy())
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            x = torch.from_numpy(points)
            x, y = x.cuda(), y.cuda().long()
            if iteration % write_every == 0:
                # progress_bar.set_postfix({"Loss": f"{loss_num/loss_denom:.{7}f}", "Acc": f"{acc_num/acc_denom:.{7}f}"})
                # progress_bar.update(10)
                mean_path = os.path.join(gaussian_dirs, 'epoch%d_%d_mean' % (epoch, iteration))
                torch.save(model.feat.mu, mean_path)
                # mlflow.log_artifact(mean_path, 'vis/mean')
                with torch.no_grad():
                    cov = torch.linalg.inv(model.feat.compute_inv_cov())
                cov_path = os.path.join(gaussian_dirs, 'epoch%d_%d_cov' % (epoch, iteration))
                torch.save(cov, cov_path)
                # mlflow.log_artifact(cov_path, 'vis/cov')
                with torch.no_grad():
                    if use_mlflow:
                        if model.feat.mu.grad is not None:
                            mlflow.log_metric('mu_grad', torch.mean(torch.linalg.norm(model.feat.mu.grad, dim=-1)), step=(len(train_loader)*epoch+iteration))
                        if model.feat.l_triangs.grad is not None:
                            mlflow.log_metric('l_triangs_grad', torch.mean(torch.linalg.norm(model.feat.l_triangs.grad, dim=-1)), step=(len(train_loader)*epoch+iteration))

            loss_denom += 1
            opt.zero_grad()#set_to_none = True)
            if dual_optim:
                opt_sgd.zero_grad()#set_to_none = True)
            elif coord_desc:
                opt_mean.zero_grad()
                opt_cov.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            if coord_desc:
                if epoch % 3 == 0:
                    opt.step()
                elif epoch % 3 == 1:
                    opt_mean.step()
                else:
                    opt_cov.step()
            else:
                opt.step()
                if dual_optim:
                    opt_sgd.step()
            loss_num += loss
            loss_denom += 1
            correct = torch.sum(torch.argmax(preds, dim=-1) == y)
            acc_num += correct
            acc_denom += x.shape[0]
            acc = correct / x.shape[0]
            # if iteration % 100 == 0:
        train_loss = (loss_num/loss_denom).item()
        train_acc = (acc_num/acc_denom).item()
        if use_mlflow:
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_acc', train_acc, step=epoch)
            # gaussians.update_learning_rate(iteration)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            mean_correct = []
            class_acc = np.zeros((k, 3))
            val_loss_num = 0
            val_loss_denom = 0
            # progress_bar = tqdm(range(first_iter, len(test_loader)), desc="Validation progress")
            for iteration, (x, y) in enumerate(valid_loader):
                # x = x.transpose(2, 1)
                x, y = x.cuda(), y.cuda().long()
                # preds, _, _, _ = pointnet(x)
                preds = model(x)
                loss = criterion(preds, y)
                pred_choice = torch.argmax(preds, dim=-1)
                correct = torch.sum(pred_choice == y)

                for cat in np.unique(y.cpu()):
                    classacc = pred_choice[y == cat].eq(y[y == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(x[y == cat].size()[0])
                    class_acc[cat, 1] += 1

                c_correct = pred_choice.eq(y.long().data).cpu().sum()
                mean_correct.append(c_correct.item() / float(x.size()[0]))
                # needed for scheduler
                val_loss_denom += 1
                val_loss_num += loss
                val_loss = val_loss_num / val_loss_denom


        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        val_loss = (val_loss_num / val_loss_denom).item()

        if use_mlflow:
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_class_acc', class_acc, step=epoch)
            mlflow.log_metric('val_instance_acc', instance_acc, step=epoch)
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_class_acc': class_acc,
            'val_instance_acc': instance_acc,
        }

        if class_acc > best_acc:
            print("Saving model. Best validation accuracy. %0.7f > %0.7f" % (class_acc, best_acc))
            best_acc = class_acc
            save_dir = os.path.join(saved_dirs, 'best.pth')
            with open(save_dir, "wb") as f:
                torch.save(model.state_dict(), f)
            if use_mlflow:
                mlflow.log_artifact(save_dir, 'checkpoints')
        if dual_optim:
            scheduler_sgd.step(val_loss)
        scheduler.step(val_loss)
        save_dir = os.path.join(saved_dirs, '%d.pth' % epoch)
        with open(save_dir, "wb") as f:
            torch.save(model.state_dict(), f)
        if 'tune' in params:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save(
                    (model.state_dict(), opt.state_dict()), path
                )
                checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
                tune.report(metrics, checkpoint=checkpoint)
        if use_mlflow:
            mlflow.log_artifact(save_dir, 'checkpoints')

    print("Best class accuracy: %0.7f" % (best_acc))
    # pointnet = get_model(normal_channel=False).cuda()
    # pointnet = pointnet.eval()
    # for param in pointnet.parameters():
        # param.requires_grad_(False)
    # with open('pointnet_cls_no_tnet/checkpoints/best_model.pth', 'rb') as f:
        # state_dict = torch.load(f, weights_only=False)
    # pointnet.load_state_dict(state_dict['model_state_dict'])
    best_accuracy = 0
    mean_correct = []
    class_acc = np.zeros((k, 3))
    save_dir = os.path.join(saved_dirs, 'best.pth')
    with open(save_dir, "rb") as f:
        state_dict = torch.load(f, weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()
    if use_mlflow:
        mlflow.pytorch.log_model(model.to('cpu'), 'model', input_example=x.detach().cpu().numpy())
    model = model.to(device)
    with torch.no_grad():
        total_num = 0
        total_denom = 0
        # progress_bar = tqdm(range(first_iter, len(test_loader)), desc="Test progress")
        for iteration, (x, y) in enumerate(test_loader):
            total_denom += x.shape[0]
            # x = x.transpose(2, 1)
            x, y = x.cuda(), y.cuda().long()
            # preds, _, _, _ = pointnet(x)
            preds = model(x)
            loss = criterion(preds, y)
            pred_choice = torch.argmax(preds, dim=-1)
            total_num += torch.sum(pred_choice == y)

            for cat in np.unique(y.cpu()):
                classacc = pred_choice[y == cat].eq(y[y == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(x[y == cat].size()[0])
                class_acc[cat, 1] += 1
            c_correct = pred_choice.eq(y.long().data).cpu().sum()
            mean_correct.append(c_correct.item() / float(x.size()[0]))

            accuracy = total_num / total_denom
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        print("Final class accuracy: %0.7f" % (class_acc))
        print("Final instance accuracy: %0.7f" % (instance_acc))

    if use_mlflow:
        mlflow.log_metric('test_class_acc', class_acc)
        mlflow.log_metric('test_instance_acc', instance_acc)

    return class_acc, instance_acc


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--n', nargs='+', type=int, default=[10])
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--no_tnet", action='store_true', default=False)
    parser.add_argument("--e2e", action='store_true', default=False)
    parser.add_argument("--dual_optim", action='store_true', default=False)
    parser.add_argument("--normalized", action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default = 'modelnet')
    parser.add_argument('--global_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mean_lr', type=float, default=1e-2)
    parser.add_argument('--cov_lr', type=float, default=1e-2)
    parser.add_argument('--weighting_lr', type=float, default=1e-3)
    parser.add_argument('--cls_lr', type=float, default=1e-4)
    parser.add_argument("--pooling", type=str, default = 'max')
    parser.add_argument("--corruption", nargs="*", type=str, default = [])
    parser.add_argument("--severity", nargs="+", type=int, default = [1])
    parser.add_argument("--iid", nargs="+", type=int, default = [1])
    parser.add_argument("--kernel", type=str, choices=['gaussians', 'half_cos', 'raised_cos'], default='gaussians')
    parser.add_argument("--init_mode", type=str, choices=['random', 'layered', 'uniform'], default='random')
    parser.add_argument("--hook", type=str, choices=['standard', 'FIM', 'mahalanobis'], default='standard')
    parser.add_argument("--optimizer", type=str, choices=['adam', 'shampoo', 'soap'], default='adam')
    parser.add_argument("--ip", type=str, default = 'http://127.0.0.1')
    parser.add_argument("--port", type=str, default = '8080')
    parser.add_argument("--isotropic", action='store_true', default=False)
    parser.add_argument("--diagonal", action='store_true', default=False)
    parser.add_argument("--coord_desc", action='store_true', default=False)
    parser.add_argument("--tune_trials", type=int, default=0)
    parser.add_argument("--use-mlflow", "--use_mlflow", dest="use_mlflow", action='store_true', default=False, help="Enable MLflow logging and tracking")
    args = parser.parse_args(sys.argv[1:])
    args.primitive = 'gaussian'
    # args.save_iterations.append(args.iterations)

    print("Optimizing")# + args.model_path)
    if args.use_mlflow:
        remote_server_uri = '%s:%s' % (args.ip, args.port)
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment('/optimizers')
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.tune_trials > 0:
        def run_tuning(config):
            params = dict(vars(args))
            for key in config:
                params[key] = config[key]
            params['tune'] = True
            params['use_tnet'] = not params['no_tnet']
            params.pop('no_tnet')
            params.pop('ip')
            params.pop('port')
            if args.use_mlflow:
                with mlflow.start_run():
                    mlflow.log_params(params)
                    end_to_end(params)
            else:
                end_to_end(params)
        config = {
            'n_primitives': tune.grid_search(args.n),
            'hook': tune.grid_search(['standard', 'mahalanobis', 'FIM']),
            # 'batch_size': tune.sample_from(lambda _: 2**np.random.randint(3, 7)),
            'cls_lr': tune.loguniform(1e-6, 1e-3),
            'mean_lr': tune.loguniform(1e-6, 1e-1),
            'cov_lr': tune.loguniform(1e-6, 1e-1),
            'batch_size': args.batch_size,
            'num_trials': args.tune_trials,
            'epochs': args.epochs,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'driver_cwd': os.getcwd(),
            'dual_optim': tune.grid_search([True, False]),
            'isotropic': args.isotropic,
            'diagonal': args.diagonal,
            'coord_desc': args.coord_desc,
            'optimizer': args.optimizer,
        }
        scheduler = ASHAScheduler(\
            time_attr='training_iteration',
            max_t=config['epochs'],
            grace_period=1,
            reduction_factor=2)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(run_tuning),
                resources={'cpu': 8, 'gpu': 0.5}
            ),
            tune_config=tune.TuneConfig(
                metric='val_instance_acc',
                mode='max',
                scheduler=scheduler,
                num_samples=args.tune_trials,
            ),
            param_space=config,
        )
        tuner.fit()
    else:
        # args_dict = { 'num_point': 1024, 'use_uniform_sample': True, 'use_normals': False, 'num_category': 40 }
        # args = Namespace(**args_dict)
        # if 'modelnet40_c' in args.dataset
        # def run_tuning
        def run_exps(corruption, severity, iid):
            for trial in range(args.trials):
                for n in args.n:
                    # Log training parameters.
                    params = dict(vars(args))
                    params.pop('n')
                    params.pop('trials')
                    params.pop('detect_anomaly')
                    params['n_primitives'] = n
                    params['corruption'] = corruption
                    params['severity'] = severity
                    params['iid'] = iid
                    params['use_tnet'] = not params['no_tnet']
                    params.pop('no_tnet')
                    params.pop('ip')
                    params.pop('port')
                    if args.use_mlflow:
                        with mlflow.start_run():
                            mlflow.log_params(params)
                            class_acc, instance_acc = end_to_end(params)
                    else:
                        class_acc, instance_acc = end_to_end(params)

        if args.corruption is not None and len(args.corruption) > 0:
            for corruption in args.corruption:
                for severity in args.severity:
                    for iid in args.iid:
                        # os.makedirs(out_dir + '-corrupt-%s-severity-%d-iid-%d' % (corruption, severity, iid), exist_ok=True)
                        # run_exps(out_dir + '-corrupt-%s-severity-%d-iid-%d' % (corruption, severity, iid), corruption, severity, iid)
                        run_exps(corruption, severity, iid)
        else:
            run_exps(None, None, None)

    # All done
    print("\nTraining complete.")

