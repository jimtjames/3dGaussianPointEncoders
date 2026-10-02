from __future__ import annotations

import argparse
import copy
import datetime as dt
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import provider
from datasets import ModelNetDataLoader, ScanObjectNN
from models import GaussianBasisPointNet


class PointNetTNet(nn.Module):
    """The input transform used by the legacy PointNet checkpoints."""

    def __init__(self, channel: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        premax = F.relu(self.bn3(self.conv3(x)))
        pooled = torch.amax(premax, dim=2)
        pooled = F.relu(self.bn4(self.fc1(pooled)))
        pooled = F.relu(self.bn5(self.fc2(pooled)))
        transform = self.fc3(pooled)
        identity = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9)
        transform = (transform + identity).reshape(-1, 3, 3)
        return transform, premax


class PointNetEncoder(nn.Module):
    """PointNet encoder with the module names used in legacy checkpoints."""

    def __init__(self, global_dim: int = 1024) -> None:
        super().__init__()
        self.stn = PointNetTNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, global_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(global_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transform, tnet_premax = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), transform).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        premax = self.bn3(self.conv3(x))
        return torch.amax(premax, dim=2), transform, tnet_premax, premax


class PointNetTeacher(nn.Module):
    """PointNet classifier used only as a frozen distillation teacher."""

    def __init__(self, k: int, global_dim: int = 1024) -> None:
        super().__init__()
        self.feat = PointNetEncoder(global_dim)
        self.fc1 = nn.Linear(global_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, transform, tnet_premax, premax = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = F.log_softmax(self.fc3(x), dim=1)
        return logits, transform, tnet_premax, premax


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def dataset_spec(dataset: str) -> tuple[int, torch.Tensor]:
    if dataset == "modelnet":
        return 40, torch.tensor([[-1.0, 1.0]] * 3)
    if dataset in {"scanobjectnn", "scanobjectnn-hard", "scanobj", "scanobj_hard"}:
        return 15, torch.tensor([[-5.0, 5.0]] * 3)
    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose modelnet, scanobjectnn, "
        "or scanobjectnn-hard"
    )


def load_teacher(
    checkpoint_path: str, k: int, global_dim: int, device: torch.device
) -> PointNetTeacher:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported teacher checkpoint format: {checkpoint_path}")
    state_dict = {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    teacher = PointNetTeacher(k=k, global_dim=global_dim)
    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The teacher checkpoint is incompatible with the requested dataset "
            f"(classes={k}) or --global_dim={global_dim}: {checkpoint_path}"
        ) from error
    teacher.requires_grad_(False)
    return teacher.to(device).eval()


def gaussian_point_features(encoder: nn.Module, points: torch.Tensor) -> torch.Tensor:
    """Return the Gaussian encoder projection before point-wise max pooling."""
    activations = encoder.likelihood(points)
    return encoder.weighting(activations)


def sample_clouds(
    batch_size: int,
    num_points: int,
    cloud_range: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    bounds = cloud_range.to(device=device)
    low, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    return low + torch.rand(batch_size, num_points, 3, device=device) * span


def encoder_parameter_groups(encoder: nn.Module, args: argparse.Namespace) -> list[dict]:
    return [
        {"params": [encoder.mu], "lr": args.mean_lr, "name": "mean"},
        {"params": [encoder.l_triangs], "lr": args.cov_lr, "name": "cov"},
        {
            "params": encoder.weighting.parameters(),
            "lr": args.weighting_lr,
            "weight_decay": args.wd_weighting,
            "name": "weighting",
        },
    ]


def make_optimizer(groups: list[dict], name: str) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.AdamW(groups, lr=0.0, eps=1e-15)
    if name == "shampoo":
        import torch_optimizer

        return torch_optimizer.Shampoo(groups, lr=1e-3)
    if name == "soap":
        from soap import SOAP

        return SOAP(groups, lr=1e-3)
    raise ValueError(f"Unknown optimizer {name!r}")


def log_metric(enabled: bool, key: str, value: float, step: int | None = None) -> None:
    if enabled:
        mlflow.log_metric(key, value, step=step)


def save_state(module: nn.Module, path: Path, use_mlflow: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)
    if use_mlflow:
        mlflow.log_artifact(str(path), artifact_path="checkpoints")


def register_gradient_hook(model: GaussianBasisPointNet, hook: str) -> None:
    if hook == "standard":
        return
    hook_name = "register_FIM_hook" if hook == "FIM" else "register_mahalanobis_hook"
    getattr(model.feat, hook_name)()
    if model.use_tnet:
        getattr(model.tnet.feat, hook_name)()


def copy_tnet_head(student: GaussianBasisPointNet, teacher: PointNetTeacher) -> None:
    mappings = (
        (student.tnet.fc1, teacher.feat.stn.fc1),
        (student.tnet.fc2, teacher.feat.stn.fc2),
        (student.tnet.fc3, teacher.feat.stn.fc3),
        (student.tnet.bn1, teacher.feat.stn.bn4),
        (student.tnet.bn2, teacher.feat.stn.bn5),
    )
    for destination, source in mappings:
        destination.load_state_dict(copy.deepcopy(source.state_dict()))


def copy_classifier_head(student: GaussianBasisPointNet, teacher: PointNetTeacher) -> None:
    for name in ("fc1", "fc2", "fc3", "bn1", "bn2"):
        destination = getattr(student, name)
        source = getattr(teacher, name)
        destination.load_state_dict(copy.deepcopy(source.state_dict()))


def run_feature_distillation(
    *,
    stage: str,
    encoder: nn.Module,
    target_fn: Any,
    input_fn: Any,
    steps: int,
    args: argparse.Namespace,
    cloud_range: torch.Tensor,
    device: torch.device,
) -> float:
    encoder.requires_grad_(True)
    encoder.train()
    optimizer = make_optimizer(encoder_parameter_groups(encoder, args), args.optimizer)
    running_loss = 0.0
    last_loss = float("nan")
    progress = tqdm(range(steps), desc=stage, unit="step")

    for step in progress:
        points = sample_clouds(
            args.distill_batch_size, args.distill_num_points, cloud_range, device
        )
        optimizer.zero_grad(set_to_none=True)
        student_input = input_fn(points)
        prediction = gaussian_point_features(encoder, student_input)
        with torch.no_grad():
            target = target_fn(points)
        loss = F.l1_loss(prediction, target)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach())
        running_loss += last_loss
        if (step + 1) % args.log_every == 0 or step + 1 == steps:
            window = args.log_every if (step + 1) % args.log_every == 0 else (step % args.log_every) + 1
            mean_loss = running_loss / window
            progress.set_postfix(loss=f"{mean_loss:.6g}")
            log_metric(args.use_mlflow, f"{stage}/loss", mean_loss, step)
            running_loss = 0.0

    return last_loss


def distill_tnet(
    model: GaussianBasisPointNet,
    teacher: PointNetTeacher,
    args: argparse.Namespace,
    cloud_range: torch.Tensor,
    device: torch.device,
) -> float:
    def target(points: torch.Tensor) -> torch.Tensor:
        _, premax = teacher.feat.stn(points.transpose(2, 1))
        return premax.transpose(2, 1)

    loss = run_feature_distillation(
        stage="tnet_distill",
        encoder=model.tnet.feat,
        target_fn=target,
        input_fn=lambda points: points,
        steps=args.tnet_distill_steps,
        args=args,
        cloud_range=cloud_range,
        device=device,
    )
    copy_tnet_head(model, teacher)
    return loss


def distill_main_encoder(
    model: GaussianBasisPointNet,
    teacher: PointNetTeacher,
    args: argparse.Namespace,
    cloud_range: torch.Tensor,
    device: torch.device,
) -> float:
    if model.use_tnet:
        model.tnet.requires_grad_(False)
        model.tnet.eval()

        def student_input(points: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return torch.bmm(points, model.tnet(points))

    else:
        student_input = lambda points: points

    def target(points: torch.Tensor) -> torch.Tensor:
        _, _, _, premax = teacher(points.transpose(2, 1))
        return premax.transpose(2, 1)

    loss = run_feature_distillation(
        stage="encoder_distill",
        encoder=model.feat,
        target_fn=target,
        input_fn=student_input,
        steps=args.encoder_distill_steps,
        args=args,
        cloud_range=cloud_range,
        device=device,
    )
    copy_classifier_head(model, teacher)
    return loss


def build_datasets(args: argparse.Namespace):
    if args.dataset == "modelnet":
        loader_args = argparse.Namespace(
            num_point=args.num_point,
            use_uniform_sample=True,
            use_normals=False,
            num_category=40,
        )
        common = {"root": args.data_path, "args": loader_args, "process_data": True}
        return (
            ModelNetDataLoader(split="train", **common),
            ModelNetDataLoader(split="valid", **common),
            ModelNetDataLoader(split="test", **common),
        )

    version = (
        "objectdataset_augmentedrot_scale75"
        if "hard" in args.dataset
        else "objectdataset"
    )
    common = {"root": args.scanobjectnn_path, "version": version}
    return (
        ScanObjectNN(subset="train", **common),
        ScanObjectNN(subset="valid", **common),
        ScanObjectNN(subset="test", **common),
    )


def build_finetune_optimizer(
    model: GaussianBasisPointNet, args: argparse.Namespace
) -> torch.optim.Optimizer:
    groups = encoder_parameter_groups(model.feat, args)
    classifier = [model.fc1, model.fc2, model.bn1, model.bn2]
    for module in classifier:
        groups.append({"params": module.parameters(), "lr": args.cls_lr})
    groups.append(
        {
            "params": model.fc3.parameters(),
            "lr": args.cls_lr,
            "weight_decay": args.wd_cls,
            "name": "classifier",
        }
    )
    if model.use_tnet:
        tnet_groups = encoder_parameter_groups(model.tnet.feat, args)
        for group in tnet_groups:
            group["name"] = f"tnet.{group['name']}"
        groups.extend(tnet_groups)
        for module in (
            model.tnet.fc1,
            model.tnet.fc2,
            model.tnet.fc3,
            model.tnet.bn1,
            model.tnet.bn2,
        ):
            groups.append({"params": module.parameters(), "lr": args.cls_lr})
    return make_optimizer(groups, args.optimizer)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data, valid_data, test_data = build_datasets(args)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_data, shuffle=True, drop_last=True, **common),
        DataLoader(valid_data, shuffle=False, drop_last=False, **common),
        DataLoader(test_data, shuffle=False, drop_last=False, **common),
    )


def prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, labels = batch
    if augment:
        array = points.detach().cpu().numpy()
        array = provider.random_point_dropout(array)
        array[:, :, :3] = provider.random_scale_point_cloud(array[:, :, :3])
        array[:, :, :3] = provider.shift_point_cloud(array[:, :, :3])
        points = torch.from_numpy(array)
    return points.float().to(device), labels.long().reshape(-1).to(device)


@torch.no_grad()
def evaluate(
    model: GaussianBasisPointNet,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for batch in loader:
        points, labels = prepare_batch(batch, device, augment=False)
        prediction = model(points)
        loss_sum += float(F.nll_loss(prediction, labels, reduction="sum"))
        choices = prediction.argmax(dim=1)
        total += labels.numel()
        correct += int((choices == labels).sum())
        for label in labels.unique():
            index = int(label)
            mask = labels == label
            class_correct[index] += (choices[mask] == labels[mask]).sum().cpu()
            class_total[index] += mask.sum().cpu()

    present = class_total > 0
    class_accuracy = (class_correct[present].float() / class_total[present]).mean()
    return {
        "loss": loss_sum / total,
        "instance_acc": correct / total,
        "class_acc": float(class_accuracy),
    }


def fine_tune(
    model: GaussianBasisPointNet,
    args: argparse.Namespace,
    output_dir: Path,
    num_classes: int,
    device: torch.device,
) -> dict[str, float]:
    train_loader, valid_loader, test_loader = make_loaders(args)
    model.requires_grad_(True)
    model.train()
    optimizer = build_finetune_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    best_class_acc = -float("inf")
    best_path = output_dir / "best.pth"

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"finetune {epoch + 1}/{args.epochs}")
        for batch in progress:
            points, labels = prepare_batch(batch, device, augment=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(points)
            loss = F.nll_loss(prediction, labels)
            loss.backward()
            optimizer.step()
            count = labels.numel()
            loss_sum += float(loss.detach()) * count
            correct += int((prediction.argmax(dim=1) == labels).sum())
            total += count

        train_loss = loss_sum / total
        train_acc = correct / total
        validation = evaluate(model, valid_loader, num_classes, device)
        metrics = {
            "finetune/train_loss": train_loss,
            "finetune/train_acc": train_acc,
            "finetune/val_loss": validation["loss"],
            "finetune/val_instance_acc": validation["instance_acc"],
            "finetune/val_class_acc": validation["class_acc"],
        }
        for key, value in metrics.items():
            log_metric(args.use_mlflow, key, value, epoch)
        print(
            f"epoch {epoch + 1}: train loss={train_loss:.6f}, "
            f"train acc={train_acc:.6f}, val instance acc="
            f"{validation['instance_acc']:.6f}, val class acc="
            f"{validation['class_acc']:.6f}"
        )
        if validation["class_acc"] > best_class_acc:
            best_class_acc = validation["class_acc"]
            torch.save(model.state_dict(), best_path)
        scheduler.step()

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, num_classes, device)
    for key, value in test_metrics.items():
        log_metric(args.use_mlflow, f"test/{key}", value)
    save_state(model, output_dir / "final.pth", args.use_mlflow)
    if args.use_mlflow:
        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
    print(
        f"test instance acc={test_metrics['instance_acc']:.7f}, "
        f"test class acc={test_metrics['class_acc']:.7f}"
    )
    return test_metrics


def run_experiment(args: argparse.Namespace, n_primitives: int, trial: int) -> dict[str, float]:
    set_seed(args.seed + trial)
    device = resolve_device(args.device)
    num_classes, cloud_range = dataset_spec(args.dataset)
    teacher = load_teacher(
        args.teacher_checkpoint, num_classes, args.global_dim, device
    )
    model = GaussianBasisPointNet(
        n_primitives=n_primitives,
        in_dim=3,
        cloud_range=cloud_range,
        init_mode=args.init_mode,
        global_dim=args.global_dim,
        k=num_classes,
        normalized=args.normalized,
        use_tnet=not args.no_tnet,
        isotropic=args.isotropic,
        diagonal=args.diagonal,
    ).to(device)
    register_gradient_hook(model, args.hook)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir) / (
        f"distill_{args.dataset}_n{n_primitives}_trial{trial}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {output_dir}")

    if model.use_tnet:
        print("stage 1/3: distilling T-Net encoder")
        tnet_loss = distill_tnet(model, teacher, args, cloud_range, device)
        log_metric(args.use_mlflow, "tnet_distill/final_loss", tnet_loss)
        save_state(model.tnet, output_dir / "tnet_distilled.pth", args.use_mlflow)
    else:
        print("stage 1/3: skipped (--no_tnet)")

    print("stage 2/3: distilling main encoder")
    encoder_loss = distill_main_encoder(model, teacher, args, cloud_range, device)
    log_metric(args.use_mlflow, "encoder_distill/final_loss", encoder_loss)
    save_state(model, output_dir / "encoder_distilled.pth", args.use_mlflow)

    print("stage 3/3: fine-tuning complete model")
    return fine_tune(model, args, output_dir, num_classes, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill PointNet into a Gaussian PointNet and fine-tune it"
    )
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--n", nargs="+", type=int, default=[10])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--no_tnet", action="store_true")
    parser.add_argument(
        "--dataset",
        choices=[
            "modelnet",
            "scanobjectnn",
            "scanobjectnn-hard",
            "scanobj",
            "scanobj_hard",
        ],
        default="modelnet",
    )
    parser.add_argument("--data_path", default="./data/modelnet40_normal_resampled/")
    parser.add_argument("--scanobjectnn_path", default="./data/ScanObjectNN/main_split")
    parser.add_argument("--global_dim", type=int, default=1024)
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mean_lr", type=float, default=1e-2)
    parser.add_argument("--cov_lr", type=float, default=1e-2)
    parser.add_argument("--weighting_lr", type=float, default=1e-3)
    parser.add_argument("--cls_lr", type=float, default=1e-4)
    parser.add_argument("--wd_cls", type=float, default=0.0)
    parser.add_argument("--wd_weighting", type=float, default=0.0)
    parser.add_argument("--tnet_distill_steps", type=int, default=30_000)
    parser.add_argument("--encoder_distill_steps", type=int, default=30_000)
    parser.add_argument("--distill_batch_size", type=int, default=32)
    parser.add_argument("--distill_num_points", type=int, default=512)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--init_mode", choices=["random", "layered", "uniform"], default="random"
    )
    parser.add_argument(
        "--hook", choices=["standard", "FIM", "mahalanobis"], default="standard"
    )
    parser.add_argument(
        "--optimizer", choices=["adam", "shampoo", "soap"], default="adam"
    )
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--isotropic", action="store_true")
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--lr_step_size", type=int, default=20)
    parser.add_argument("--lr_gamma", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="saved")
    parser.add_argument("--ip", default="http://127.0.0.1")
    parser.add_argument("--port", default="8080")
    parser.add_argument("--mlflow_experiment", default="/distillation")
    parser.add_argument(
        "--use-mlflow",
        "--use_mlflow",
        dest="use_mlflow",
        action="store_true",
        help="Enable MLflow parameter, metric, and checkpoint logging",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.isotropic and args.diagonal:
        parser.error("--isotropic and --diagonal are mutually exclusive")
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if not args.n or any(value <= 0 for value in args.n):
        parser.error("--n values must be positive")
    for name in (
        "epochs",
        "batch_size",
        "num_point",
        "distill_batch_size",
        "distill_num_points",
        "log_every",
        "tnet_distill_steps",
        "encoder_distill_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not Path(args.teacher_checkpoint).is_file():
        parser.error(f"teacher checkpoint does not exist: {args.teacher_checkpoint}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.use_mlflow:
        mlflow.set_tracking_uri(f"{args.ip}:{args.port}")
        mlflow.set_experiment(args.mlflow_experiment)

    for trial in range(args.trials):
        for n_primitives in args.n:
            params = vars(args).copy()
            params["n_primitives"] = n_primitives
            params["trial"] = trial
            if args.use_mlflow:
                with mlflow.start_run():
                    mlflow.log_params(params)
                    run_experiment(args, n_primitives, trial)
            else:
                run_experiment(args, n_primitives, trial)

    print("Training complete.")


if __name__ == "__main__":
    main()
