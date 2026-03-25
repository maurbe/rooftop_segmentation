import tomllib, os, torch, csv, json
from datetime import datetime
from typing import List, Dict, Optional
import torch.nn as nn
from .net import UNet
from .losses import build_loss, resolve_loss_config
from .dataloader import NPZSplitSegmentationDataset, boundary_mask_torch
from torch.utils.data import DataLoader
import random, numpy as np, shutil


class SegmentationModel:

    def __init__(self,
                 config_path: str,
                 set_id: int,
                 base_dir: str,
                 ):
        
        # Config and environment setup
        self.config_path = config_path
        with open(config_path, "rb") as f:
            self.cfg = tomllib.load(f)

        # Envornment setup
        self.set_id = set_id
        self._setup_environment(base_dir=base_dir, set_id=set_id)

        # Seed and device setup
        self.seed = int(self.cfg.get("experiment", {}).get("seed", 42))
        self._set_seed(self.seed)
        self.device = self._resolve_device()
        print(f"Using device: {self.device}")

        # Model setup
        self.model = self._build_model().to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M parameters.")
        opt_cfg = self.cfg.get("optimizer", {})
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # Training/validation objectives from [loss] config.
        loss_cfg = self.cfg.get("loss", {})
        self.use_edge_detection=bool(loss_cfg.get("use_edge_detection", False))
        train_loss_name, valid_loss_names = resolve_loss_config(loss_cfg)
        self.train_loss_name = train_loss_name
        self.valid_loss_names = valid_loss_names
        self.train_criterion = build_loss(self.train_loss_name, loss_cfg)
        self.valid_criteria = {
            name: build_loss(name, loss_cfg) for name in self.valid_loss_names
        }
        self.primary_val_metric_key = f"val_{self.valid_loss_names[0]}"
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        # Dataloader init
        self._build_dataloaders()

    
    def _setup_environment(self, base_dir: str, set_id: int):
        self.data_dir = self.cfg.get("data", {}).get("sets_root", "data/sets") + f"/set_{self.set_id}/"
        self.run_dir = os.path.join(base_dir, f"set_{set_id}")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")


    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(requested_device: Optional[str] = None) -> torch.device:
        """Resolve device safely with priority cuda -> mps -> cpu.
        If requested_device is provided, validate it and fallback to CPU when unavailable.
        """
        if requested_device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if requested_device == "mps":
            if not torch.backends.mps.is_built():
                print("MPS requested but not built in this PyTorch install. Falling back to CPU.")
                return torch.device("cpu")
            if not torch.backends.mps.is_available():
                print("MPS requested but unavailable on this machine/runtime. Falling back to CPU.")
                return torch.device("cpu")

        return torch.device(requested_device)

    def _build_model(self) -> nn.Module:
        model_cfg = self.cfg.get("model", {})
        model_name = model_cfg.get("name", "unet")
        in_channels = int(model_cfg.get("in_channels", 3))
        out_channels = int(model_cfg.get("out_channels", 1))
        base_channels = int(model_cfg.get("base_channels", 32))
        num_layers = int(model_cfg.get("num_layers", 7))
        dropout_p = float(model_cfg.get("dropout_p", 0.3))
        simple_encoder = bool(model_cfg.get("simple_encoder", False))

        if model_name != "unet":
            raise ValueError(f"Unsupported model.name: {model_name}")

        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )

    def _build_dataloaders(self):
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("training", {})

        batch_size  = int(train_cfg.get("batch_size", 8))
        num_workers = int(train_cfg.get("num_workers", 0))

        train_split_dir = self.data_dir + "train"
        validate_split_dir = self.data_dir + "validate"
        test_split_dir = self.data_dir + "test"
        holdout_split_dir = self.data_dir + "holdout"

        common_ds_kwargs = {
            "normalize_image": bool(data_cfg.get("normalize_image", True)),
            "binarize_mask": bool(data_cfg.get("binarize_mask", True)),
            "mask_threshold": float(data_cfg.get("mask_threshold", 0.5)),
            "apply_edge_maps": bool(data_cfg.get("apply_edge_maps", True)),
        }

        train_ds = NPZSplitSegmentationDataset(
            split_dir=train_split_dir,
            **common_ds_kwargs,
            apply_augmentations=bool(data_cfg.get("train_augment", True)),
            apply_albumentations=bool(data_cfg.get("train_albumentations", True)),
            apply_edge_detection=self.use_edge_detection,
            quadrant_crop_N=int(data_cfg.get("quadrant_crop_N", 1)),
        )

        val_ds = NPZSplitSegmentationDataset(
            split_dir=validate_split_dir,
            **common_ds_kwargs,
            apply_augmentations=False,
            apply_albumentations=False,
        )

        test_ds = NPZSplitSegmentationDataset(
            split_dir=test_split_dir,
            **common_ds_kwargs,
            apply_augmentations=False,
            apply_albumentations=False,

        )

        holdout_ds = NPZSplitSegmentationDataset(
            split_dir=holdout_split_dir,
            **common_ds_kwargs,
            apply_augmentations=False,
            apply_albumentations=False,
        )

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        self.holdout_dataset = holdout_ds

        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.holdout_loader = holdout_loader

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
            "val_loss": float(val_loss),
        }

        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(state, best_path)

    def _save_training_history(self, history: List[Dict[str, float]]) -> None:
        """Persist training curves for this run under log_dir."""
        if not history:
            return

        os.makedirs(self.log_dir, exist_ok=True)

        payload = {
            "config_path": self.config_path,
            "set_id": self.set_id,
            "run_dir": self.run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "history": history,
        }

        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        csv_fields = ["epoch", "train_loss"]
        for row in history:
            for key in row.keys():
                if key not in csv_fields:
                    csv_fields.append(key)

        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k) for k in csv_fields})

    def load_checkpoint(
            self, 
            which: str = "best",
    ):
        ckpt_path = os.path.join(self.checkpoint_dir, f"{which}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint '{which}' not found at {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        #print(f"Loaded checkpoint '{which}' from epoch {checkpoint.get('epoch', 'unknown')} with val_loss={checkpoint.get('val_loss', 'unknown'):.4f}")

    @torch.no_grad()
    def _evaluate_loader_metrics(
        self,
        loader: DataLoader,
        prefix: str,
        include_primary_alias: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        totals = {f"{prefix}_{name}": 0.0 for name in self.valid_loss_names}
        num_batches = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images)
            for name, criterion in self.valid_criteria.items():
                metric_name = f"{prefix}_{name}"
                metric_value = criterion(outputs, labels)
                totals[metric_name] += float(metric_value.item())

            num_batches += 1

        metrics = {}
        for key, value in totals.items():
            metrics[key] = value / num_batches if num_batches > 0 else float("inf")

        if include_primary_alias:
            primary_key = f"{prefix}_{self.valid_loss_names[0]}"
            metrics[f"{prefix}_loss"] = metrics.get(primary_key, float("inf"))

        return metrics

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        return self._evaluate_loader_metrics(
            loader=val_loader,
            prefix="val",
            include_primary_alias=True,
        )

    @torch.no_grad()
    def evaluate_test_metrics(self) -> Dict[str, float]:
        """Evaluate configured validation criteria on the test split."""
        raw_metrics = self._evaluate_loader_metrics(
            loader=self.test_loader,
            prefix="test",
            include_primary_alias=True,
        )

        compact_metrics = {
            name: raw_metrics[f"test_{name}"] for name in self.valid_loss_names
        }
        compact_metrics["test_loss"] = raw_metrics.get(
            f"test_{self.valid_loss_names[0]}",
            float("inf"),
        )
        return compact_metrics
    
    def train(self):

        # create necessary directories for checkpoints and logs
        # Clean previous run artifacts
        if os.path.isdir(self.run_dir):
            shutil.rmtree(self.run_dir)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # gather training config parameters
        train_cfg = self.cfg.get("training", {})
        num_epochs = int(train_cfg.get("epochs", 100))

        history = []
        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                edge_labels = batch["edge_labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss_label = self.train_criterion(outputs, labels)

                edge_labels_pred = boundary_mask_torch(outputs, kernel_size=3)
                loss_edges = self.train_criterion(edge_labels_pred, edge_labels)

                if self.use_edge_detection:
                    loss = loss_label + loss_edges  # Combine both losses
                else:
                    loss = loss_label  # Use only the primary loss

                loss.backward()
                self.optimizer.step()

                total_loss += float(loss_label.item()) # only track the primary loss for training curve purposes
                num_batches += 1

            avg_train_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            val_metrics = self.validate_epoch(self.val_loader)

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                **val_metrics,
            })

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self._save_checkpoint(epoch=epoch, val_loss=val_metrics["val_loss"], is_best=is_best)
            if (epoch + 1) % 100 == 0:
                val_summary = " | ".join(
                    [f"val_{name}={val_metrics[f'val_{name}']:.4f}" for name in self.valid_loss_names]
                )
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | {val_summary}"
                )
                
        self._save_training_history(history)


    @torch.no_grad()
    def predict_instance(
        self, 
        image_path: str,
        label_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        threshold: float = 0.5,
        ) -> Dict:
        self.model.eval()

        # only need the function method from the class
        image, label, mask, edge_labels, _has_label, _has_mask = self.val_dataset.load_instance_from_paths(
            image_path=image_path,
            label_path=label_path,
            mask_path=mask_path,
        )
        label = label[0] if _has_label else None

        x = torch.from_numpy(image[None, ...]).to(self.device)
        mask_2d = mask[0]

        probs = self.model(x)[0, 0].cpu().numpy()
        label_pred = (probs >= threshold)

        label_pred[mask_2d <= 0] = 0
        probs[mask_2d <= 0] = 0

        return {
            "image": np.transpose(image, (1, 2, 0)),
            "label_true": label if _has_label else None,
            "label_pred": label_pred,
            "probs": probs,
            "mask": mask_2d,
        }
    

    @torch.no_grad()
    def compute_metric_on_test_set(self, metric_fn) -> Dict[str, float]:
        self.model.eval()
        total_metric = 0.0
        num_samples = 0

        for batch in self.test_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            valid = batch["mask"].to(self.device)

            outputs = self.model(images)
            outputs = outputs >= 0.5  # binarize outputs at threshold 0.5
            
            # cast back to numpy
            outputs = outputs.cpu().numpy().astype(np.uint)
            labels = labels.cpu().numpy().astype(np.uint)
            valid = valid.cpu().numpy().astype(np.uint)

            # metric_fn can only take outputs, labels, need to mask manually and then just feed flattened arrays
            outputs_masked = outputs[valid > 0]
            labels_masked = labels[valid > 0]

            batch_metric = metric_fn(outputs_masked, labels_masked)
            total_metric += float(batch_metric)
            num_samples += 1

        avg_metric = total_metric / num_samples if num_samples > 0 else 0.0
        return avg_metric