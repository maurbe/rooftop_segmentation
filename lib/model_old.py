import os
import random
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.dataloader import NPZSplitSegmentationDataset
from lib.losses import build_loss

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from lib.net import UNet


@dataclass
class CheckpointState:
    epoch: int
    best_val_loss: float


class SegmentationModel:
    """
    End-to-end trainer class:
    1) init: environment setup + model/optimizer/scheduler creation
    2) load_checkpoint: restore training state
    3) train: build dataloaders, train and validate each epoch
    4) validate_instance: run inference for one instance (+ optional metrics)
    """

    def __init__(self, config_path: str,
                 set_id: int,
                 run_dir: str,
                 prepare_environment: bool = True,
                 ):
        
        # Config and environment setup
        self.config_path = config_path
        self.cfg = self._load_toml(config_path)
        
        paths = self.cfg.get("paths", {})
        self.sets_root = paths.get("sets_root", "data/sets")
        self.set_id = set_id
        self._set_run_dir(run_dir)
        if prepare_environment:
            self._prepare_environment()


        # Seed and device setup
        self.seed = int(self.cfg.get("experiment", {}).get("seed", 42))
        self._set_seed(self.seed)


        # Set device with safe fallback to cpu
        self.device = self._resolve_device()
        print(f"Using device: {self.device}")


        # Model setup
        self.model = self._build_model().to(self.device)
        opt_cfg = self.cfg.get("optimizer", {})
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)


        # Training objectives
        loss_cfg = self.cfg.get("loss", {"name": "dice"})
        self.criterion = build_loss(loss_cfg)
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.train_dataset: Optional[NPZSplitSegmentationDataset] = None
        self.val_dataset: Optional[NPZSplitSegmentationDataset] = None
        self.test_dataset: Optional[NPZSplitSegmentationDataset] = None


    @staticmethod
    def _load_toml(path: str) -> Dict:
        with open(path, "rb") as f:
            return tomllib.load(f)

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

    def _prepare_environment(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _set_run_dir(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")

    def _build_model(self) -> nn.Module:
        model_cfg = self.cfg.get("model", {})
        model_name = model_cfg.get("name", "unet")
        in_channels = int(model_cfg.get("in_channels", 3))
        out_channels = int(model_cfg.get("out_channels", 1))
        base_channels = int(model_cfg.get("base_channels", 32))
        num_layers = int(model_cfg.get("num_layers", 3))
        dropout_p = float(model_cfg.get("dropout_p", 0.3))

        if model_name != "unet":
            raise ValueError(f"Unsupported model.name: {model_name}")

        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )

    def _resolve_set_split_dir(self, split_name: str) -> str:
        split_dir = os.path.join(self.sets_root, f"set_{self.set_id}", split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found for set_id={self.set_id}, split='{split_name}': {split_dir}. "
                "Run prepare_dataset.ipynb first or ensure the selected set exists in [ensemble].sets."
            )
        return split_dir

    def _build_dataloaders(self):
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("training", {})

        batch_size = int(train_cfg.get("batch_size", 8))
        num_workers = int(train_cfg.get("num_workers", 0))

        train_split_dir = self._resolve_set_split_dir("train")
        validate_split_dir = self._resolve_set_split_dir("validate")
        test_split_dir = self._resolve_set_split_dir("test")

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
        )

        val_ds = NPZSplitSegmentationDataset(
            split_dir=validate_split_dir,
            **common_ds_kwargs,
            apply_augmentations=True,
        )

        test_ds = NPZSplitSegmentationDataset(
            split_dir=test_split_dir,
            **common_ds_kwargs,
            apply_augmentations=True,
        )

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
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

        return train_loader, val_loader, test_loader

    
    def _get_eval_dataset(self, split: str = "validate") -> NPZSplitSegmentationDataset:
        if getattr(self, "val_dataset", None) is None or getattr(self, "test_dataset", None) is None:
            self._build_dataloaders()

        if split == "validate":
            return self.val_dataset
        if split == "test":
            return self.test_dataset
        raise ValueError("split must be either 'validate' or 'test'.")
    

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

        csv_fields = ["epoch", "train_loss", "val_loss", "val_iou"]
        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k) for k in csv_fields})

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        which: Optional[str] = "latest",
    ) -> CheckpointState:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{which}.pt")

        if not os.path.isfile(checkpoint_path):
            model_name = str(which)
            cfg_runs_root = self.cfg.get("paths", {}).get("run_dir", "runs")
            cfg_stem = os.path.splitext(os.path.basename(self.config_path))[0]
            set_name = f"set_{self.set_id}"
            candidates = []

            # Canonical per-set path for ensemble runs.
            candidates.append(os.path.join(cfg_runs_root, cfg_stem, set_name, "checkpoints", f"{model_name}.pt"))
            # Canonical single-run path.
            candidates.append(os.path.join(cfg_runs_root, cfg_stem, "checkpoints", f"{model_name}.pt"))

            # Legacy/stale in-memory path pattern: runs/set_i/checkpoints/model.pt
            # Redirect to runs/<config_stem>/set_i/checkpoints/model.pt when available.
            ckpt_dir = os.path.dirname(checkpoint_path)
            run_dir = os.path.dirname(ckpt_dir)
            if os.path.basename(ckpt_dir) == "checkpoints" and os.path.basename(run_dir).startswith("set_"):
                root_dir = os.path.dirname(run_dir)
                candidates.append(os.path.join(root_dir, cfg_stem, os.path.basename(run_dir), "checkpoints", f"{model_name}.pt"))

            resolved_path = None
            for candidate in candidates:
                if candidate and os.path.isfile(candidate):
                    resolved_path = candidate
                    break

            if resolved_path is None:
                raise FileNotFoundError(
                    "Checkpoint not found. Tried: "
                    + ", ".join([checkpoint_path] + candidates)
                )

            checkpoint_path = resolved_path

            # Keep trainer directories coherent with the recovered checkpoint location.
            self.checkpoint_dir = os.path.dirname(checkpoint_path)
            self.run_dir = os.path.dirname(self.checkpoint_dir)
            self.log_dir = os.path.join(self.run_dir, "logs")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        return CheckpointState(epoch=self.start_epoch, best_val_loss=self.best_val_loss)

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total = 0

        for batch in val_loader:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)
            valid = batch["mask"].to(self.device)

            preds = self.model(x)
            loss = self.criterion(preds, y, mask=valid)
            total_loss += float(loss.item()) * x.size(0)

            preds_bin = (preds > 0.5).float()
            valid_bin = (valid > 0.5).float()
            preds_bin = preds_bin * valid_bin
            y_masked = y * valid_bin
            inter = (preds_bin * y_masked).sum(dim=(1, 2, 3))
            union = (preds_bin + y_masked - preds_bin * y_masked).sum(dim=(1, 2, 3)).clamp(min=1.0)
            iou = (inter / union).mean().item()
            total_iou += float(iou) * x.size(0)
            total += x.size(0)

        return {
            "val_loss": total_loss / max(total, 1),
            "val_iou": total_iou / max(total, 1),
        }

    def train(self):
        
        train_cfg = self.cfg.get("training", {})
        max_epochs = int(train_cfg.get("epochs", 20))

        train_loader, val_loader, _test_loader = self._build_dataloaders()

        history = []
        for epoch in range(self.start_epoch, max_epochs):
            self.model.train()
            running_loss = 0.0
            n_seen = 0

            for batch in train_loader:
                x = batch["image"].to(self.device)
                y = batch["label"].to(self.device)
                valid = batch["mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(x)
                loss = self.criterion(preds, y, mask=valid)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item()) * x.size(0)
                n_seen += x.size(0)

            train_loss = running_loss / max(n_seen, 1)
            val_metrics = self.validate_epoch(val_loader=val_loader)
            val_loss = val_metrics["val_loss"]

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self._save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=is_best)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_metrics["val_iou"],
            }
            history.append(row)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_iou={val_metrics['val_iou']:.4f}"
                )

        self._save_training_history(history)
        return history

    @torch.no_grad()
    def predict_instance(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        split: str = "test",
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:

        self.model.eval()

        eval_dataset = self._get_eval_dataset(split=split)
        image, _label, mask, _has_label, _has_mask = eval_dataset.load_instance_from_paths(
            image_path=image_path,
            label_path=label_path,
            mask_path=mask_path,
        )
        x = torch.from_numpy(image[None, ...]).to(self.device)
        mask_2d = mask[0]

        probs = self.model(x)[0, 0].cpu().numpy()
        labels_pred = (probs >= threshold)

        labels_pred[mask_2d <= 0] = 0
        probs[mask_2d <= 0] = 0

        return {
            "image": np.transpose(image, (1, 2, 0)),
            "labels_pred": labels_pred,
            "probs": probs,
            "mask": mask_2d,
        }

    @torch.no_grad()
    def validate_instance(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        split: str = "validate",
        threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, float]:
        pred = self.predict_instance(
            image_path=image_path,
            label_path=label_path,
            mask_path=mask_path,
            split=split,
            threshold=threshold,
        )

        out = {
            "mean_prob": float(np.mean(pred["probs"])),
        }

        eval_dataset = self._get_eval_dataset(split=split)
        image, gt_label, _mask, has_label, _has_mask = eval_dataset.load_instance_from_paths(
            image_path=image_path,
            label_path=label_path,
            mask_path=mask_path,
        )

        _ = image  # keep signature symmetry; image already used in prediction

        if has_label:
            gt = (gt_label >= 0.5).astype(np.uint8)
            pm = pred["labels_pred"].astype(np.uint8)
            inter = np.logical_and(pm == 1, gt == 1).sum()
            union = np.logical_or(pm == 1, gt == 1).sum()
            out["iou"] = float(inter / max(union, 1))
            out["has_label"] = True
        else:
            out["has_label"] = False

        return out

class SegmentationEnsemble:
    """Manage multiple fold trainers from one config and perform voting."""

    def __init__(
        self,
        config_path: str,
        sets: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)

        ensemble_cfg = cfg.get("ensemble", {})
        if sets is None:
            sets = list(ensemble_cfg.get("sets", []))
        if threshold is None:
            threshold = float(ensemble_cfg.get("threshold", 0.5))

        if len(sets) == 0:
            raise ValueError("No sets provided. Add [ensemble].sets in TOML or pass sets explicitly.")

        self.cfg = cfg
        self.config_path = config_path
        self.sets = sets
        self.threshold = float(threshold)
        self.models: List[SegmentationModel] = []

        paths = cfg.get("paths", {})
        runs_root = paths.get("run_dir", "runs")
        run_name = os.path.splitext(os.path.basename(config_path))[0]
        base_run_dir = os.path.join(runs_root, run_name)

        #  LEGACY??Keep root clean: trainers own only set-specific subdirectories.
        for folder_name in ("checkpoints", "logs"):
            stale_path = os.path.join(base_run_dir, folder_name)
            if os.path.isdir(stale_path):
                shutil.rmtree(stale_path)

        for set_name in sets:
            set_id = self._parse_set_name(set_name)
            set_run_dir = os.path.join(base_run_dir, f"set_{set_id}")
            trainer = SegmentationModel(
                config_path=config_path,
                set_id=set_id,
                run_dir=set_run_dir,
                prepare_environment=True,
            )

            self.models.append(trainer)

    @staticmethod
    def _parse_set_name(set_name: str) -> int:
        if not str(set_name).startswith("set_"):
            raise ValueError(f"Invalid set name '{set_name}'. Expected format like 'set_0'.")
        try:
            return int(str(set_name).split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Could not parse set id from '{set_name}'.") from exc

    def train_all(self) -> Dict[str, List[Dict[str, float]]]:
        """Sequentially train all models, each on its configured set_i."""
        histories: Dict[str, List[Dict[str, float]]] = {}
        for i, model in enumerate(self.models):
            set_name = f"set_{model.set_id}"
            print(f"[Ensemble] Training model {i} on {set_name}")
            histories[set_name] = model.train()
        return histories

    def load_all_checkpoints(self, which: str = "best") -> None:
        for i, model in enumerate(self.models):
            print(f"[Ensemble] Loading {model} checkpoint for model {i} (set_{model.set_id})")
            model.load_checkpoint(which=which)

    
    @staticmethod
    def _collect_npz_paths(split_dir: str) -> List[str]:
        return [p.path for p in sorted(os.scandir(split_dir), key=lambda e: e.name) if p.is_file() and p.name.endswith(".npz")]

    def _predict_single_model_on_split(
        self,
        trainer: SegmentationModel,
        split: str,
        threshold: float,
        max_samples: Optional[int] = None,
        set_name: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, np.ndarray]]]:
        if set_name is None:
            split_dir = trainer._resolve_set_split_dir(split)
        else:
            set_id = self._parse_set_name(set_name)
            split_dir = os.path.join(trainer.sets_root, f"set_{set_id}", split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

        npz_paths = self._collect_npz_paths(split_dir)
        if max_samples is not None:
            npz_paths = npz_paths[: max(0, int(max_samples))]

        results = []
        for p in npz_paths:
            pred = trainer.predict_instance(image_path=p, split=split, threshold=threshold)
            results.append({
                "path": p,
                "image": pred["image"],
                "labels_pred": pred["labels_pred"],
                "probs": pred["probs"],
                "mask": pred["mask"],
            })

        return {"split_dir": split_dir, "results": results}


    @torch.no_grad()
    def validate_split(
        self,
        split: str = "validate",
        model: int = 0,
        threshold: Optional[float] = None,
        set_name: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict:

        if model < 0 or model >= len(self.trainers):
            raise IndexError(f"Model index {model} out of range [0, {len(self.trainers)-1}].")

        trainer = self.trainers[model]
        vote_threshold = self.threshold if threshold is None else float(threshold)

        if set_name is None:
            split_dir = trainer._resolve_set_split_dir(split)
            eval_set_name = f"set_{trainer.set_id}"
        else:
            set_id = self._parse_set_name(set_name)
            split_dir = os.path.join(trainer.sets_root, f"set_{set_id}", split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            eval_set_name = set_name

        npz_paths = self._collect_npz_paths(split_dir)
        if max_samples is not None:
            npz_paths = npz_paths[: max(0, int(max_samples))]

        results = []
        for p in npz_paths:
            metrics = trainer.validate_instance(
                image_path=p,
                split=split,
                threshold=vote_threshold,
            )
            results.append({
                "path": p,
                **metrics,
            })

        return {
            "mode": "single",
            "model_index": model,
            "set_name": eval_set_name,
            "split": split,
            "split_dir": split_dir,
            "results": results,
        }
    
    
    @torch.no_grad()
    def predict_split(
        self,
        split: str = "test",
        model: Union[int, str] = 0,
        threshold: Optional[float] = None,
        set_name: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict:

        if model == "all":
            if set_name is None:
                raise ValueError("When model='all', provide set_name (e.g. 'set_2') for a common evaluation split.")

            # Use paths from the first trainer and run all models on the same sample list.
            reference_trainer = self.models[0]
            set_id = self._parse_set_name(set_name)
            split_dir = os.path.join(reference_trainer.sets_root, f"set_{set_id}", split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            npz_paths = self._collect_npz_paths(split_dir)
            if max_samples is not None:
                npz_paths = npz_paths[: max(0, int(max_samples))]

            voted_results = []
            for p in npz_paths:
                member_probs = []
                member_preds = []
                base_image = None
                base_mask = None

                for trainer in self.models:
                    pred = trainer.predict_instance(image_path=p, split=split, threshold=threshold)
                    member_probs.append(pred["probs"])
                    member_preds.append(pred["labels_pred"].astype(np.uint8))
                    if base_image is None:
                        base_image = pred["image"]
                        base_mask = pred["mask"]

                probs_stack = np.stack(member_probs, axis=0)
                avg_probs = np.mean(probs_stack, axis=0)
                labels_pred = (avg_probs >= threshold)

                if base_mask is not None:
                    labels_pred[base_mask <= 0] = 0
                    avg_probs[base_mask <= 0] = 0

                voted_results.append({
                    "path": p,
                    "image": base_image,
                    "labels_pred": labels_pred,
                    "probs": avg_probs,
                    "mask": base_mask,
                    "member_probs": probs_stack,
                    "member_preds": np.stack(member_preds, axis=0),
                })

            return {
                "mode": "all",
                "split": split,
                "set_name": set_name,
                "split_dir": split_dir,
                "results": voted_results,
            }

        # model-specific prediction on its own set_i by default.
        if not isinstance(model, int):
            raise ValueError("model must be an integer index or 'all'.")
        if model < 0 or model >= len(self.models):
            raise IndexError(f"Model index {model} out of range [0, {len(self.models)-1}].")

        trainer = self.models[model]
        payload = self._predict_single_model_on_split(
            trainer=trainer,
            split=split,
            threshold=threshold,
            max_samples=max_samples,
            set_name=set_name,
        )
        return {
            "mode": "single",
            "model_index": model,
            "set_name": f"set_{trainer.set_id}" if set_name is None else set_name,
            "split": split,
            **payload,
        }
    
    @torch.no_grad()
    def predict_and_vote(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        vote_threshold = self.threshold if threshold is None else float(threshold)

        all_probs = []
        all_preds = []
        base_image = None
        base_mask = None

        for trainer in self.models:
            pred = trainer.predict_instance(
                image_path=image_path,
                mask_path=mask_path,
                threshold=vote_threshold,
            )
            all_probs.append(pred["probs"])
            all_preds.append(pred["labels_pred"].astype(np.uint8))
            if base_image is None:
                base_image = pred["image"]
                base_mask = pred["mask"]

        probs_stack = np.stack(all_probs, axis=0)
        mean_probs = np.mean(probs_stack, axis=0)
        labels_pred = (mean_probs >= vote_threshold)

        if base_mask is not None:
            labels_pred[base_mask <= 0] = 0
            mean_probs[base_mask <= 0] = 0

        return {
            "image": base_image,
            "mask": base_mask,
            "labels_pred": labels_pred,
            "probs": mean_probs,
            "member_probs": probs_stack,
            "member_preds": np.stack(all_preds, axis=0),
        }
