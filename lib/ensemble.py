import tomllib, os, torch
import numpy as np

from typing import Dict, List, Optional
from .model import SegmentationModel


class SegmentationEnsemble:

    def __init__(
            self, 
            config_path: str,
    ):
        
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)

        runs_root = cfg.get("paths", {}).get("run_dir", "runs")
        run_name = os.path.splitext(os.path.basename(config_path))[0]
        base_run_dir = os.path.join(runs_root, run_name)
        
        sets = list(cfg.get("ensemble", {}).get("sets", {}))
        threshold = cfg.get("ensemble", {}).get("threshold", 0.5)
        if len(sets) == 0:
            raise ValueError("No sets defined in the configuration file.")
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        
        self.cfg = cfg
        self.config_path = config_path
        self.sets = sets
        self.threshold = threshold

        self.models = []
        for set_name in sets:
            set_id = self._parse_set_name(set_name)
            trainer = SegmentationModel(
                config_path=config_path,
                set_id=set_id,
                base_dir=base_run_dir,
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


    def train_all(self) -> None:
        for i, model in enumerate(self.models):
            set_name = f"set_{model.set_id}"
            print(f"\n[Ensemble] Training model {i} on {set_name}")
            model.train()
    

    def load_all_checkpoints(self, which: str = "best") -> None:
        for i, model in enumerate(self.models):
            print(f"[Ensemble] Loading {which} checkpoint for model {i}")
            model.load_checkpoint(which=which)


    @staticmethod
    def _collect_npz_paths(split_dir: str) -> List[str]:
        return [p.path for p in sorted(os.scandir(split_dir), key=lambda e: e.name) if p.is_file() and p.name.endswith(".npz")]

    @torch.no_grad()
    def predict_single_model_on_split(
        self,
        split: str = "test",
        model_idx: int = None,
        max_samples: Optional[int] = None,
    ) -> Dict:
        
        if model_idx < 0 or model_idx >= len(self.models):
            raise IndexError(f"Model index {model_idx} out of range [0, {len(self.models)-1}].")
        if split not in ("train", "validate", "test", "holdout"):
            raise ValueError(f"Invalid split '{split}'. Expected one of: 'train', 'validate', 'test', 'holdout'.")
        
        model = self.models[model_idx]

        split_dir = model.data_dir + f"{split}"
        npz_paths = self._collect_npz_paths(split_dir)
        if max_samples is not None:
            npz_paths = npz_paths[: max(0, int(max_samples))]



        results =[]
        for p in npz_paths:
            pred = model.predict_instance(image_path=p)
            results.append({
                "path": p,
                "image": pred["image"],
                "label_true": pred["label_true"],
                "label_pred": pred["label_pred"],
                "probs": pred["probs"],
                "mask": pred["mask"],
            })

        return results
    

    @torch.no_grad()
    def predict_and_vote(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> Dict:
        
        if split not in ("train", "validate", "test", "holdout"):
            raise ValueError(f"Invalid split '{split}'. Expected one of: 'train', 'validate', 'test', 'holdout'.")
        
        all_model_results = []
        for i in range(len(self.models)):
            model_results = self.predict_single_model_on_split(
                split=split,
                model_idx=i,
                max_samples=max_samples,
            )
            all_model_results.append(model_results)

        
        # now we need to combine the predictions from all models for each sample
        # we dont need the predicted labels from each model, just the probabilities, and we will apply the threshold to the average probability across models
        combined_results = []
        for sample_idx in range(len(all_model_results[0])):
            sample_results = [model_results[sample_idx] for model_results in all_model_results]
            image = sample_results[0]["image"]
            label_true = sample_results[0]["label_true"]
            mask = sample_results[0]["mask"]

            # average probabilities across models
            probs_stack = np.stack([r["probs"] for r in sample_results], axis=0)
            avg_probs = np.mean(probs_stack, axis=0)

            # apply threshold to get final predicted label
            label_pred = (avg_probs >= self.threshold).astype(np.uint8)
            label_pred[mask <= 0] = 0

            combined_results.append({
                "path": sample_results[0]["path"],
                "image": image,
                "label_true": label_true,
                "label_pred": label_pred,
                "probs": avg_probs,
                "mask": mask,
            })
        return combined_results
    

    @torch.no_grad()
    def compute_metrics(
        self,
        split: str = "test",
        include_per_set: bool = True,
    ) -> Dict:
        if split != "test":
            raise ValueError("compute_metrics currently supports split='test' only.")

        per_set = []
        metric_names = None

        for model in self.models:
            metrics = model.evaluate_test_metrics()
            if metric_names is None:
                metric_names = list(metrics.keys())
            else:
                # Keep stable order from first model and append unseen metrics.
                for key in metrics.keys():
                    if key not in metric_names:
                        metric_names.append(key)

            per_set.append(
                {
                    "set_name": f"set_{model.set_id}",
                    "metrics": metrics,
                }
            )

        metric_names = metric_names or []
        metrics_mean = {}
        metrics_std = {}
        for name in metric_names:
            values = [entry["metrics"].get(name, np.nan) for entry in per_set]
            values = np.asarray(values, dtype=np.float64)
            metrics_mean[name] = float(np.nanmean(values))
            metrics_std[name] = float(np.nanstd(values))

        out = {
            "config_path": self.config_path,
            "split": split,
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
        }
        if include_per_set:
            out["per_set"] = per_set
        return out