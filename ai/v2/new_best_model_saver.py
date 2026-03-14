"""
new_best_model_saver.py
=======================
Enhanced model saving functionality with anatomical grounding evaluation.

Features:
- Save models with comprehensive metadata
- Include anatomical evaluation metrics
- Track model improvements over training
- Save best models based on multiple criteria
"""

import os
import torch
import json
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path


class AnatomicalModelSaver:
    """
    Enhanced model saver that includes anatomical grounding evaluation.
    """

    def __init__(self, save_dir="models", experiment_name="hcct_anatomical"):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.save_dir.mkdir(exist_ok=True)

        # Tracking files
        self.models_log_path = self.save_dir / "models_log.json"
        self.best_models_path = self.save_dir / "best_models.json"

        # Initialize logs if they don't exist
        if not self.models_log_path.exists():
            self._init_logs()

    def _init_logs(self):
        """Initialize model tracking logs."""
        initial_log = {
            "experiment_name": self.experiment_name,
            "created": datetime.now().isoformat(),
            "models": [],
            "best_models": {
                "accuracy": None,
                "f1_score": None,
                "anatomical_grounding": None,
                "clinical_relevance": None,
                "anti_shortcut": None
            }
        }

        with open(self.models_log_path, 'w') as f:
            json.dump(initial_log, f, indent=2)

    def save_model_with_anatomical_evaluation(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        anatomical_metrics=None,
        validation_data=None,
        checkpoint_name=None
    ):
        """
        Save model with comprehensive anatomical evaluation.

        Parameters
        ----------
        model : nn.Module
            Model to save
        optimizer : torch.optim.Optimizer
            Optimizer state
        epoch : int
            Training epoch
        metrics : dict
            Standard metrics (accuracy, f1, loss, etc.)
        anatomical_metrics : dict, optional
            Anatomical evaluation results
        validation_data : dict, optional
            Validation data for reproducibility
        checkpoint_name : str, optional
            Custom checkpoint name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if checkpoint_name is None:
            checkpoint_name = f"model_epoch_{epoch}_{timestamp}"

        checkpoint_path = self.save_dir / f"{checkpoint_name}.pth"

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": timestamp,
            "experiment_name": self.experiment_name
        }

        if anatomical_metrics:
            checkpoint["anatomical_metrics"] = anatomical_metrics

        if validation_data:
            checkpoint["validation_data"] = validation_data

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update logs
        self._update_logs(checkpoint_name, checkpoint)

        # Check if this is a new best model
        self._check_and_update_best_models(checkpoint_name, metrics, anatomical_metrics)

        print(f"Model saved: {checkpoint_path}")
        return checkpoint_path

    def _update_logs(self, checkpoint_name, checkpoint):
        """Update the models log with new checkpoint info."""
        # Load current log
        with open(self.models_log_path, 'r') as f:
            log_data = json.load(f)

        # Add new model entry
        model_entry = {
            "name": checkpoint_name,
            "epoch": checkpoint["epoch"],
            "timestamp": checkpoint["timestamp"],
            "metrics": checkpoint["metrics"],
            "path": str(self.save_dir / f"{checkpoint_name}.pth")
        }

        if "anatomical_metrics" in checkpoint:
            model_entry["anatomical_metrics"] = checkpoint["anatomical_metrics"]

        log_data["models"].append(model_entry)

        # Save updated log
        with open(self.models_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _check_and_update_best_models(self, checkpoint_name, metrics, anatomical_metrics):
        """Check if this model is best in any category and update tracking."""
        # Load current best models
        with open(self.models_log_path, 'r') as f:
            log_data = json.load(f)

        best_models = log_data["best_models"]
        updated = False

        # Check standard metrics
        if best_models["accuracy"] is None or metrics.get("val_acc", 0) > best_models["accuracy"]["value"]:
            best_models["accuracy"] = {
                "name": checkpoint_name,
                "value": metrics.get("val_acc", 0),
                "epoch": metrics.get("epoch", 0)
            }
            updated = True

        if best_models["f1_score"] is None or metrics.get("val_f1", 0) > best_models["f1_score"]["value"]:
            best_models["f1_score"] = {
                "name": checkpoint_name,
                "value": metrics.get("val_f1", 0),
                "epoch": metrics.get("epoch", 0)
            }
            updated = True

        # Check anatomical metrics
        if anatomical_metrics:
            overall_scores = anatomical_metrics.get("overall_scores", {})

            if (best_models["anatomical_grounding"] is None or
                overall_scores.get("anatomical_grounding_score", -1) > best_models["anatomical_grounding"]["value"]):
                best_models["anatomical_grounding"] = {
                    "name": checkpoint_name,
                    "value": overall_scores.get("anatomical_grounding_score", 0),
                    "epoch": metrics.get("epoch", 0)
                }
                updated = True

            if (best_models["clinical_relevance"] is None or
                anatomical_metrics.get("clinical_relevance", {}).get("clinical_score", 0) > best_models["clinical_relevance"]["value"]):
                best_models["clinical_relevance"] = {
                    "name": checkpoint_name,
                    "value": anatomical_metrics["clinical_relevance"]["clinical_score"],
                    "epoch": metrics.get("epoch", 0)
                }
                updated = True

            if (best_models["anti_shortcut"] is None or
                (1 - anatomical_metrics.get("shortcut_detection", {}).get("shortcut_score", 1)) > best_models["anti_shortcut"]["value"]):
                best_models["anti_shortcut"] = {
                    "name": checkpoint_name,
                    "value": 1 - anatomical_metrics["shortcut_detection"]["shortcut_score"],
                    "epoch": metrics.get("epoch", 0)
                }
                updated = True

        if updated:
            # Save updated best models
            with open(self.models_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)

            print(f"Updated best models for {checkpoint_name}")

    def save_best_model_symlinks(self):
        """Create symlinks to best models for easy access."""
        with open(self.models_log_path, 'r') as f:
            log_data = json.load(f)

        best_models = log_data["best_models"]

        for criterion, best_info in best_models.items():
            if best_info is not None:
                source_path = self.save_dir / f"{best_info['name']}.pth"
                link_path = self.save_dir / f"best_{criterion}.pth"

                if source_path.exists():
                    # Remove existing link if it exists
                    if link_path.exists():
                        link_path.unlink()

                    # Create symlink
                    try:
                        link_path.symlink_to(source_path)
                        print(f"Created symlink: {link_path} -> {source_path}")
                    except OSError:
                        # Symlinks might not be supported on Windows
                        print(f"Could not create symlink for {criterion}")

    def get_model_history(self):
        """Get DataFrame with model training history."""
        with open(self.models_log_path, 'r') as f:
            log_data = json.load(f)

        models = log_data["models"]
        df = pd.DataFrame(models)

        # Expand metrics columns
        if not df.empty:
            metrics_df = pd.json_normalize(df["metrics"])
            anatomical_df = pd.json_normalize(df.get("anatomical_metrics", [{}] * len(df)))

            df = pd.concat([df.drop(["metrics", "anatomical_metrics"], axis=1), metrics_df, anatomical_df], axis=1)

        return df

    def load_best_model(self, criterion="f1_score", model_class=None, device="cuda"):
        """Load the best model for a given criterion."""
        with open(self.models_log_path, 'r') as f:
            log_data = json.load(f)

        best_info = log_data["best_models"].get(criterion)
        if best_info is None:
            raise ValueError(f"No best model found for criterion: {criterion}")

        checkpoint_path = self.save_dir / f"{best_info['name']}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if model_class:
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            return model, checkpoint
        else:
            return checkpoint

    def export_training_summary(self, output_path=None):
        """Export training summary as CSV."""
        if output_path is None:
            output_path = self.save_dir / "training_summary.csv"

        df = self.get_model_history()
        df.to_csv(output_path, index=False)
        print(f"Training summary exported to: {output_path}")
        return df


def save_model_with_anatomical_checkpoint(
    model,
    optimizer,
    epoch,
    val_metrics,
    anatomical_evaluation,
    save_dir="models",
    experiment_name="hcct_anatomical"
):
    """
    Convenience function to save model with anatomical evaluation.

    Parameters
    ----------
    model : nn.Module
        The model
    optimizer : torch.optim.Optimizer
        Optimizer
    epoch : int
        Current epoch
    val_metrics : dict
        Validation metrics
    anatomical_evaluation : dict
        Anatomical evaluation results
    save_dir : str
        Save directory
    experiment_name : str
        Experiment name

    Returns
    -------
    str
        Path to saved checkpoint
    """
    saver = AnatomicalModelSaver(save_dir, experiment_name)

    return saver.save_model_with_anatomical_evaluation(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=val_metrics,
        anatomical_metrics=anatomical_evaluation
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the saver
    saver = AnatomicalModelSaver()

    # Mock data
    mock_metrics = {
        "val_acc": 0.85,
        "val_f1": 0.82,
        "val_loss": 0.45,
        "epoch": 10
    }

    mock_anatomical = {
        "overall_scores": {
            "anatomical_grounding_score": 0.65,
            "adc_score": 0.12
        },
        "clinical_relevance": {"clinical_score": 0.7},
        "shortcut_detection": {"shortcut_score": 0.3}
    }

    # This would normally be called during training
    # saver.save_model_with_anatomical_evaluation(
    #     model=my_model,
    #     optimizer=my_optimizer,
    #     epoch=10,
    #     metrics=mock_metrics,
    #     anatomical_metrics=mock_anatomical
    # )

    print("AnatomicalModelSaver initialized. Use during training to save models with anatomical evaluation.")
