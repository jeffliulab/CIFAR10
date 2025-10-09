import os, torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Checkpointer:
    path: str
    device: torch.device

    def save(self, epoch: int, model, optimizer=None, hist=None, config=None):
        """
        Atomically save a checkpoint to `path`.
        This reduces the risk of a half-written file if the job is killed mid-save.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if hist is not None:
            payload["hist"] = hist
        if config is not None:
            payload["config"] = config

        tmp = self.path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.path)

    def resume(self, model, optimizer=None, hist=None):
        """
        Resume training state from `ckpt_path` if it exists.

        Restores:
          - model parameters
          - optimizer state (if provided)
          - RNG states (CPU + all CUDA devices, if available)
          - last finished epoch
          - training/validation history

        Returns:
          start_epoch (int): last finished epoch number (0 if none)
          hist      (dict): metric history dict with lists
        """
        p = Path(self.path)
        start_epoch = 0
        hist = hist or {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time": [] }
        if not p.exists():
            print(f"[resume] No checkpoint at {p}; starting from scratch.")
            print("-------------------------------------------------------------------")
            return start_epoch, hist

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "cpu_rng_state" in ckpt:
            torch.set_rng_state(ckpt["cpu_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        start_epoch = ckpt.get("epoch", 0)
        if "hist" in ckpt and ckpt["hist"] is not None:
            hist = ckpt["hist"]

        print(f"[resume] Loaded {p}: last finished epoch = {start_epoch}")
        print("-------------------------------------------------------------------")
        return start_epoch, hist

    def remove(self):
        """
        Delete checkpoint file if it exists (fresh start).
        """
        p = Path(self.path)
        if p.exists():
            p.unlink()
            print(f"Removed checkpoint: {p.resolve()}")
        else:
            print(f"No checkpoint found at: {p.resolve()}")

