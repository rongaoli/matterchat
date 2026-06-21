"""Stage 1: Feature Alignment — train projector only, LLM frozen."""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from model.llava import MatterChatLLaVA
from dataset import MatterChatDataset, collate_fn, LengthGroupedSampler
from model.matterchat_module import MatterChatModule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MATERIAL_TRAIN = "/root/autodl-fs/matterchat/Material_data_postprocess1_out_correct_train.pkl"
EMBED_TRAIN = "/root/autodl-fs/matterchat/mp_train_potnet_emb.pt"
LLM_PATH = "/root/autodl-fs/llm_hub/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/root/autodl-fs/matterchat/stage1"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 64
ACCUM = 1
NUM_WORKERS = 4
DEVICES = 8


def main():
    torch.set_float32_matmul_precision("medium")
    model = MatterChatLLaVA(llm_model_path=LLM_PATH, encoder_dim=256)
    model.freeze_llm()

    dataset = MatterChatDataset(MATERIAL_TRAIN, EMBED_TRAIN, stage="pretrain")
    sampler = LengthGroupedSampler(dataset, batch_size=BATCH_SIZE)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=True,
    )

    steps_per_epoch = len(loader) // (ACCUM * DEVICES)
    total_steps = steps_per_epoch * EPOCHS

    module = MatterChatModule(model, lr=LR, warmup_ratio=0.03, total_steps=total_steps)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=DEVICES,
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=ACCUM,
        gradient_clip_val=1.0,
        logger=WandbLogger(project="matterchat", name="stage1-align"),
        callbacks=[
            ModelCheckpoint(
                dirpath=OUTPUT_DIR, save_top_k=1, monitor="train_loss",
                filename="epoch{epoch:02d}-loss{train_loss:.4f}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=5,
    )
    trainer.fit(module, loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(OUTPUT_DIR, "projector.pt"))
    print(f"Projector saved to {OUTPUT_DIR}/projector.pt")


if __name__ == "__main__":
    main()
