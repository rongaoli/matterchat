"""Two-stage training for MatterChat (LLaVA-style)."""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from Model.matterchat_llava import MatterChatLLaVA
from dataset import MatterChatDataset, collate_fn


class MatterChatModule(pl.LightningModule):
    def __init__(self, model, lr, warmup_ratio, total_steps, weight_decay):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        loss = self.model(
            material_embeds=batch["material_embeds"].to(self.device),
            material_masks=batch["material_masks"].to(self.device),
            text_input=batch["questions"],
            text_output=batch["answers"],
        )
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.999))
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Paths — adjust for your server
MATERIAL_TRAIN = "/root/autodl-fs/matterchat/Material_data_postprocess1_out_correct_train.pkl"
EMBED_TRAIN = "/root/autodl-fs/matterchat/mp_train_potnet_emb.pt"
LLM_PATH = "/root/autodl-fs/matterchat/Mistral-7B-Instruct-v0.3"

# Stage 1: alignment
STAGE1_LR = 1e-3
STAGE1_EPOCHS = 1
STAGE1_BATCH = 32
STAGE1_ACCUM = 4  # effective batch = 128

# Stage 2: instruction tuning
STAGE2_LR = 2e-5
STAGE2_EPOCHS = 3
STAGE2_BATCH = 4
STAGE2_ACCUM = 32  # effective batch = 128

NUM_WORKERS = 4
OUTPUT_DIR = "./output"


def make_trainer(stage, max_epochs, accum, total_steps):
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(OUTPUT_DIR, f"stage{stage}"),
            filename="epoch{epoch:02d}-loss{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(project="matterchat-llava", name=f"stage{stage}")
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=8,
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=accum,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )


def main():
    # ===================== Stage 1: Alignment =====================
    print("=" * 60)
    print("Stage 1: Feature Alignment (train projector only)")
    print("=" * 60)

    model = MatterChatLLaVA(llm_model_path=LLM_PATH, encoder_dim=256)
    model.freeze_llm()
    print(model.trainable_param_summary())

    train_dataset = MatterChatDataset(MATERIAL_TRAIN, EMBED_TRAIN, stage="pretrain")
    train_loader = DataLoader(
        train_dataset, batch_size=STAGE1_BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=True,
    )

    steps_per_epoch = len(train_loader) // (STAGE1_ACCUM * 8)  # 8 GPUs
    total_steps = steps_per_epoch * STAGE1_EPOCHS

    module = MatterChatModule(
        model, lr=STAGE1_LR, warmup_ratio=0.03,
        total_steps=total_steps, weight_decay=0.0,
    )
    trainer = make_trainer(1, STAGE1_EPOCHS, STAGE1_ACCUM, total_steps)
    trainer.fit(module, train_loader)

    # Save stage 1 projector
    stage1_path = os.path.join(OUTPUT_DIR, "stage1_projector.pt")
    torch.save(model.projector.state_dict(), stage1_path)
    print(f"Stage 1 projector saved to {stage1_path}")

    # ===================== Stage 2: Instruction Tuning =====================
    print("=" * 60)
    print("Stage 2: Instruction Tuning (train projector + LLM)")
    print("=" * 60)

    model.unfreeze_llm()
    print(model.trainable_param_summary())

    train_dataset_s2 = MatterChatDataset(MATERIAL_TRAIN, EMBED_TRAIN, stage="finetune")
    train_loader_s2 = DataLoader(
        train_dataset_s2, batch_size=STAGE2_BATCH, shuffle=True,
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=True,
    )

    steps_per_epoch_s2 = len(train_loader_s2) // (STAGE2_ACCUM * 8)
    total_steps_s2 = steps_per_epoch_s2 * STAGE2_EPOCHS

    module_s2 = MatterChatModule(
        model, lr=STAGE2_LR, warmup_ratio=0.03,
        total_steps=total_steps_s2, weight_decay=0.05,
    )
    trainer_s2 = make_trainer(2, STAGE2_EPOCHS, STAGE2_ACCUM, total_steps_s2)
    trainer_s2.fit(module_s2, train_loader_s2)

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "matterchat_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
