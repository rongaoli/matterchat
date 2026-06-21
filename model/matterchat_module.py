"""Shared Lightning module for both training stages."""

import torch
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup


class MatterChatModule(pl.LightningModule):
    def __init__(self, model, lr, warmup_ratio, total_steps, weight_decay=0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        loss = self.model(
            material_embeds=batch["material_embeds"],
            questions=batch["questions"],
            answers=batch["answers"],
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
