"""MatterChat LLaVA-style model: MLP projector + Qwen2.5-7B."""

import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM

IGNORE_INDEX = -100
MATERIAL_TOKEN = "<material>"


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class MatterChatLLaVA(nn.Module):
    def __init__(
        self,
        llm_model_path: str,
        encoder_dim: int = 256,
        max_txt_len: int = 512,
        max_output_txt_len: int = 256,
    ):
        super().__init__()
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, use_fast=False, truncation_side="left",
        )
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.bfloat16,
        )
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        llm_dim = self.llm_model.config.hidden_size
        self.projector = MLPProjector(encoder_dim, llm_dim)

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def _maybe_autocast(self):
        if self.projector.proj[0].weight.device == torch.device("cpu"):
            return contextlib.nullcontext()
        return autocast(dtype=torch.bfloat16)

    def _build_inputs(self, material_embeds, material_masks, text_input, text_output, device):
        """Build concat(material_tokens, prompt) -> answer sequence for training."""
        projected = self.projector(material_embeds)  # [B, N_atoms_max, llm_dim]

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        input_tokens = self.llm_tokenizer(
            text_input, return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_txt_len,
        ).to(device)

        self.llm_tokenizer.truncation_side = "right"
        output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in text_output],
            return_tensors="pt", padding="longest",
            truncation=True, max_length=self.max_output_txt_len,
        ).to(device)

        # concat input_ids and output_ids (skip output BOS)
        llm_input_ids = torch.cat([input_tokens.input_ids, output_tokens.input_ids[:, 1:]], dim=1)
        llm_attn_mask = torch.cat([input_tokens.attention_mask, output_tokens.attention_mask[:, 1:]], dim=1)

        # build targets: mask input portion with IGNORE_INDEX
        targets = llm_input_ids.clone()
        targets[targets == self.llm_tokenizer.pad_token_id] = IGNORE_INDEX
        input_len = input_tokens.attention_mask.sum(dim=1)  # [B]
        for i, l in enumerate(input_len):
            targets[i, :l] = IGNORE_INDEX

        # prepend material tokens: also masked in targets
        text_embeds = self.get_input_embeddings()(llm_input_ids)
        inputs_embeds = torch.cat([projected, text_embeds], dim=1)

        mat_attn = material_masks  # [B, N_atoms_max]
        attention_mask = torch.cat([mat_attn, llm_attn_mask], dim=1)

        mat_targets = torch.full(
            (projected.size(0), projected.size(1)), IGNORE_INDEX,
            dtype=targets.dtype, device=device,
        )
        targets = torch.cat([mat_targets, targets], dim=1)

        return inputs_embeds, attention_mask, targets

    def forward(self, material_embeds, material_masks, text_input, text_output):
        """
        Args:
            material_embeds: [B, N_max, encoder_dim] padded atom embeddings
            material_masks:  [B, N_max] attention mask (1=real, 0=pad)
            text_input:      list[str] question prompts
            text_output:     list[str] answer texts
        """
        device = material_embeds.device
        inputs_embeds, attention_mask, targets = self._build_inputs(
            material_embeds, material_masks, text_input, text_output, device,
        )
        with self._maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=targets,
                return_dict=True,
            )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        material_embed,
        prompt,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.9,
        num_beams=1,
    ):
        """
        Inference for a single material.
        Args:
            material_embed: [N_atoms, encoder_dim] single structure embedding
            prompt: str question
        """
        device = next(self.parameters()).device
        material_embed = material_embed.to(device)
        if material_embed.dim() == 2:
            material_embed = material_embed.unsqueeze(0)  # [1, N, D]

        projected = self.projector(material_embed)  # [1, N, llm_dim]
        mat_attn = torch.ones(projected.size()[:-1], dtype=torch.long, device=device)

        self.llm_tokenizer.padding_side = "left"
        prompt_tokens = self.llm_tokenizer(
            [prompt], return_tensors="pt", padding="longest",
        ).to(device)

        with self._maybe_autocast():
            prompt_embeds = self.get_input_embeddings()(prompt_tokens.input_ids)
            inputs_embeds = torch.cat([projected, prompt_embeds], dim=1)
            attention_mask = torch.cat([mat_attn, prompt_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=temperature > 0,
            )

        output_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return output_text

    def freeze_llm(self):
        for p in self.llm_model.parameters():
            p.requires_grad = False

    def unfreeze_llm(self):
        for p in self.llm_model.parameters():
            p.requires_grad = True

    def trainable_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Trainable: {trainable/1e6:.1f}M / Total: {total/1e6:.1f}M ({100*trainable/total:.1f}%)"
