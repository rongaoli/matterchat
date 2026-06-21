"""MatterChat LLaVA-style model: MLP projector + Qwen2.5-7B."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a materials science assistant."


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

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        llm_dim = self.llm_model.config.hidden_size
        self.projector = MLPProjector(encoder_dim, llm_dim)

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def _format_prompt(self, question):
        return self.llm_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def forward(self, material_embeds, questions, answers):
        """
        Args:
            material_embeds: list[Tensor], each [N_atoms_i, encoder_dim]
            questions:       list[str]
            answers:         list[str]
        """
        device = next(self.parameters()).device
        batch_size = len(material_embeds)
        embed_fn = self.get_input_embeddings()

        all_embeds = []
        all_labels = []

        for embed, q, a in zip(material_embeds, questions, answers):
            projected = self.projector(embed.to(device))  # [N_atoms, llm_dim]

            prompt_text = self._format_prompt(q)
            answer_text = f"{a}<|im_end|>"

            prompt_ids = self.llm_tokenizer(
                prompt_text, truncation=True, max_length=self.max_txt_len, return_tensors="pt",
            ).input_ids.squeeze(0).to(device)

            answer_ids = self.llm_tokenizer(
                answer_text, truncation=True, max_length=self.max_output_txt_len, return_tensors="pt",
            ).input_ids.squeeze(0).to(device)

            text_ids = torch.cat([prompt_ids, answer_ids])
            text_embeds = embed_fn(text_ids)

            seq_embeds = torch.cat([projected, text_embeds], dim=0)  # [seq_i, dim]
            labels = torch.cat([
                torch.full((projected.size(0) + prompt_ids.size(0),), IGNORE_INDEX, dtype=torch.long, device=device),
                answer_ids,
            ])  # [seq_i]

            all_embeds.append(seq_embeds)
            all_labels.append(labels)

        max_len = max(e.size(0) for e in all_embeds)
        llm_dim = all_embeds[0].size(1)

        inputs_embeds = torch.zeros(batch_size, max_len, llm_dim, dtype=all_embeds[0].dtype, device=device)
        labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

        for i, (emb, lab) in enumerate(zip(all_embeds, all_labels)):
            seq_len = emb.size(0)
            inputs_embeds[i, :seq_len] = emb
            labels[i, :seq_len] = lab
            attention_mask[i, :seq_len] = 1

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, material_embed, prompt, max_new_tokens=128, temperature=0.1, top_p=0.9):
        device = next(self.parameters()).device
        projected = self.projector(material_embed.to(device))  # [N_atoms, llm_dim]
        if projected.dim() == 2:
            projected = projected.unsqueeze(0)  # [1, N_atoms, llm_dim]

        chatml_prompt = self._format_prompt(prompt)
        prompt_ids = self.llm_tokenizer(chatml_prompt, return_tensors="pt").input_ids.to(device)

        prompt_embeds = self.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([projected, prompt_embeds], dim=1)
        attention_mask = torch.ones(1, inputs_embeds.size(1), dtype=torch.long, device=device)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def freeze_llm(self):
        for p in self.llm_model.parameters():
            p.requires_grad = False

    def unfreeze_llm(self):
        for p in self.llm_model.parameters():
            p.requires_grad = True
