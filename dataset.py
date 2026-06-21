"""MatterChat dataset: loads precomputed PotNet embeddings + generates QA pairs from templates."""

import json
import random
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

TEMPLATES_PATH = Path(__file__).parent / "task_templates.json"

with open(TEMPLATES_PATH) as f:
    TASK_TEMPLATES = json.load(f)

DESCRIPTIVE_TASKS = [k for k, v in TASK_TEMPLATES.items() if v["type"] == "descriptive"]
ALL_TASKS = list(TASK_TEMPLATES.keys())


def _resolve_value(tmpl, raw_val):
    if "value_map" in tmpl:
        return tmpl["value_map"][str(raw_val).lower()]
    if "format" in tmpl:
        return f"{raw_val:{tmpl['format']}}"
    if raw_val is None:
        return "NM"
    return str(raw_val)


class MatterChatDataset(Dataset):
    def __init__(self, data_path, embedding_path, stage="finetune"):
        with open(data_path, "rb") as f:
            self.material_data = pickle.load(f)

        self.embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)

        self.mp_ids = [k for k in self.material_data if k in self.embeddings]
        self.atom_counts = [self.embeddings[k].size(0) for k in self.mp_ids]
        self.tasks = DESCRIPTIVE_TASKS if stage == "pretrain" else ALL_TASKS
        print(f"Dataset: {len(self.mp_ids)} samples, {len(self.tasks)} tasks (stage={stage})")

    def __len__(self):
        return len(self.mp_ids)

    def __getitem__(self, idx):
        mp_id = self.mp_ids[idx]
        sample = self.material_data[mp_id]
        embed = self.embeddings[mp_id]

        task_name = random.choice(self.tasks)
        tmpl = TASK_TEMPLATES[task_name]
        val = _resolve_value(tmpl, sample[tmpl["field"]])

        i = random.randrange(min(len(tmpl["questions"]), len(tmpl["answers"])))
        return {
            "mp_id": mp_id,
            "material_embed": embed,
            "question": tmpl["questions"][i],
            "answer": tmpl["answers"][i].format(val=val),
        }


def collate_fn(batch):
    return {
        "material_embeds": [item["material_embed"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
    }


class LengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, bucket_size=1000):
        self.indices = sorted(range(len(dataset)), key=lambda i: dataset.atom_counts[i])
        self.bucket_size = bucket_size

    def __iter__(self):
        chunks = [self.indices[i:i + self.bucket_size]
                  for i in range(0, len(self.indices), self.bucket_size)]
        random.shuffle(chunks)
        for chunk in chunks:
            shuffled = chunk.copy()
            random.shuffle(shuffled)
            yield from shuffled

    def __len__(self):
        return len(self.indices)
