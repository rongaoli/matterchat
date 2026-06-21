"""MatterChat dataset: loads precomputed PotNet embeddings + generates QA pairs from templates."""

import random
import pickle
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Question / Answer templates (from supplementary material Sections 3-8)
# ---------------------------------------------------------------------------

TASK_TEMPLATES = {
    # ---- Descriptive tasks (Stage 1 + Stage 2) ----
    "reduced_formula": {
        "field": "reduced_formula",
        "type": "descriptive",
        "questions": [
            "What is the chemical formula for this material?",
            "Can you tell me the chemical formula of this material?",
            "Please provide the chemical formula for the material.",
            "What is the formula for this material?",
            "Could you tell me the formula of the material?",
            "What elements make up this material?",
            "How would you write the chemical formula of this material?",
            "What is the exact chemical formula of this material?",
            "Can you provide the chemical formula for this material?",
        ],
        "answers": [
            "The chemical formula for this material is {val}.",
            "The chemical formula of this material is {val}.",
            "The chemical formula for the material is {val}.",
            "The formula for this material is {val}.",
            "The formula of the material is {val}.",
            "The chemical formula of this material is written as {val}.",
            "The exact chemical formula of this material is {val}.",
        ],
    },
    "space_group": {
        "field": "space_group",
        "type": "descriptive",
        "questions": [
            "What is the space group for this material?",
            "To which space group does this material belong?",
            "Can you tell me the space group of this material?",
            "Please provide the space group for the material.",
            "What is the crystallographic space group of this material?",
            "How is the space group of this material classified?",
            "Can you specify the space group for this material?",
            "Could you tell me the space group classification of this material?",
            "What is the space group number of this material?",
        ],
        "answers": [
            "The space group for this material is {val}.",
            "This material belongs to the space group {val}.",
            "The space group of this material is {val}.",
            "The space group for the material is {val}.",
            "The crystallographic space group of this material is {val}.",
            "The space group of this material is classified as {val}.",
            "The space group for this material is specified as {val}.",
        ],
    },
    "crystal_system": {
        "field": "crystal_system",
        "type": "descriptive",
        "questions": [
            "What is the crystal system of this material?",
            "Can you tell me the crystal system of this material?",
            "Please provide the crystal system for the material.",
            "What crystal system does this material belong to?",
            "How is the crystal system of this material classified?",
            "Can you specify the crystal system for this material?",
            "What is the crystallographic system of this material?",
            "Could you tell me the crystal system classification of this material?",
            "Which crystallographic system does this material belong to?",
        ],
        "answers": [
            "The crystal system of this material is {val}.",
            "The crystal system for the material is {val}.",
            "This material belongs to the {val} crystal system.",
            "The crystal system of this material is classified as {val}.",
            "The crystallographic system of this material is {val}.",
        ],
    },
    # ---- Property classification tasks (Stage 2 only) ----
    "is_metal": {
        "field": "is_metal",
        "type": "property",
        "questions": [
            "Is this material metal or non-metal?",
            "Can you tell me if this material is metal or not?",
            "What is the classification of this material: metal or non-metal?",
            "Is this material considered a metal?",
            "How is this material categorized: metal or non-metal?",
            "Could you specify if this material is metal or non-metal?",
            "Is the material metallic or non-metallic?",
            "Is this material identified as a metal or non-metal?",
            "What type of material is this: metal or non-metal?",
        ],
        "answers": [
            "This material is classified as {val}.",
            "This material is a {val}.",
            "The classification of this material is {val}.",
            "This material is considered {val}.",
            "This material is categorized as {val}.",
            "This material is {val}.",
        ],
        "value_map": {True: "metal", False: "non-metal"},
    },
    "direct_bandgap": {
        "field": "direct_bandgap",
        "type": "property",
        "questions": [
            "Does the material have a direct bandgap or indirect bandgap?",
            "Is the bandgap of this material direct or indirect?",
            "Can you tell me if this material has a direct or indirect bandgap?",
            "What type of bandgap does this material have: direct or indirect?",
            "Is this material characterized by a direct or indirect bandgap?",
            "Could you specify if the bandgap of this material is direct or indirect?",
            "Does this material exhibit a direct or indirect bandgap?",
            "Is the bandgap in this material direct or indirect?",
            "How is the bandgap of this material classified: direct or indirect?",
            "Is this a direct or indirect bandgap material?",
        ],
        "answers": [
            "The material has a {val} bandgap.",
            "The bandgap of this material is {val}.",
            "This material has a {val} bandgap.",
            "This material is characterized by a {val} bandgap.",
            "The bandgap of this material is specified as {val}.",
            "This material exhibits a {val} bandgap.",
            "The bandgap in this material is {val}.",
            "The bandgap of this material is classified as {val}.",
            "This is a {val} bandgap material.",
        ],
        "value_map": {True: "direct", False: "indirect"},
    },
    "stable": {
        "field": "stable",
        "type": "property",
        "questions": [
            "Is this material stable?",
            "Can you tell me if this material is stable?",
            "What is the stability of this material?",
            "Please provide the stability information for this material.",
            "Is the material stable under standard conditions?",
            "Is this material thermodynamically stable?",
        ],
        "answers": [
            "This material is {val}.",
            "Yes, this material is {val}.",
            "The stability of this material is {val}.",
            "The stability information for this material is {val}.",
            "This material is {val} under standard conditions.",
        ],
        "value_map": {True: "stable", False: "not stable"},
    },
    "exp_observe": {
        "field": "exp_observe",
        "type": "property",
        "questions": [
            "Is the material experimentally observed or not?",
            "Can you tell me if the material is observed in experiments?",
        ],
        "answers": [
            "The material is {val}.",
        ],
        "value_map": {True: "experimentally observed", False: "not experimentally observed"},
    },
    "is_magnetic": {
        "field": "is_magnetic",
        "type": "property",
        "questions": [
            "Is the material magnetic or not?",
            "Is the material magnetic or non-magnetic?",
            "Can you tell me if this material is magnetic?",
            "What is the magnetic nature of this material?",
            "Is this material classified as magnetic?",
            "Does this material have magnetic properties?",
            "Is this a magnetic or non-magnetic material?",
        ],
        "answers": [
            "The material is {val}.",
            "This material is {val}.",
            "Yes, this material is {val}.",
            "The magnetic nature of this material is {val}.",
            "This material is classified as {val}.",
            "This material has {val} properties.",
            "This is a {val} material.",
        ],
        "value_map": {True: "magnetic", False: "not magnetic"},
    },
    "magnetic_order": {
        "field": "magnetic_order",
        "type": "property",
        "questions": [
            "What is the magnetic order of the material?",
            "Can you tell me the magnetic order of this material?",
            "Could you specify the magnetic order of the material?",
            "What type of magnetic order does this material have?",
            "Please provide the magnetic ordering of the material.",
            "What is the magnetic arrangement in this material?",
            "Could you tell me the type of magnetic order of this material?",
        ],
        "answers": [
            "The magnetic order of the material is {val}.",
            "The magnetic order of this material is {val}.",
            "The magnetic order of the material is specified as {val}.",
            "This material has a {val} type of magnetic order.",
            "The magnetic ordering of the material is {val}.",
            "The magnetic arrangement in this material is {val}.",
            "The type of magnetic order of this material is {val}.",
        ],
    },
    # ---- Property regression tasks (Stage 2 only) ----
    "bandgap": {
        "field": "bandgap",
        "type": "property",
        "questions": [
            "What is the bandgap of the material?",
            "Can you tell me the bandgap of this material?",
            "What is the energy bandgap for this material?",
            "Could you specify the bandgap of the material?",
            "Could you tell me the bandgap energy level of this material?",
        ],
        "answers": [
            "The bandgap of the material is {val} eV.",
            "The bandgap of this material is {val} eV.",
            "The energy bandgap for this material is {val} eV.",
            "The bandgap of the material is specified as {val} eV.",
            "The bandgap energy level of this material is {val} eV.",
        ],
        "format": lambda v: f"{v:.5f}",
    },
    "formation_energy": {
        "field": "formation_energy",
        "type": "property",
        "questions": [
            "Can you tell me the formation energy of this material?",
            "Please provide the formation energy for the material.",
            "What is the formation energy value for this material?",
            "How much is the formation energy of this material?",
            "Can you specify the formation energy of this material?",
        ],
        "answers": [
            "The formation energy of this material is {val} eV per atom.",
            "The formation energy for the material is {val} eV per atom.",
            "The formation energy value of this material is {val} eV per atom.",
            "The formation energy of this material is {val} eV per atom.",
            "The formation energy of this material is specified as {val} eV per atom.",
        ],
        "format": lambda v: f"{v:.5f}",
    },
    "energy_above_hull": {
        "field": "energy_above_hull",
        "type": "property",
        "questions": [
            "Can you tell me the energy above hull of this material?",
            "Please provide the energy above hull for the material.",
            "What is the energy above the hull for this material?",
            "How much is the energy above hull for this material?",
            "Can you specify the energy above hull of the material?",
            "Could you tell me the energy above hull of the material?",
        ],
        "answers": [
            "The energy above hull of this material is {val} eV per atom.",
            "The energy above hull for the material is {val} eV per atom.",
            "The energy above the hull for this material is {val} eV per atom.",
            "The energy above hull for this material is {val} eV per atom.",
            "The energy above hull of this material is specified as {val} eV per atom.",
            "The energy above hull of the material is {val} eV per atom.",
        ],
        "format": lambda v: f"{v:.5f}",
    },
}

DESCRIPTIVE_TASKS = [k for k, v in TASK_TEMPLATES.items() if v["type"] == "descriptive"]
ALL_TASKS = list(TASK_TEMPLATES.keys())


class MatterChatDataset(Dataset):
    """
    Each sample yields:
        material_embed: [N_atoms, 256]
        question: str
        answer: str
    """

    def __init__(self, material_data_path, embedding_path, stage="finetune"):
        """
        Args:
            material_data_path: path to pkl with {mp_id: {structure, properties...}}
            embedding_path: path to pt with {mp_id: tensor[N_atoms, 256]}
            stage: "pretrain" (descriptive only) or "finetune" (all 12 tasks)
        """
        with open(material_data_path, "rb") as f:
            self.material_data = pickle.load(f)

        self.embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)

        # only keep samples that have both property data and embeddings
        self.mp_ids = [
            k for k in self.material_data.keys()
            if k in self.embeddings
        ]
        print(f"Dataset: {len(self.mp_ids)} samples (stage={stage})")

        self.tasks = DESCRIPTIVE_TASKS if stage == "pretrain" else ALL_TASKS

    def __len__(self):
        return len(self.mp_ids)

    def __getitem__(self, idx):
        mp_id = self.mp_ids[idx]
        sample = self.material_data[mp_id]
        embed = self.embeddings[mp_id]  # [N_atoms, 256]

        task_name = random.choice(self.tasks)
        tmpl = TASK_TEMPLATES[task_name]
        field = tmpl["field"]
        raw_val = sample[field]

        # resolve value
        if "value_map" in tmpl:
            val = tmpl["value_map"][raw_val]
        elif "format" in tmpl:
            val = tmpl["format"](raw_val)
        elif raw_val is None:
            val = "NM"
        else:
            val = str(raw_val)

        question = random.choice(tmpl["questions"])
        answer = random.choice(tmpl["answers"]).format(val=val)

        return {
            "mp_id": mp_id,
            "material_embed": embed,
            "question": question,
            "answer": answer,
        }


def collate_fn(batch):
    """Pad material embeddings to the same length within a batch."""
    embeds = [item["material_embed"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    max_atoms = max(e.size(0) for e in embeds)
    embed_dim = embeds[0].size(1)

    padded_embeds = torch.zeros(len(embeds), max_atoms, embed_dim)
    masks = torch.zeros(len(embeds), max_atoms, dtype=torch.long)

    for i, e in enumerate(embeds):
        n = e.size(0)
        padded_embeds[i, :n] = e
        masks[i, :n] = 1

    return {
        "material_embeds": padded_embeds,
        "material_masks": masks,
        "questions": questions,
        "answers": answers,
    }
