"""Inference script for MatterChat."""

import os
import re
import pickle
from datetime import datetime

import torch
from tqdm import tqdm

from model.llava import MatterChatLLaVA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LLM_PATH = "/root/autodl-fs/llm_hub/Qwen2.5-7B-Instruct"
MODEL_CKPT = "./output/stage2/matterchat_final.pt"
EMBED_VAL = "/root/autodl-fs/matterchat/mp_val_potnet_emb.pt"
MATERIAL_VAL = "/root/autodl-fs/matterchat/Material_data_postprocess1_out_correct_val.pkl"
OUTPUT_DIR = "./output/conversations"
DEVICE = "cuda:0"


def load_model():
    model = MatterChatLLaVA(llm_model_path=LLM_PATH, encoder_dim=256)
    sd = torch.load(MODEL_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def extract_float(text):
    m = re.search(r"-?\d+\.\d+", text)
    return float(m.group()) if m else 0.0


def evaluate(model, data, embeddings):
    results = {
        "is_metal_correct": 0, "direct_bandgap_correct": 0,
        "stable_correct": 0, "exp_observe_correct": 0,
        "is_magnetic_correct": 0, "magnetic_order_correct": 0,
        "bandgap_se": 0.0, "energy_above_hull_se": 0.0,
        "formation_energy_se": 0.0,
    }
    count = 0

    for mp_id, sample in tqdm(data.items()):
        if mp_id not in embeddings:
            continue
        emb = embeddings[mp_id]
        count += 1

        def ask(prompt):
            return model.generate(emb, prompt)

        def contains(sub, text):
            return sub.lower() in text.lower()

        out = ask("Is the material metal or not metal?")
        results["is_metal_correct"] += int((not contains("not metal", out)) == sample["is_metal"])

        out = ask("Does the material have a direct bandgap or indirect bandgap?")
        results["direct_bandgap_correct"] += int((not contains("ind", out)) == sample["direct_bandgap"])

        out = ask("Is this material stable?")
        results["stable_correct"] += int((not contains("not", out)) == sample["stable"])

        out = ask("Is the material experimentally observed or not?")
        results["exp_observe_correct"] += int((not contains("not", out)) == sample["exp_observe"])

        out = ask("Is the material magnetic or not?")
        results["is_magnetic_correct"] += int((not contains("not", out)) == sample["is_magnetic"])

        out = ask("What is the magnetic order of the material?")
        gt = sample["magnetic_order"]
        if gt is None:
            results["magnetic_order_correct"] += int("nm" in out.lower() or "none" in out.lower())
        else:
            results["magnetic_order_correct"] += int(gt.lower() in out.lower())

        out = ask("What is the bandgap of the material?")
        results["bandgap_se"] += (sample["bandgap"] - extract_float(out)) ** 2

        out = ask("What is the energy above the hull for this material?")
        results["energy_above_hull_se"] += (sample["energy_above_hull"] - extract_float(out)) ** 2

        out = ask("What is the formation energy value for this material?")
        results["formation_energy_se"] += (sample["formation_energy"] - extract_float(out)) ** 2

    print(f"\nResults over {count} samples:")
    for k in ["is_metal", "direct_bandgap", "stable", "exp_observe", "is_magnetic", "magnetic_order"]:
        print(f"  {k} accuracy: {results[f'{k}_correct'] / count:.4f}")
    for k in ["bandgap", "formation_energy", "energy_above_hull"]:
        print(f"  {k} RMSE: {(results[f'{k}_se'] / count) ** 0.5:.5f}")


def interactive_demo(model, embeddings, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mp_ids = sorted(set(embeddings.keys()) & set(data.keys()))
    print(f"\nAvailable materials: {len(mp_ids)}")
    print("Enter mp-id to select a material, then ask questions.")
    print("Type 'next' for another material, 'quit' to exit.\n")

    while True:
        mp_id = input("mp-id (e.g. mp-1001021): ").strip()
        if mp_id == "quit":
            break
        if mp_id not in embeddings:
            print(f"  {mp_id} not found.")
            continue

        emb = embeddings[mp_id]
        info = data.get(mp_id, {})
        formula = info.get("reduced_formula", "?")
        print(f"  Loaded {mp_id}: {formula}")

        log_lines = [
            f"# MatterChat Conversation Log",
            f"# Material: {mp_id} ({formula})",
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        while True:
            question = input("  Q: ").strip()
            if question in ("", "quit", "next"):
                break
            answer = model.generate(emb, question)
            print(f"  A: {answer}\n")
            log_lines.append(f"Q: {question}")
            log_lines.append(f"A: {answer}")
            log_lines.append("")

        if len(log_lines) > 4:
            log_path = os.path.join(OUTPUT_DIR, f"conversation_log_{mp_id}.out")
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines))
            print(f"  Conversation saved to {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "chat"], default="eval")
    args = parser.parse_args()

    model = load_model()
    embeddings = torch.load(EMBED_VAL, map_location="cpu", weights_only=True)
    with open(MATERIAL_VAL, "rb") as f:
        data = pickle.load(f)

    if args.mode == "eval":
        evaluate(model, data, embeddings)
    else:
        interactive_demo(model, embeddings, data)
