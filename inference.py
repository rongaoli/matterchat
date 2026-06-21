"""Inference script for MatterChat LLaVA-style model."""

import torch
import pickle
from tqdm import tqdm

from Model.matterchat_llava import MatterChatLLaVA

# ---------------------------------------------------------------------------
# Config — adjust for your server
# ---------------------------------------------------------------------------
LLM_PATH = "/root/autodl-fs/matterchat/Mistral-7B-Instruct-v0.3"
MODEL_CKPT = "./output/matterchat_final.pt"
EMBED_VAL = "/root/autodl-fs/matterchat/mp_val_potnet_emb.pt"
MATERIAL_VAL = "/root/autodl-fs/matterchat/Material_data_postprocess1_out_correct_val.pkl"
DEVICE = "cuda:0"


def load_model():
    model = MatterChatLLaVA(llm_model_path=LLM_PATH, encoder_dim=256)
    sd = torch.load(MODEL_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    return model


def evaluate(model, data, embeddings):
    """Run 9-task evaluation matching inference_MatterChat_fig5.py."""
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

        def extract_float(text):
            import re
            m = re.search(r"-?\d+\.\d+", text)
            return float(m.group()) if m else 0.0

        # Classification tasks
        out = ask("Is the material metal or not metal?")
        pred = not contains("not metal", out)
        results["is_metal_correct"] += int(pred == sample["is_metal"])

        out = ask("Does the material have a direct bandgap or indirect bandgap?")
        pred = not contains("ind", out)
        results["direct_bandgap_correct"] += int(pred == sample["direct_bandgap"])

        out = ask("Is this material stable?")
        pred = not contains("not", out)
        results["stable_correct"] += int(pred == sample["stable"])

        out = ask("Is the material experimentally observed or not?")
        pred = not contains("not", out)
        results["exp_observe_correct"] += int(pred == sample["exp_observe"])

        out = ask("Is the material magnetic or not?")
        pred = not contains("not", out)
        results["is_magnetic_correct"] += int(pred == sample["is_magnetic"])

        out = ask("What is the magnetic order of the material?")
        gt = sample["magnetic_order"]
        if gt is None:
            results["magnetic_order_correct"] += int("nm" in out.lower() or "none" in out.lower())
        else:
            results["magnetic_order_correct"] += int(gt.lower() in out.lower())

        # Regression tasks
        out = ask("What is the bandgap of the material?")
        results["bandgap_se"] += (sample["bandgap"] - extract_float(out)) ** 2

        out = ask("What is the energy above the hull for this material?")
        results["energy_above_hull_se"] += (sample["energy_above_hull"] - extract_float(out)) ** 2

        out = ask("What is the formation energy value for this material?")
        results["formation_energy_se"] += (sample["formation_energy"] - extract_float(out)) ** 2

    print(f"\nResults over {count} samples:")
    for k in ["is_metal", "direct_bandgap", "stable", "exp_observe", "is_magnetic", "magnetic_order"]:
        acc = results[f"{k}_correct"] / count
        print(f"  {k} accuracy: {acc:.4f}")
    for k in ["bandgap", "formation_energy", "energy_above_hull"]:
        rmse = (results[f"{k}_se"] / count) ** 0.5
        print(f"  {k} RMSE: {rmse:.5f}")


def interactive_demo(model, embeddings, data):
    """Interactive chat with a material."""
    mp_ids = list(set(embeddings.keys()) & set(data.keys()))
    print(f"\nAvailable materials: {len(mp_ids)}")
    print("Enter mp-id to select a material, then ask questions. Type 'quit' to exit.\n")

    while True:
        mp_id = input("mp-id (e.g. mp-1001021): ").strip()
        if mp_id == "quit":
            break
        if mp_id not in embeddings:
            print(f"  {mp_id} not found in embeddings.")
            continue

        emb = embeddings[mp_id]
        info = data.get(mp_id, {})
        print(f"  Loaded {mp_id}: {info.get('reduced_formula', '?')}, {info.get('space_group', '?')}")

        while True:
            question = input("  Q: ").strip()
            if question in ("", "quit", "next"):
                break
            answer = model.generate(emb, question)
            print(f"  A: {answer}\n")


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
        interactive_demo(model, data, embeddings)
