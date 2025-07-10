#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neutralize_eval.py
────────────────────────────────────────────────────────────────────────────
• Lee los tweets “claros” (json o jsonl) generados en el paso anterior
• Genera 3 versiones neutralizadas de cada texto
• Para cada versión:
      – recalcula GloVe + features lingüísticas originales
      – pasa todo por el modelo (checkpoint + scaler)
• Devuelve:
      1) CSV con todas las filas/variantes/neutralizadores
      2) Resumen de caída media de confianza y F1 por neutralizador
────────────────────────────────────────────────────────────────────────────
Uso:
  python neutralize_eval.py \
         --tweets outputs/clear_tweets_ideology.json \
         --ckpt   outputs/best_model.pt \
         --scaler outputs/scaler.pkl
"""

# ───────────────── IMPORTS ─────────────────
import argparse, json, warnings, random, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import f1_score
from joblib import load as joblib_load
from transformers import (
    AutoTokenizer, AutoModel,
    pipeline, logging as hf_log
)
from googletrans import Translator
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()

# ──────────────── FEATURES & MODELOS ────────────────
FEATURES = [
    "Xwords", "Xtwice", "Xmax_length", "Xcharacter", "Xcapital",
    "Xword_par", "Xchar_par", "Xdet", "Xprep", "Xpronouns", "Xmentions",
]

# ------------------------------- CONFIG ----------------------------------
NEUTRALIZERS = {
    "parrot": "prithivida/parrot_paraphraser_on_T5",
    "t5_paws": "Vamsi/T5_Paraphrase_Paws",
    "t5_chatgpt": "humarin/chatgpt_paraphraser_on_T5_base"
}

# ---------------------- Pipeline neutralización --------------------------
def pipeline_neutralizers():
    pipes = {}
    for tag, model_name in NEUTRALIZERS.items():
        try:
            print(f"🔄 Cargando modelo de neutralización '{tag}'...")
            pipes[tag] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=128,
                do_sample=False
            )
        except Exception as e:
            print(f"⚠️ No se pudo cargar {tag}: {e}")
    return pipes
CONF_THR = 0.80          # umbral de confianza original

# ────────────── UTILIDADES DE FEATURES ─────────────
def compute_glove_embedding(text: str) -> np.ndarray:
    return np.array([])  # Ya no se usa

def compute_ling_features(text: str) -> dict:
    words = text.split()
    num_words = len(words)
    num_sentences = text.count(".") + 1  # Evitar división por cero

    return {
        "Xwords": num_words,
        "Xtwice": sum([text.count(w) > 1 for w in set(words)]),
        "Xmax_length": max([len(w) for w in words]) if words else 0,
        "Xcharacter": len(text),
        "Xcapital": sum(1 for c in text if c.isupper()),
        "Xword_par": num_words / num_sentences,
        "Xchar_par": len(text) / num_sentences,
        "Xdet": sum(w.lower() in ["the", "a", "an"] for w in words),
        "Xprep": sum(w.lower() in ["in", "on", "at", "by", "with"] for w in words),
        "Xpronouns": sum(w.lower() in ["he", "she", "they", "i", "we", "you", "it"] for w in words),
        "Xmentions": text.count("@")
    }

# ──────────────── MODELO TABULAR + TEXT ─────────────
class TransfTab(nn.Module):
    def __init__(self, name: str, n_tab: int):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(name)      # 768-dim CLS
        # ----- parte tabular -----
        self.tabular_net = nn.Sequential(
            nn.Linear(n_tab, 64),
            nn.ReLU(),
            nn.Dropout(0.2),          #  ←  ¡dropout original!
            nn.Linear(64, 32)
        )
        # ----- clasificador final -----
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),          #  ←  ¡dropout original!
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, nums):
        cls = self.transformer(input_ids, attention_mask).last_hidden_state[:, 0]
        tab = self.tabular_net(nums)
        logits = self.classifier(torch.cat([cls, tab], dim=1))
        return logits
# ───────────────────────── MAIN ─────────────────────
def main(args):
    # --- cargar checkpoint & scaler ---
    ckpt   = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransfTab(ckpt["backbone"], len(FEATURES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt["backbone"])
    scaler    = joblib_load(args.scaler)

    # --- pipelines de neutralización (perezosos) ---
    neutralizers = {
        tag: pipeline(
            "text2text-generation", model=mdl,
            device=0 if torch.cuda.is_available() else -1,
            max_length=128, do_sample=False
        )
        for tag, mdl in NEUTRALIZERS.items()
    }

    def infer_one(txt, feats):
        tok = tokenizer(
            txt, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            out = model(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
                torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            )
            prob = torch.softmax(out, 1).cpu().numpy()[0]
        return int(prob.argmax()), float(prob.max())

    # --- leer tweets claros (json o jsonl) ---
    path_in = Path(args.tweets)
    txt_raw = path_in.read_text("utf-8").strip()
    tweets  = (
        json.loads(txt_raw)                               # JSON lista
        if txt_raw.lstrip().startswith("[")
        else [json.loads(l) for l in txt_raw.splitlines()]  # JSONL
    )

    translator = Translator()
    rows = []
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    for tw in tqdm(tweets, desc="Neutralizando"):
        base_txt   = tw["text_en"] if "text_en" in tw else tw["clean_text"]
        true_label = int(tw["true"]) if "true" in tw else int(tw["label"])

        # ----- features originales -----
        feats = np.concatenate([
            compute_glove_embedding(base_txt),
            np.array(list(compute_ling_features(base_txt).values()))
        ])
        feats = scaler.transform([feats])[0]
        pred_o, prob_o = infer_one(base_txt, feats)

        rows.append({
            **tw,
            "variant": "original", "pred": pred_o, "prob": prob_o
        })


        # ----- pasar por cada neutralizador -----
        for tag, pipe in neutralizers.items():
            try:
                neutral_txt = pipe(base_txt)[0]["generated_text"]
            except Exception:
                neutral_txt = base_txt  # fallback si falla

            feats_n = np.concatenate([
                compute_glove_embedding(neutral_txt),
                np.array(list(compute_ling_features(neutral_txt).values()))
            ])
            feats_n = scaler.transform([feats_n])[0]
            pred_n, prob_n = infer_one(neutral_txt, feats_n)

            rows.append({
                **tw,
                "variant": tag, "pred": pred_n, "prob": prob_n,
                "text_en": neutral_txt,
            })

    # ---------------- salvar resultados ----------------
    df = pd.DataFrame(rows)
    out_csv = "neutralized_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅  Guardado CSV con {len(df)} filas → {out_csv}")

    # ----------- resumen por tipo de texto ------------
    for tag in ["original"] + list(NEUTRALIZERS.keys()):
        sub = df[df["variant"] == tag]
        f1  = f1_score(sub["true"], sub["pred"], average="macro")
        print(f"{tag:<10}  F1={f1:.4f}   confianza media={sub['prob'].mean():.3f}")
    with open("neutralized_summary.txt", "w") as fsum:
        for tag in ["original"] + list(NEUTRALIZERS.keys()):
            sub = df[df["variant"] == tag]
            f1  = f1_score(sub["true"], sub["pred"], average="macro")
            line = f"{tag:<10}  F1={f1:.4f}   confianza media={sub['prob'].mean():.3f}\n"
            fsum.write(line)

# ───────────────── args / CLI ───────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tweets", required=True,
                   help="Ruta al JSON/JSONL de tweets claros")
    p.add_argument("--ckpt",   required=True,
                   help="Ruta al checkpoint *.pt guardado")
    p.add_argument("--scaler", required=True,
                   help="Ruta al scaler.pkl correspondiente")
    main(p.parse_args())
