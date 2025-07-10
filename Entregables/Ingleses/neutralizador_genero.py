#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neutralize_eval_gender.py
────────────────────────────────────────────────────────────────────────────
• Lee los tweets «claros» (JSON / JSONL) aparecidos tras el fine-tuning
  para GENDER (Female / Male).
• Crea tres variantes neutralizadas por tuit con diferentes paraphrasers.
• Calcula las *mismas* 5 características lingüísticas usadas en el modelo
  (sin GloVe), las normaliza con el `StandardScaler` del entrenamiento y
  vuelve a inferir con el checkpoint.
• Devuelve:
     1) CSV con todas las variantes y su probabilidad / predicción.
     2) fichero *neutralized_summary.txt* con F1 y confianza media por
        neutralizador.
────────────────────────────────────────────────────────────────────────────
Ejemplo de ejecución:

python neutralize_eval_gender.py \
       --tweets outputs_gender/clear_tweets_gender.json \
       --ckpt   outputs_gender/best_model_gender.pt \
       --scaler outputs_gender/scaler_gender.pkl
"""

# ───────────────────── IMPORTS ─────────────────────
import argparse, json, warnings
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
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()

# ─────────────── CONFIGURACIÓN BÁSICA ──────────────
# Variables numéricas que el modelo espera (en el mismo orden)
FEATURES = ['Xtwice', 'Xstop', 'Xdet', 'Xprep', 'Xmentions']

NEUTRALIZERS = {
    "parrot":     "prithivida/parrot_paraphraser_on_T5",
    "t5_paws":    "Vamsi/T5_Paraphrase_Paws",
    "t5_chatgpt": "humarin/chatgpt_paraphraser_on_T5_base"
}

CONF_THR_ORIG = 0.70   # confianza mínima que tenían los tweets «claros»

# ───────────── UTILIDADES DE FEATURES ──────────────
def compute_ling_features(text: str) -> dict:
    """Devuelve exactamente las 5 features tabulares del modelo GENDER."""
    words = text.split()
    return {
        "Xtwice":     sum(text.count(w) > 1 for w in set(words)),
        "Xstop":      sum(w.lower() in {"the", "a", "an", "and", "or"} for w in words),
        "Xdet":       sum(w.lower() in {"the", "a", "an"}             for w in words),
        "Xprep":      sum(w.lower() in {"in", "on", "at", "by", "with"} for w in words),
        "Xmentions":  text.count("@")
    }

# ────────────── MODELO TEXT + TABLAS ───────────────
class TransfTab(nn.Module):
    """RoBERTa → CLS (768) + MLP sobre 5 features → 2 clases."""
    def __init__(self, backbone: str, n_tab: int):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(backbone)
        self.tabular_net = nn.Sequential(
            nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        self.classifier  = nn.Sequential(
            nn.Linear(768 + 32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, nums):
        cls = self.transformer(input_ids, attention_mask).last_hidden_state[:, 0]
        tab = self.tabular_net(nums)
        return self.classifier(torch.cat([cls, tab], dim=1))

# ───────────────────────── MAIN ────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —— checkpoint & scaler —— #
    ckpt   = torch.load(args.ckpt, map_location="cpu")
    model  = TransfTab(ckpt["backbone"], len(FEATURES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt["backbone"])
    scaler    = joblib_load(args.scaler)

    # —— neutralizers —— #
    pipes = {}
    for tag, mdl in NEUTRALIZERS.items():
        try:
            print(f"🔄 Cargando neutralizador: {tag}")
            pipes[tag] = pipeline(
                "text2text-generation", model=mdl,
                device=0 if torch.cuda.is_available() else -1,
                max_length=128, do_sample=False
            )
        except Exception as e:
            print(f"⚠️  No se pudo cargar {tag}: {e}")

    # —— tweets claros —— #
    path_in = Path(args.tweets)
    raw = path_in.read_text("utf-8").strip()
    tweets = json.loads(raw) if raw.lstrip().startswith("[") else \
             [json.loads(l) for l in raw.splitlines()]

    rows = []

    def infer(txt: str, feats_vec: np.ndarray):
        tok = tokenizer(txt, truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
                torch.tensor(feats_vec, dtype=torch.float32, device=device).unsqueeze(0)
            )
            prob = torch.softmax(logits, 1).cpu().numpy()[0]
        return int(prob.argmax()), float(prob.max())

    for tw in tqdm(tweets, desc="Neutralizando"):
        base_txt   = tw["text_en"]
        true_label = int(tw["true"])

        feats_base = scaler.transform([np.array(
            list(compute_ling_features(base_txt).values()), dtype=np.float32
        )])[0]
        pred_o, prob_o = infer(base_txt, feats_base)

        rows.append({**tw, "variant": "original",
                     "pred": pred_o, "prob": prob_o})

        # —— variantes neutralizadas —— #
        for tag, pipe in pipes.items():
            try:
                neutral_txt = pipe(base_txt)[0]["generated_text"]
            except Exception:
                neutral_txt = base_txt  # fallback

            feats_n = scaler.transform([np.array(
                list(compute_ling_features(neutral_txt).values()), dtype=np.float32
            )])[0]
            pred_n, prob_n = infer(neutral_txt, feats_n)

            rows.append({
                **tw, "variant": tag, "pred": pred_n, "prob": prob_n,
                "text_en": neutral_txt
            })

    # —— guardar CSV —— #
    df_out = pd.DataFrame(rows)
    out_csv = "neutralized_gender_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Guardado CSV con {len(df_out)} filas → {out_csv}")

    # —— resumen —— #
    variants = ["original"] + list(pipes.keys())
    with open("neutralized_gender_summary.txt", "w") as fh:
        for v in variants:
            sub = df_out[df_out["variant"] == v]
            f1  = f1_score(sub["true"], sub["pred"], average="macro")
            conf = sub["prob"].mean()
            line = f"{v:<12} F1={f1:.4f}  conf_media={conf:.3f}\n"
            print(line.strip())
            fh.write(line)

# ─────────────── CLI / ARGUMENTOS ────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", required=True,
                    help="Ruta al JSON/JSONL de tweets claros de género")
    ap.add_argument("--ckpt",   required=True,
                    help="Checkpoint .pt del mejor modelo de género")
    ap.add_argument("--scaler", required=True,
                    help="StandardScaler .pkl del entrenamiento")
    main(ap.parse_args())
