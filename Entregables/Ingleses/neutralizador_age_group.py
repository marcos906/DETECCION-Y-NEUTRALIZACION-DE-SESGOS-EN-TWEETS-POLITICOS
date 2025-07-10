#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
neutralize_eval_age.py
────────────────────────────────────────────────────────────────────────────
• Lee los tweets “claros” del modelo AGE (json/jsonl)
• Genera 3 versiones neutralizadas de cada texto
• Para cada versión:
      – recalcula las 20 features lingüísticas seleccionadas para AGE
      – pasa todo por el modelo (checkpoint + scaler)
• Devuelve:
      1) CSV con todas las filas/variantes/neutralizadores
      2) Resumen de caída media de confianza y F1 por neutralizador
────────────────────────────────────────────────────────────────────────────
Uso:
  python neutralize_eval_age.py \
         --tweets outputs/clear_tweets_age.json \
         --ckpt   outputs/best_model_age.pt \
         --scaler outputs/scaler_age.pkl
"""

# ───────────────── IMPORTS ─────────────────
import argparse, json, warnings, hashlib, re
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
import spacy                                     # ← spaCy para POS

warnings.filterwarnings("ignore")
hf_log.set_verbosity_error()

# ──────────────── FEATURES & MODELO ────────────────
FEATURES = [
    "Xwords", "Xunique", "Xmax_length", "Xlength_3", "Xstop",
    "Xcharacter", "Xcapital", "Xpunctuation", "Xword_par", "Xchar_par",
    "Xdet", "Xprep", "Xsing", "Xplural", "Xadv", "Xnouns", "Xconj",
    "Xpast", "Xhashtag", "Xurl"
]

# ------------------------------- CONFIG ----------------------------------
NEUTRALIZERS = {
    "parrot":    "prithivida/parrot_paraphraser_on_T5",
    "t5_paws":   "Vamsi/T5_Paraphrase_Paws",
    "t5_chatgpt":"humarin/chatgpt_paraphraser_on_T5_base"
}

CONF_THR = 0.80   # umbral confianza original

# ────────────── spaCy para POS ─────────────
print("🔄  Cargando spaCy modelo ‘en_core_web_sm’...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ────────────── UTILIDADES FEATURE  ─────────────
def compute_ling_features(text: str) -> dict:
    doc = nlp(text)
    words           = [t.text for t in doc if not t.is_space]
    word_lens       = [len(w) for w in words]
    num_words       = len(words)
    num_sentences   = max(1, text.count(".") + text.count("!") + text.count("?"))

    pos_counts = doc.count_by(spacy.attrs.POS)
    num_nouns  = sum(pos_counts.get(p, 0) for p in (spacy.symbols.NOUN, spacy.symbols.PROPN))
    num_adv    = pos_counts.get(spacy.symbols.ADV, 0)
    num_det    = pos_counts.get(spacy.symbols.DET, 0)
    num_adp    = pos_counts.get(spacy.symbols.ADP, 0)
    num_conj   = pos_counts.get(spacy.symbols.CCONJ, 0) + pos_counts.get(spacy.symbols.SCONJ, 0)
    num_past   = sum(1 for t in doc if t.tag_ in {"VBD", "VBN"})
    num_sing   = sum(1 for t in doc if t.tag_ in {"NN", "NNP"})
    num_plural = sum(1 for t in doc if t.tag_ in {"NNS", "NNPS"})

    return {
        "Xwords"       : num_words,
        "Xunique"      : len(set(w.lower() for w in words)),
        "Xmax_length"  : max(word_lens) if words else 0,
        "Xlength_3"    : sum(1 for l in word_lens if l <= 3) / num_words if num_words else 0,
        "Xstop"        : sum(t.is_stop for t in doc),
        "Xcharacter"   : len(text),
        "Xcapital"     : sum(1 for c in text if c.isupper()),
        "Xpunctuation" : sum(1 for c in text if c in ".,;:!?"),
        "Xword_par"    : num_words / num_sentences,
        "Xchar_par"    : len(text) / num_sentences,
        "Xdet"         : num_det,
        "Xprep"        : num_adp,
        "Xsing"        : num_sing,
        "Xplural"      : num_plural,
        "Xadv"         : num_adv,
        "Xnouns"       : num_nouns,
        "Xconj"        : num_conj,
        "Xpast"        : num_past,
        "Xhashtag"     : text.count("#"),
        "Xurl"         : len(re.findall(r"https?://\S+", text))
    }

# ────────────── MODELO TABULAR + TEXT ─────────────
class TransfTab(nn.Module):
    def __init__(self, name: str, n_tab: int):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(name)
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

    # --- neutralizadores ---
    neutralizers = pipeline_neutralizers()

    def infer_one(txt, feats):
        tok = tokenizer(txt, truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
                torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            )
        prob = torch.softmax(logits, 1).cpu().numpy()[0]
        return int(prob.argmax()), float(prob.max())

    # --- leer tweets claros ---
    tweets_path = Path(args.tweets)
    raw = tweets_path.read_text("utf-8").strip()
    tweets = json.loads(raw) if raw[0] == "[" else [json.loads(l) for l in raw.splitlines()]

    rows = []
    for tw in tqdm(tweets, desc="Neutralizando"):
        base_txt   = tw.get("text_en") or tw.get("clean_text") or ""
        true_label = int(tw.get("true", tw.get("label", 0)))

        feats_base = scaler.transform([list(compute_ling_features(base_txt).values())])[0]
        pred_o, prob_o = infer_one(base_txt, feats_base)
        rows.append({**tw, "variant":"original", "pred":pred_o, "prob":prob_o})

        # versiones neutralizadas
        for tag, pipe in neutralizers.items():
            try:
                neutral_txt = pipe(base_txt)[0]["generated_text"]
            except Exception:
                neutral_txt = base_txt

            feats_n = scaler.transform([list(compute_ling_features(neutral_txt).values())])[0]
            pred_n, prob_n = infer_one(neutral_txt, feats_n)
            rows.append({**tw, "variant":tag, "pred":pred_n, "prob":prob_n,
                         "text_en":neutral_txt})

    # ---------- salvar resultados ----------
    df = pd.DataFrame(rows)
    out_csv = "neutralized_results_age.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅  CSV con {len(df)} filas → {out_csv}")

    # ---------- resumen ----------
    summary_lines = []
    for tag in ["original"] + list(NEUTRALIZERS.keys()):
        sub = df[df["variant"] == tag]
        f1  = f1_score(sub["true"], sub["pred"], average="macro")
        line = f"{tag:<10}  F1={f1:.4f}  conf_media={sub['prob'].mean():.3f}"
        print(line)
        summary_lines.append(line + "\n")

    Path("neutralized_summary_age.txt").write_text("".join(summary_lines))

# ------------------ helpers ------------------
def pipeline_neutralizers():
    pipes = {}
    for tag, model_name in NEUTRALIZERS.items():
        try:
            print(f"🔄  Cargando neutralizador '{tag}' …")
            pipes[tag] = pipeline(
                "text2text-generation", model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_length=128, do_sample=False
            )
        except Exception as e:
            print(f"⚠️  No se pudo cargar '{tag}': {e}")
    return pipes

# ───────────────── args / CLI ─────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweets", required=True,  help="JSON/JSONL de tweets claros (AGE)")
    ap.add_argument("--ckpt",   required=True,  help="Checkpoint *.pt del mejor modelo AGE")
    ap.add_argument("--scaler", required=True,  help="Scaler.pkl entrenado para AGE")
    main(ap.parse_args())
