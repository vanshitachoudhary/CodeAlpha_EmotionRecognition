"""CodeAlpha Task 2 - Speech Emotion Recognition (RAVDESS + TESS).
Run:  python train.py --data data                 (classical + deep models)
      python train.py --data data --fast          (quick run)
      python train.py --data data --models classical --cross"""
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import augment as A
import data as D
import dl
import features as F
import ml

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
K = len(D.EMOTIONS)
slug = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ---------------------------------------------------------------------------------------------- features
def build_features(df, k_aug, seed, cache=None):
    """Har clip -> (sequence, statistics). Sirf TRAIN clips ke k_aug augmented copies bante hain."""
    if cache and Path(cache).exists():
        z = np.load(cache)
        print(f"Loaded cached features: {cache}")
        return {n: z[n] for n in z.files}
    rng = np.random.default_rng(seed)
    seqs, stats, ys, rows, aug = [], [], [], [], []
    bad = 0
    for i, r in enumerate(df.itertuples()):
        try:
            y, _ = F.load_wav(r.path)
        except Exception as e:                                   # 24-bit WAV, corrupt file, etc.
            bad += 1
            print(f"  skipped {Path(r.path).name}: {type(e).__name__}")
            continue
        if len(y) < 1600:
            continue
        variants = [(y, 0)] + ([(A.augment(y, rng), 1) for _ in range(k_aug)] if r.split == "train" else [])
        for w, is_aug in variants:
            q, s = F.extract(w)
            seqs.append(q); stats.append(s); ys.append(r.label); rows.append(i); aug.append(is_aug)
        if (i + 1) % 400 == 0:
            print(f"  features {i + 1}/{len(df)}")
    out = {"seq": np.stack(seqs).astype(np.float32), "stats": np.stack(stats), "y": np.array(ys),
           "row": np.array(rows), "aug": np.array(aug)}
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------------------------------------- plots
def plot_spectrograms(df, out):
    fig, ax = plt.subplots(2, 4, figsize=(15, 6))
    for a, emo in zip(ax.ravel(), D.EMOTIONS):
        r = df[(df.emotion == emo) & (df.split == "train")].iloc[0]
        y = F.fix_length(F.peak_normalize(F.trim_silence(F.load_wav(r.path)[0])))
        a.imshow(F.log_mel(y).T, origin="lower", aspect="auto", cmap="magma", extent=[0, F.DURATION, 0, F.SR / 2000])
        a.set_title(emo); a.set_xlabel("time (s)"); a.set_ylabel("kHz")
    ax.ravel()[-1].axis("off")
    plt.suptitle("Log-mel spectrogram of one training clip per emotion"); plt.tight_layout()
    plt.savefig(out / "emotion_spectrograms.png", dpi=120); plt.close()


def plot_prosody(stats, y, out):
    names = F.PROSODY_NAMES
    cols = {"f0_mean": "Mean pitch (Hz)", "rms_mean": "Mean loudness (RMS)", "voiced_ratio": "Voiced fraction"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, (c, t) in zip(ax, cols.items()):
        j = 4 * 2 * F.N_MFCC + names.index(c)              # stats = 4 x 80 MFCC-stats, phir prosody
        m = [stats[y == i, j].mean() for i in range(K)]
        s = [stats[y == i, j].std() for i in range(K)]
        a.bar(D.EMOTIONS, m, yerr=s, color=[D.COLOR[e] for e in D.EMOTIONS], capsize=3)
        a.set_title(t); a.tick_params(axis="x", rotation=40)
    plt.suptitle("Do emotions differ in pitch, loudness and voicing? (training data, mean ± std)"); plt.tight_layout()
    plt.savefig(out / "prosody_by_emotion.png", dpi=120); plt.close()


def plot_confusion(y, pred, title, out):
    cm = confusion_matrix(y, pred, labels=range(K)).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(6.4, 5.6)); plt.imshow(cmn, cmap="Purples", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .55 else "black", fontsize=8)
    plt.xticks(range(K), D.EMOTIONS, rotation=40, ha="right"); plt.yticks(range(K), D.EMOTIONS)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title); plt.colorbar(label="row-normalised")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()


def plot_bars(values, labels, title, fname, out, colors=None, ylabel="macro-F1"):
    plt.figure(figsize=(6.5, 4)); plt.bar(labels, values, color=colors or "#7c3aed")
    for i, v in enumerate(values):
        plt.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9)
    plt.ylim(0, 1.08); plt.ylabel(ylabel); plt.title(title); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(out / fname, dpi=120); plt.close()


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="all", help="all | classical | comma list of cnn,bilstm,cnn_lstm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--aug", type=int, default=1, help="augmented copies per training clip")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cross", action="store_true", help="cross-corpus test (RAVDESS <-> TESS)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    out, mdir = Path(a.out), Path(a.models_dir)
    out.mkdir(parents=True, exist_ok=True); mdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)

    # ---- data
    df = D.scan(a.data)
    if df.empty:
        sys.exit(f"No RAVDESS/TESS .wav files found under '{a.data}'. Run `python download_data.py` or see the README.")
    df = D.speaker_split(df, a.seed).reset_index(drop=True)
    counts = df.groupby(["corpus", "split"]).size().unstack(fill_value=0)
    print("\nClips per corpus / split:\n", counts.to_string())
    if (df.split == "val").sum() == 0 or (df.split == "test").sum() == 0:
        sys.exit("Need validation and test clips: add more speakers (RAVDESS) or both corpora.")
    if "tess" in set(df.corpus):
        print("NOTE: TESS has only 2 speakers, so its split is speaker-DEPENDENT (inflated); RAVDESS is speaker-independent.")

    # ---- features
    key = hashlib.md5(("|".join(df.path + df.split) + f"{a.aug}-{a.seed}").encode()).hexdigest()[:10]
    feats = build_features(df, a.aug, a.seed, None if a.no_cache else Path(a.cache) / f"feats_{key}.npz")
    Xq, Xs, y, rowi, aug = feats["seq"], feats["stats"], feats["y"], feats["row"], feats["aug"]
    sp, corpus, gender = df.split.values[rowi], df.corpus.values[rowi], df.gender.values[rowi]
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    print(f"Train {len(tr)} (incl. {int(aug[tr].sum())} augmented) | val {len(va)} | test {len(te)}")

    # ---- train + evaluate every model
    reg = {}                                                         # name -> info dict (+ in-memory objects)
    for name, model in ml.classical_models(a.seed).items():
        print(f"Training {name} ...")
        m = clone(model).fit(Xs[tr], y[tr])
        Pv, Pt = ml.full_proba(m, Xs[va], K), ml.full_proba(m, Xs[te], K)
        reg[name] = {"kind": "sklearn", "obj": m, "Pv": Pv, "Pt": Pt}

    dl_names = []
    if a.models != "classical":
        dl_names = dl.NAMES if a.models == "all" else [n for n in a.models.split(",") if n in dl.NAMES]
        if a.fast:
            dl_names = dl_names[:1]
    hist = {}
    if dl_names and not dl.available():
        print("\nTensorFlow not installed -> skipping CNN/LSTM models. Install with:  pip install -r requirements-dl.txt\n")
        dl_names = []
    if dl_names:
        mu, sd = Xq[tr].mean(axis=(0, 1)), Xq[tr].std(axis=(0, 1)) + 1e-6
        np.savez(mdir / "norm.npz", mu=mu, sd=sd)
        Xn = (Xq - mu) / sd
        for n in dl_names:
            print(f"\nTraining {dl.TITLES[n]} ...")
            model = dl.build(n, Xn.shape[1:], K, a.seed)
            hist[dl.TITLES[n]] = dl.fit(model, Xn[tr], y[tr], Xn[va], y[va], 15 if a.fast else a.epochs,
                                        class_weight=ml.class_weights(y[tr]))
            model.save(mdir / f"{n}.keras")
            reg[dl.TITLES[n]] = {"kind": "keras", "file": f"{n}.keras", "Pv": model.predict(Xn[va], verbose=0),
                                 "Pt": model.predict(Xn[te], verbose=0)}

    for name, r in reg.items():
        r["val"], r["test"] = ml.evaluate(y[va], r["Pv"].argmax(1)), ml.evaluate(y[te], r["Pt"].argmax(1))
        if r["kind"] == "sklearn":
            r["file"] = f"{slug(name)}.joblib"
            joblib.dump(r["obj"], mdir / r["file"], compress=3)
    best = max(reg, key=lambda n: reg[n]["val"]["macro_f1"])           # selection: validation speakers only
    best_cl = max((n for n in reg if reg[n]["kind"] == "sklearn"), key=lambda n: reg[n]["val"]["macro_f1"])
    pred = reg[best]["Pt"].argmax(1)
    ci = {"accuracy": ml.bootstrap_ci(y[te], pred, lambda p, q: float((p == q).mean())),
          "macro_f1": ml.bootstrap_ci(y[te], pred, ml.macro_f1)}
    pcf = ml.per_class_f1(y[te], pred, K)
    print(f"\nBest model (validation macro-F1): {best}")

    # ---- subgroups on the test split
    sub = []
    for g, mask in [("RAVDESS (speaker-independent)", corpus[te] == "ravdess"), ("TESS (speaker-dependent)", corpus[te] == "tess"),
                    ("Male speakers", gender[te] == "male"), ("Female speakers", gender[te] == "female")]:
        if mask.sum() >= 10:
            e = ml.evaluate(y[te][mask], pred[mask])
            sub.append({"Group": g, "n": int(mask.sum()), "Accuracy": round(e["accuracy"], 3), "Macro-F1": round(e["macro_f1"], 3)})

    # ---- leaky vs honest (RAVDESS): same model type, random utterance split vs speaker-independent
    leaky = None
    ri = np.where((corpus == "ravdess") & (aug == 0))[0]
    rt = np.where((corpus[te] == "ravdess"))[0]
    if len(ri) >= 100 and len(rt) >= 20:
        a_i, b_i = train_test_split(ri, test_size=0.25, stratify=y[ri], random_state=a.seed)
        m = clone(ml.classical_models(a.seed)[best_cl]).fit(Xs[a_i], y[a_i])
        lk = ml.evaluate(y[b_i], ml.full_proba(m, Xs[b_i], K).argmax(1))
        honest = ml.evaluate(y[te][rt], reg[best_cl]["Pt"].argmax(1)[rt])
        leaky = {"model": best_cl, "random_split_accuracy": lk["accuracy"], "speaker_independent_accuracy": honest["accuracy"],
                 "random_split_f1": lk["macro_f1"], "speaker_independent_f1": honest["macro_f1"]}
        print(f"Leakage check ({best_cl}): random split acc {lk['accuracy']:.3f} vs speaker-independent {honest['accuracy']:.3f}")

    # ---- cross-corpus
    cross = {}
    if a.cross and {"ravdess", "tess"} <= set(corpus):
        for src, dst in [("ravdess", "tess"), ("tess", "ravdess")]:
            s_i, d_i = np.where((corpus == src) & (aug == 0))[0], np.where((corpus == dst) & (aug == 0))[0]
            m = clone(ml.classical_models(a.seed)[best_cl]).fit(Xs[s_i], y[s_i])
            e = ml.evaluate(y[d_i], ml.full_proba(m, Xs[d_i], K).argmax(1))
            cross[f"{src} -> {dst}"] = {"accuracy": e["accuracy"], "macro_f1": e["macro_f1"]}
            print(f"Cross-corpus {src} -> {dst}: acc {e['accuracy']:.3f}, macro-F1 {e['macro_f1']:.3f}")

    # ---- plots
    plot_spectrograms(df, out)
    plot_prosody(Xs[tr][aug[tr] == 0], y[tr][aug[tr] == 0], out)
    plot_confusion(y[te], pred, f"Confusion matrix: {best} (test speakers)", out)
    plot_bars(pcf, list(D.EMOTIONS), f"Per-emotion F1: {best}", "per_class_f1.png", out,
              [D.COLOR[e] for e in D.EMOTIONS], "F1")
    plot_bars([reg[n]["test"]["macro_f1"] for n in reg], list(reg), "Test macro-F1 by model", "model_comparison.png", out)
    if leaky:
        plot_bars([leaky["random_split_accuracy"], leaky["speaker_independent_accuracy"]],
                  ["random utterance split\n(leaky)", "speaker-independent\n(honest)"],
                  "Why speaker-independent evaluation matters", "leaky_vs_honest.png", out, ["#f59e0b", "#7c3aed"], "accuracy")
    if hist:
        plt.figure(figsize=(6.5, 4))
        for n, h in hist.items():
            plt.plot(h["val_accuracy"], label=n)
        plt.xlabel("epoch"); plt.ylabel("validation accuracy"); plt.title("Training curves"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "training_curves.png", dpi=120); plt.close()

    # ---- reports + meta
    table = pd.DataFrame({n: {"Val acc": round(r["val"]["accuracy"], 3), "Val macro-F1": round(r["val"]["macro_f1"], 3),
                              "Test acc": round(r["test"]["accuracy"], 3), "Test macro-F1": round(r["test"]["macro_f1"], 3)}
                          for n, r in reg.items()}).T
    table.to_csv(out / "model_comparison.csv")
    cnt = counts.reset_index().rename(columns={"corpus": "Corpus"}).set_index("Corpus")
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    pc_df = pd.DataFrame({"F1": [round(v, 3) for v in pcf]}, index=[f"{D.EMOJI[e]} {e}" for e in D.EMOTIONS])
    leaky_md = (f"\n## Why speaker-independent evaluation matters\nSame model type ({leaky['model']}), same RAVDESS data:\n\n"
                f"| Evaluation | Accuracy | Macro-F1 |\n|---|:---:|:---:|\n"
                f"| Random utterance split (same actors in train and test, **leaky**) | {leaky['random_split_accuracy']:.3f} | {leaky['random_split_f1']:.3f} |\n"
                f"| Speaker-independent (unseen actors, **honest**) | {leaky['speaker_independent_accuracy']:.3f} | {leaky['speaker_independent_f1']:.3f} |\n") if leaky else ""
    cross_md = ("\n## Cross-corpus generalisation\nTrained on one corpus, tested on the other (never seen speakers, recording setup or sentences):\n\n"
                "| Direction | Accuracy | Macro-F1 |\n|---|:---:|:---:|\n" +
                "\n".join(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |" for k, v in cross.items()) + "\n") if cross else ""
    sub_md = ("\n## Subgroup performance (test split)\n" + ml.md_table(pd.DataFrame(sub).set_index("Group"), "Group") + "\n") if sub else ""
    (out / "RESULTS.md").write_text(f"""# Results (auto-generated by `train.py`, seed {a.seed})

## Data
{ml.md_table(cnt, "Corpus")}

RAVDESS is split **by actor** (train / validation / test actors never overlap, gender-balanced). TESS has only two speakers, so its split is
by utterance and therefore speaker-dependent: treat TESS numbers as optimistic.

## Model comparison
Models are selected on the **validation speakers**; the test speakers are used once, for reporting.

{ml.md_table(table)}

**Selected model:** {best}

## Selected model on the test split (95% bootstrap confidence intervals)
| Metric | Estimate [95% CI] |
|---|:---:|
| Accuracy | {f(ci['accuracy'])} |
| Macro-F1 | {f(ci['macro_f1'])} |

{ml.md_table(pc_df, "Emotion")}
{sub_md}{leaky_md}{cross_md}""", encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(f"""# Model card: Speech emotion classifier

**Task:** classify a speech clip into 7 emotions ({", ".join(D.EMOTIONS)}).
**Selected model:** {best}, trained on MFCC / delta-MFCC sequences (deep models) or MFCC statistics + prosody (classical models).
Educational demo.

## Intended use
Demonstrating an explainable speech-emotion pipeline. **Not** for hiring, surveillance, mental-health or any decision about people.

## Data
RAVDESS (24 actors, acted North-American English speech) and TESS (2 actresses, acted words). RAVDESS "calm" is merged into "neutral";
TESS "pleasant surprise" is mapped to "surprise". All clips are acted, not spontaneous.

## Evaluation
Test accuracy {f(ci['accuracy'])}, macro-F1 {f(ci['macro_f1'])} on unseen speakers (n={len(te)} clips).
Model selection used validation speakers only.

## Limitations
- Acted emotions in studio conditions; real conversational speech is far harder and the model will be less accurate.
- Very few speakers (24 + 2), mostly North-American English; accents, languages, ages and recording devices outside this are unseen.
- Emotion categories are culturally constructed and speakers express them differently; predictions are probabilities, not facts.
- Small test split: wide confidence intervals; differences between models can be within noise.
{"- Cross-corpus tests show how much accuracy drops on a different corpus (see RESULTS.md)." if cross else ""}

## Ethical considerations
Emotion recognition can be misused. Do not use it to evaluate people (interviews, employees, students) or for surveillance.
""", encoding="utf-8")

    meta = {"classes": D.EMOTIONS, "sr": F.SR, "duration": F.DURATION, "best": best, "best_classical": best_cl,
            "models": {n: {"kind": r["kind"], "file": r["file"], "val_f1": r["val"]["macro_f1"], "val_acc": r["val"]["accuracy"],
                           "test_f1": r["test"]["macro_f1"], "test_acc": r["test"]["accuracy"]} for n, r in reg.items()},
            "ci": {k: list(v) for k, v in ci.items()}, "per_class_f1": dict(zip(D.EMOTIONS, [float(v) for v in pcf])),
            "counts": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                       "corpora": sorted(set(corpus.tolist()))}, "subgroups": sub, "leaky": leaky, "cross": cross}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for fpath in sorted(mdir.glob("*")):
        mb = fpath.stat().st_size / 1e6
        if mb > 50:
            print(f"WARNING: {fpath.name} is {mb:.0f} MB; GitHub rejects files over 100 MB. Delete it from models/ (and meta.json) or use fewer trees.")
    print(f"\n=========== MODEL COMPARISON ===========\n{table}\n\nSaved models, plots and reports in {time.time() - t0:.0f}s")
    return meta


if __name__ == "__main__":
    main()
