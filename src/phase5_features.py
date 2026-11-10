"""Phase 5.2: what the feature space looks like per region, and which feature families each region uses.

    python src/phase5_features.py                    # everything
    python src/phase5_features.py --only space       # PCA / t-SNE / region separability / cosine matrix
    python src/phase5_features.py --only families    # per-region feature-family importance
    python src/phase5_features.py --summarize        # rebuild tables and figures from the saved runs
    options: --n-jobs 4  --seed 42  --perplexity 30  --force  --no-log

TRAINING IMAGES ONLY (3,266 rows, frozen folds). Nothing here is fitted on test rows and nothing here
chooses a model: these are descriptive analyses that explain the Phase 3 / Phase 4 result.

Four questions:
  1. Do the regions occupy different parts of the feature space?  -> PCA / t-SNE pictures, and a region
     classifier (quantitative: t-SNE is never the evidence).
  2. Do the regions need different FRACTURE boundaries?           -> per-region linear SVM weight vectors and
     their cosine similarities. Two regions whose fracture directions are near-parallel can share a model by
     construction; near-orthogonal ones cannot.
  3. Which feature families carry the signal in each region?      -> per-region family ablation (single family
     and leave-one-family-out), RBF SVM at the fixed Phase 3 hyperparameters.
  4. Can the unified model separate the regions by itself?        -> question 1's classifier; if the regions
     are nearly perfectly separable, an RBF SVM can in principle learn region-conditional behaviour without
     being told the region, which bounds what Phase 6's explicit region term can add.

Family subsets rescale the block weights by sqrt(5 / k) for k retained families, so the total feature weight -
and therefore the typical squared distance that gamma multiplies - is what it was for all five families.
Without that, a single-family run would silently be a run at a 5x smaller effective gamma.

Protocol and decision rules: docs/phase5_analysis.md (written before any Phase 5 result).

Outputs: results/phase5/families.csv, embed.csv, separability.json, cosine.csv
         results/tables/phase5_family_importance.csv, phase5_region_separability.csv, phase5_cosine.csv
         results/figures/phase5_feature_space_pca.png, phase5_feature_space_tsne.png,
                         phase5_region_separability.png, phase5_family_importance.png,
                         phase5_fracture_direction.png
"""
import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from evaluate import REGIONS
from expt_log import log_run
from feature_extractors import FAMILIES
from feature_utils import load_features, make_preprocessor
from model_zoo import build, scores
from paths import REPO_ROOT
from phase4_regional import _fast_ap
from phase5_viz import C, DIV, FAMILY_LABEL, REGION_LABEL, SEQ, header, save, setup
import matplotlib.pyplot as plt      # after phase5_viz, which selects the Agg backend

FIXED = {"C": 10, "gamma": 0.05}
MODEL, IMB = "rbfsvm", "none"
LIN_C = 0.1                      # the Phase 2 reference linear SVM, used only for the direction/|w| views
ALL_REGIONS = list(REGIONS) + ["other"]


def load_train_full(tag=None):
    """Like cv_protocol.load_train, but keeps `slices` (which columns belong to which family)."""
    d = load_features(tag) if tag else load_features()
    tr = d["split"] == "train"
    out = dict(X=d["X"][tr], y=d["y"][tr].astype(int), fold=d["fold"][tr], region=d["region"][tr],
               image_id=d["image_id"][tr], col_weights=d["col_weights"], slices=d["slices"],
               config=d["config"])
    del d
    assert len(out["y"]) == 3266 and out["y"].sum() == 576, "unexpected training set; is this the frozen split?"
    return out


def subset(tr, families):
    """Columns of the given families and their block weights, rescaled so the total weight stays 5."""
    cols = np.concatenate([np.arange(tr["slices"][f].start, tr["slices"][f].stop) for f in families])
    w = tr["col_weights"][cols] * np.sqrt(len(FAMILIES) / len(families))
    return cols, w.astype(np.float32)


# ----------------------------------------------------------------------------- 1 & 4: the space itself

def embed(tr, seed, perplexity, n_pcs=50):
    """PCA (fitted on the training rows, after the usual in-fold preprocessing) and a t-SNE of its first PCs."""
    prep = make_preprocessor(tr["col_weights"])
    Z = prep.fit_transform(tr["X"])
    t0 = time.time()
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed).fit(Z)
    P = pca.transform(Z)
    print(f"    PCA: {n_pcs} components, {pca.explained_variance_ratio_[:2].sum():.1%} of the variance in "
          f"PC1-PC2, {pca.explained_variance_ratio_.sum():.1%} in all {n_pcs} ({time.time() - t0:.0f}s)",
          flush=True)
    t0 = time.time()
    T = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(P)
    print(f"    t-SNE: perplexity {perplexity} on the first {n_pcs} PCs ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(dict(image_id=tr["image_id"], region=tr["region"], y=tr["y"], fold=tr["fold"],
                             pc1=P[:, 0], pc2=P[:, 1], tsne1=T[:, 0], tsne2=T[:, 1])), \
        pca.explained_variance_ratio_


def _region_fold(X, region, fold, col_weights, k, seed):
    """One fold of the region classifier. Module level so joblib memmaps X instead of pickling a closure."""
    trn, va = fold != k, fold == k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = make_preprocessor(col_weights).fit(X[trn])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        clf.fit(p.transform(X[trn]), region[trn])
        return va, clf.predict(p.transform(X[va]))


def region_separability(tr, seed, n_jobs):
    """Can the region be read off the features? 5-fold CV over the frozen folds, macro-F1 and a confusion
    matrix. High separability means the unified RBF SVM could learn region-conditional behaviour unaided."""
    res = Parallel(n_jobs=min(n_jobs, 5))(
        delayed(_region_fold)(tr["X"], tr["region"], tr["fold"], tr["col_weights"], k, seed)
        for k in sorted(set(tr["fold"])))
    pred = np.empty(len(tr["y"]), dtype=object)
    for va, pv in res:
        pred[va] = pv
    labels = [r for r in ALL_REGIONS if r in set(tr["region"])]
    cm = confusion_matrix(tr["region"], pred, labels=labels)
    return dict(macro_f1=float(f1_score(tr["region"], pred, average="macro", labels=labels, zero_division=0)),
                accuracy=float((pred == tr["region"]).mean()), labels=labels, confusion=cm.tolist())


def fracture_directions(tr, seed):
    """One linear SVM per region plus one on all training rows; cosine similarity of their weight vectors.
    Descriptive companion to the transfer matrix - the linear shadow of the question the RBF SVM answers."""
    names, W = [], []
    for r in list(REGIONS) + ["all regions"]:
        m = np.ones(len(tr["y"]), bool) if r == "all regions" else tr["region"] == r
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = make_preprocessor(tr["col_weights"]).fit(tr["X"][m])
            clf = LinearSVC(C=LIN_C, dual="auto", max_iter=20000, random_state=seed)
            clf.fit(p.transform(tr["X"][m]), tr["y"][m])
        names.append(r)
        W.append(clf.coef_[0])
    W = np.array(W)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos = pd.DataFrame(Wn @ Wn.T, index=names, columns=names)
    # share of |w| carried by each family, per region
    share = {}
    for n, w in zip(names, W):
        a = np.abs(w)
        share[n] = {f: float(a[tr["slices"][f]].sum() / a.sum()) for f in FAMILIES}
    return cos, pd.DataFrame(share).T


# ----------------------------------------------------------------------------- 3: family importance

def family_jobs(tr):
    """Single family and leave-one-family-out, for each region and for the unified training set."""
    subsets = [("single", (f,)) for f in FAMILIES] + \
              [("leave_one_out", tuple(g for g in FAMILIES if g != f)) for f in FAMILIES] + \
              [("all", tuple(FAMILIES))]
    jobs = []
    for scope in list(REGIONS) + ["all regions"]:
        m = np.ones(len(tr["y"]), bool) if scope == "all regions" else tr["region"] == scope
        ev_region = None if scope == "all regions" else scope
        for kind, fams in subsets:
            for k in sorted(set(tr["fold"])):
                jobs.append(dict(scope=scope, kind=kind, families=fams,
                                 dropped=("-" if kind == "all" else
                                          (fams[0] if kind == "single" else
                                           next(f for f in FAMILIES if f not in fams))),
                                 fold=k, mask=m, ev_region=ev_region))
    return jobs


def _family_one(X, y, fold, cols, w, mask, k, seed):
    """np.ix_ slices rows and columns in one step, so a single-family fit never materialises a copy of the
    full 8,641-column matrix (which, times n_jobs, would not fit comfortably in memory)."""
    trn = np.flatnonzero(mask & (fold != k))
    va = np.flatnonzero(mask & (fold == k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build(MODEL, IMB, FIXED, w, seed).fit(X[np.ix_(trn, cols)], y[trn])
        return _fast_ap(y[va], scores(pipe, X[np.ix_(va, cols)]))


def run_families(tr, args):
    jobs = family_jobs(tr)
    print(f"  family ablation: {len(jobs)} fits "
          f"({len(REGIONS) + 1} scopes x {2 * len(FAMILIES) + 1} subsets x 5 folds)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_family_one)(tr["X"], tr["y"], tr["fold"], *subset(tr, j["families"]), j["mask"], j["fold"],
                             args.seed) for j in jobs)
    rows = [dict(scope=j["scope"], kind=j["kind"], dropped=j["dropped"],
                 families="+".join(j["families"]), n_families=len(j["families"]), fold=j["fold"],
                 n_train=int((j["mask"] & (tr["fold"] != j["fold"])).sum()),
                 chance=float(tr["y"][j["mask"]].mean()), pr_auc=v) for j, v in zip(jobs, res)]
    print(f"  family ablation done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def family_table(fam):
    # dropna=False and the "-" sentinel: an empty "dropped" would be read back from CSV as NaN and
    # groupby would silently drop the all-five reference rows that delta_vs_all is measured against.
    t = (fam.groupby(["scope", "kind", "dropped", "families", "n_families"], as_index=False, dropna=False)
         .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), chance=("chance", "mean")))
    t["dropped"] = t["dropped"].fillna("-")
    base = {r.scope: r.pr_auc for _, r in t[t.kind == "all"].iterrows()}
    t["all_families_pr_auc"] = t.scope.map(base)
    t["delta_vs_all"] = t.pr_auc - t.all_families_pr_auc
    t["lift_over_chance"] = t.pr_auc - t.chance
    return t


# ----------------------------------------------------------------------------- figures

def fig_space(emb, figdir, kind, labels, title, caption, fname):
    """Small multiples rather than one many-coloured scatter: with five regions a single scatter cannot be
    made colour-blind safe, and facets read better anyway. Grey = every other training image."""
    setup()
    x, yc = (f"{kind}1", f"{kind}2")
    fig, axes = plt.subplots(2, len(labels), figsize=(2.45 * len(labels), 5.0), sharex=True, sharey=True)
    for j, lab in enumerate(labels):
        g = emb[emb.region == lab]
        for row, (ax, sel, col, nm) in enumerate((
                (axes[0, j], np.ones(len(g), bool), C["region"], "all images"),
                (axes[1, j], g.y.to_numpy() == 1, C["pos"], "fractured"))):
            ax.scatter(emb[x], emb[yc], s=3, c="#e3e3dd", linewidths=0, rasterized=True)
            if row == 1:
                ax.scatter(g.loc[g.y == 0, x], g.loc[g.y == 0, yc], s=4, c=C["neg"], linewidths=0,
                           label="not fractured", rasterized=True)
            ax.scatter(g[x][sel], g[yc][sel], s=5, c=col, linewidths=0,
                       label=nm if row == 1 else None, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(f"{REGION_LABEL.get(lab, lab.title())} ({len(g)})", fontsize=9.5)
        if j == 0:
            axes[0, j].set_ylabel("all images of the region")
            axes[1, j].set_ylabel("fractured vs not")
    axes[1, 0].scatter([], [], s=18, c="#e3e3dd", label="every other training image")
    header(fig, title, axes[1, 0], ncols=3, top=0.86, legend_y=0.935)
    save(fig, figdir / fname, caption)


def fig_separability(sep, figdir):
    setup()
    labels = sep["labels"]
    cm = np.array(sep["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{int(cm[i, j])}", ha="center", va="center", fontsize=7.5,
                    color=C["surface"] if cmn[i, j] > 0.55 else C["ink"])
    ax.set_xticks(range(len(labels)), [REGION_LABEL.get(r, r.title()) for r in labels], rotation=30,
                  ha="right")
    ax.set_yticks(range(len(labels)), [REGION_LABEL.get(r, r.title()) for r in labels])
    ax.set_xlabel("predicted region")
    ax.set_ylabel("true region")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("share of the row", fontsize=8.5)
    cb.outline.set_visible(False)
    header(fig, f"Region is readable from the features (macro-F1 {sep['macro_f1']:.2f})", top=0.88)
    save(fig, figdir / "phase5_region_separability.png",
         "Multinomial logistic regression predicting the REGION (not the fracture) from the same 8,641 "
         "features, 5-fold CV over the frozen folds; cells show the share of each true region's images and "
         "the count. The more separable the regions are, the more a single unified model could in principle "
         "learn region-conditional behaviour without being told the region.")


def fig_family(ft, figdir):
    setup()
    scopes = list(REGIONS) + ["all regions"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, (kind, title, vcol, cmap, vmin, vmax, note) in zip(axes, [
            ("single", "One family alone: PR-AUC above chance", "lift_over_chance", SEQ, 0, None, ""),
            ("leave_one_out", "Dropping one family: change vs all five", "delta_vs_all", DIV, None, None, "")]):
        M = np.full((len(scopes), len(FAMILIES)), np.nan)
        for i, s in enumerate(scopes):
            for j, f in enumerate(FAMILIES):
                g = ft[(ft.scope == s) & (ft.kind == kind) & (ft.dropped == f)]
                if len(g):
                    M[i, j] = g[vcol].iloc[0]
        if kind == "leave_one_out":
            lim = np.nanmax(np.abs(M)) or 0.01
            vmin, vmax = -lim, lim
        else:
            vmax = np.nanmax(M)
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(scopes)):
            for j in range(len(FAMILIES)):
                if np.isnan(M[i, j]):
                    continue
                far = (M[i, j] > 0.72 * vmax) if kind == "single" else (abs(M[i, j]) > 0.62 * vmax)
                ax.text(j, i, f"{M[i, j]:+.3f}" if kind == "leave_one_out" else f"{M[i, j]:.3f}",
                        ha="center", va="center", fontsize=7.5,
                        color=C["surface"] if far else C["ink"])
        ax.set_xticks(range(len(FAMILIES)), [FAMILY_LABEL[f] for f in FAMILIES])
        ax.set_yticks(range(len(scopes)), [REGION_LABEL.get(s, "All regions") for s in scopes])
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.outline.set_visible(False)
    header(fig, "Which feature families carry the signal, per region", top=0.86)
    save(fig, figdir / "phase5_family_importance.png",
         "5-fold CV PR-AUC on that scope's training rows, RBF SVM at the fixed Phase 3 hyperparameters, with "
         "the block weights rescaled so a subset carries the same total weight as all five families. LEFT: a "
         "family on its own, minus that scope's chance level (regions have different chance levels, so raw "
         "PR-AUC is not comparable down a column). RIGHT: all five minus that one; near zero means the family "
         "is redundant given the others, not that it is uninformative. Plain CV, not nested - a descriptive "
         "ranking within a row, not a model choice.")


def fig_cosine(cos, share, figdir):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), width_ratios=[1, 1.05])
    names = list(cos.index)
    M = cos.to_numpy()
    im = axes[0].imshow(M, cmap=DIV, vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                         color=C["surface"] if abs(M[i, j]) > 0.62 else C["ink"])
    lab = [REGION_LABEL.get(n, "All regions") for n in names]
    axes[0].set_xticks(range(len(names)), lab, rotation=30, ha="right")
    axes[0].set_yticks(range(len(names)), lab)
    axes[0].set_title("Cosine similarity of the fracture directions", fontsize=9.5)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)

    S = share.loc[names]
    bottom = np.zeros(len(names))
    ramp = [SEQ(v) for v in np.linspace(0.25, 0.95, len(FAMILIES))]
    for f, col in zip(FAMILIES, ramp):
        axes[1].barh(range(len(names)), S[f].to_numpy(), left=bottom, color=col, label=FAMILY_LABEL[f],
                     edgecolor=C["surface"], linewidth=1.2, height=0.7)
        bottom += S[f].to_numpy()
    axes[1].set_yticks(range(len(names)), lab)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("share of total |w|")
    axes[1].set_title("Where each region's linear weight sits", fontsize=9.5)
    axes[1].grid(axis="x", visible=True)
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(ncols=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for s in axes[0].spines.values():
        s.set_visible(False)
    axes[0].grid(False)
    header(fig, "Do the regions need different fracture boundaries?", top=0.86)
    save(fig, figdir / "phase5_fracture_direction.png",
         "Linear SVM (C=0.1, block-weighted features) fitted on each region's training rows and on all of "
         "them. LEFT: cosine similarity of the weight vectors - 1.00 means two regions want the same linear "
         "boundary, 0.00 means unrelated directions. In 8,641 dimensions two unrelated vectors sit near 0, so "
         "read the ordering, not the absolute size. RIGHT: how each region's weight mass is split across the "
         "feature families. HOG is 8,100 of the 8,641 columns, so a large HOG share is expected and the share is "
         "only worth comparing BETWEEN regions, never between families within a row - the family ablation is "
         "the proper per-family measure. Descriptive only: this is the linear shadow of the RBF model used.")


# ----------------------------------------------------------------------------- main

def summarize(out_dir, tables_dir, figures_dir):
    pd.set_option("display.width", 220, "display.max_columns", 40)
    ep, fp, sp, cp = (out_dir / "embed.csv", out_dir / "families.csv",
                      out_dir / "separability.json", out_dir / "cosine.csv")
    tables_dir.mkdir(parents=True, exist_ok=True)
    if ep.exists():
        emb = pd.read_csv(ep)
        labs = [r for r in ALL_REGIONS if r in set(emb.region)]
        fig_space(emb, figures_dir, "pc", labs, "Feature space by region and fracture status (PCA)",
                  "Principal components 1 and 2 of the 8,641 block-weighted features, fitted on the training "
                  "rows. Grey points are every other training image, so each panel shows where that region "
                  "sits inside the whole set. PCA is a linear projection, so overlap here does not prove the "
                  "regions are inseparable - the region classifier does that test properly.",
                  "phase5_feature_space_pca.png")
        fig_space(emb, figures_dir, "tsne", labs, "Feature space by region and fracture status (t-SNE)",
                  "t-SNE of the first 50 principal components. Used for looking only: t-SNE distances are not "
                  "metric and its clusters shift with perplexity, so no claim in this phase rests on it. The "
                  "quantitative versions of this picture are the region classifier and the transfer matrix.",
                  "phase5_feature_space_tsne.png")
    if sp.exists():
        sep = json.loads(sp.read_text())
        pd.DataFrame(sep["confusion"], index=sep["labels"], columns=sep["labels"]) \
            .to_csv(tables_dir / "phase5_region_separability.csv")
        fig_separability(sep, figures_dir)
        print(f"\n=== Region separability from the features: macro-F1 {sep['macro_f1']:.3f}, "
              f"accuracy {sep['accuracy']:.3f} (5-fold CV) ===")
    if cp.exists():
        cos = pd.read_csv(cp, index_col=0)
        share = pd.read_csv(out_dir / "wshare.csv", index_col=0)
        cos.round(4).to_csv(tables_dir / "phase5_cosine.csv")
        fig_cosine(cos, share, figures_dir)
        print("\n=== Cosine similarity of the per-region linear fracture directions ===")
        print(cos.round(3).to_string())
    if fp.exists():
        ft = family_table(pd.read_csv(fp))
        ft.round(4).to_csv(tables_dir / "phase5_family_importance.csv", index=False)
        fig_family(ft, figures_dir)
        print("\n=== Single feature family alone: PR-AUC minus that scope's chance level ===")
        print(ft[ft.kind == "single"].pivot_table(index="scope", columns="dropped",
                                                  values="lift_over_chance").round(3).to_string())
        print("\n=== Leave one family out: PR-AUC minus the all-five value (negative = the family helped) ===")
        print(ft[ft.kind == "leave_one_out"].pivot_table(index="scope", columns="dropped",
                                                         values="delta_vs_all").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["space", "families"], default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase5")
    ap.add_argument("--tables-dir", type=Path, default=REPO_ROOT / "results" / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize:
        tr = load_train_full(args.tag)
        dims = ", ".join("%s %d" % (f, tr["slices"][f].stop - tr["slices"][f].start) for f in FAMILIES)
        print(f"{len(tr['y'])} training images, {tr['y'].sum()} fractured, {tr['X'].shape[1]} features "
              f"({dims})", flush=True)
        if args.only in (None, "space"):
            if (args.out_dir / "embed.csv").exists() and not args.force:
                print("[skip] embed.csv exists; --force to redo")
            else:
                print("  embedding (PCA + t-SNE)...", flush=True)
                emb, evr = embed(tr, args.seed, args.perplexity)
                emb.to_csv(args.out_dir / "embed.csv", index=False)
                np.save(args.out_dir / "pca_explained_variance.npy", evr)
                print("  region separability...", flush=True)
                sep = region_separability(tr, args.seed, args.n_jobs)
                (args.out_dir / "separability.json").write_text(json.dumps(sep, indent=1))
                print(f"    macro-F1 {sep['macro_f1']:.3f}, accuracy {sep['accuracy']:.3f}", flush=True)
                print("  per-region fracture directions...", flush=True)
                cos, share = fracture_directions(tr, args.seed)
                cos.to_csv(args.out_dir / "cosine.csv")
                share.to_csv(args.out_dir / "wshare.csv")
                if not args.no_log:
                    log_run(regions="all (descriptive, per region)",
                            feature_set=f"all five families [{args.tag or 'default'}]",
                            model="PCA + t-SNE; multinomial logistic (region); linear SVM C=0.1 (direction)",
                            imbalance_strategy="n/a (descriptive)",
                            key_hyperparameters=f"PCA 50 comps; t-SNE perplexity {args.perplexity}; "
                                                f"seed={args.seed}",
                            cv_score=f"region macro-F1={sep['macro_f1']:.3f} (5-fold)",
                            test_metrics="not used (CV only)",
                            notes="Phase 5.2 feature-space view, region separability and per-region "
                                  "fracture directions; training rows only")
        if args.only in (None, "families"):
            p = args.out_dir / "families.csv"
            if p.exists() and not args.force:
                print("[skip] families.csv exists; --force to redo")
            else:
                fam = run_families(tr, args)
                fam.to_csv(p, index=False)
                if not args.no_log:
                    ft = family_table(fam)
                    best = ft[ft.kind == "single"].sort_values("lift_over_chance", ascending=False)
                    log_run(regions="per region + all regions",
                            feature_set="family subsets (single and leave-one-out), rescaled block weights",
                            model="RBF SVM (fixed C=10, gamma=0.05)", imbalance_strategy="none (PR-AUC only)",
                            key_hyperparameters=f"plain 5-fold CV over the frozen folds; seed={args.seed}",
                            cv_score="; ".join(f"{r.scope}: best single family {r.dropped}"
                                               for _, r in best.drop_duplicates("scope").iterrows()),
                            test_metrics="not used (CV only)",
                            notes="Phase 5.2 per-region feature-family importance (descriptive ranking, "
                                  "not nested, no model chosen from it)")

    summarize(args.out_dir, args.tables_dir, args.figures_dir)


if __name__ == "__main__":
    main()
