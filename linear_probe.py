# Linear probe router: logistic regression on frozen all-MiniLM-L6-v2 embeddings with one
# independent sigmoid output per candidate model.
# Reads results/labels.csv, the train/val/test split in results/splits.csv and the embedding cache
# results/embeddings.npz and writes results/linear/curve.csv and results/linear/accuracy_vs_cost.png.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).parent / 'results'
OUT = ROOT / 'linear'

MODELS = ['llama-8b', 'llama-70b']   # cheapest firs, routing picks the first model over threshold
K = len(MODELS)
SEED = 0
EPOCHS, LR = 300, 1e-2
WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1]   # picked on the validation split
THRESHOLDS = np.linspace(0, 1, 101)
TARGET_GAP, TARGET_COST = 3.0, 0.40


def load():
    # Labels joined with the split column and the (n, 384) embedding matrix, in labels.csv order
    labels = pd.read_csv(ROOT / 'labels.csv').join(pd.read_csv(ROOT / 'splits.csv').set_index('qid'), on='qid')
    z = np.load(ROOT / 'embeddings.npz', allow_pickle=True)
    X = pd.DataFrame(z['X'], index=z['qid']).loc[labels.qid].to_numpy()
    return labels, X


def train(X, Y, weight_decay):
    # Full-batch Adam on a single Linear layer. Loss is BCE on each output, summed across models.
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    torch.manual_seed(SEED)
    model = nn.Linear(X.shape[1], K)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = bce(model(X), Y).sum(1).mean()
        loss.backward()
        opt.step()
    return model


def predict(model, X):
    # (n, K) predicted P(correct) for each model
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def route(P, t):
    # Index of the cheapest model with P(correct) > t, falling back to the most expensive
    ok = P > t
    return np.where(ok.any(1), ok.argmax(1), K - 1)


def oracle_choice(bits):
    # Cheapest model that answered correctly and the cheapest model if none did
    return np.where(bits.any(1), bits.argmax(1), 0)


def evaluate(choice, bits, cost):
    rows = np.arange(len(choice))
    return {'acc': bits[rows, choice].mean(),
            'cost_frac': cost[rows, choice].sum() / cost[:, -1].sum(),
            'routing_acc': (choice == oracle_choice(bits)).mean(),
            'frac_expensive': (choice == K - 1).mean()}


def curve(choices, bits, cost):
    # One evaluate() row per threshold
    return pd.DataFrame([{'t': t, **evaluate(c, bits, cost)} for t, c in zip(THRESHOLDS, choices)])


def plot(probe, rand, length, points, path):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    ax.plot(probe.cost_frac, probe.acc, color='#2a78d6', lw=2, label='linear probe')
    ax.plot(rand.cost_frac, rand.acc, color='#eb6834', lw=2, label='random')
    ax.plot(length.cost_frac, length.acc, color='#1baf7a', lw=2, label='length heuristic')
    for label, p in points.items():
        ax.plot(p['cost_frac'], p['acc'], 'o', ms=8, color='#0b0b0b')
        ax.annotate(label, (p['cost_frac'], p['acc']), xytext=(6, 4), textcoords='offset points',
                    fontsize=9, color='#52514e', bbox=dict(boxstyle='round,pad=0.15', fc='#fcfcfb', ec='none'))
    ax.set_xlabel('cost as fraction of always-expensive', color='#52514e')
    ax.set_ylabel('task accuracy', color='#52514e')
    ax.set_title('Linear probe: accuracy vs cost on the test split', color='#0b0b0b', loc='left')
    ax.grid(color='#e6e5e1', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    labels, X = load()
    bits = labels[MODELS].to_numpy()
    cost = labels[[f'cost_{m}' for m in MODELS]].to_numpy()
    tr, va, te = (labels.split.to_numpy() == s for s in ('train', 'val', 'test'))
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)   # standardize with train statistics

    # Pick weight decay by validation loss, then train the final probe
    def val_bce(wd):
        P = np.clip(predict(train(X[tr], bits[tr], wd), X[va]), 1e-6, 1 - 1e-6)
        return -(bits[va] * np.log(P) + (1 - bits[va]) * np.log(1 - P)).sum(1).mean()
    wd = min(WEIGHT_DECAYS, key=val_bce)
    model = train(X[tr], bits[tr], wd)

    # Threshold sweep on the test split
    b, c = bits[te], cost[te]
    n = len(b)
    P = predict(model, X[te])
    probe = curve([route(P, t) for t in THRESHOLDS], b, c)
    probe.to_csv(OUT / 'curve.csv', index=False)

    # Baselines on the same test rows
    points = {'always-cheap': evaluate(np.zeros(n, int), b, c),
              'always-expensive': evaluate(np.full(n, K - 1), b, c),
              'oracle': evaluate(oracle_choice(b), b, c)}
    u = np.random.default_rng(SEED).random(n)   # one draw, so the random curve is nested in cost
    rand = curve([np.where(u < f, K - 1, 0) for f in THRESHOLDS], b, c)
    rank = labels.question[te].str.len().rank(pct=True).to_numpy()   # longest questions go expensive first
    length = curve([np.where(rank > 1 - f, K - 1, 0) for f in THRESHOLDS], b, c)

    # Headline operating points: best accuracy under the cost cap, and cheapest within the accuracy gap
    exp_acc = points['always-expensive']['acc']
    under_cost = probe[probe.cost_frac <= TARGET_COST]
    within_gap = probe[probe.acc >= exp_acc - TARGET_GAP / 100]
    rows = {**points,
            f'linear @ cost<={TARGET_COST:.0%}': under_cost.loc[under_cost.acc.idxmax()].to_dict(),
            f'linear @ acc>=exp-{TARGET_GAP:g}pts': within_gap.loc[within_gap.cost_frac.idxmin()].to_dict()}
    table = pd.DataFrame(rows).T[['t', 'acc', 'cost_frac', 'routing_acc', 'frac_expensive']]
    print(f'{n} test questions, weight decay {wd:g}\n')
    print(table.round(3).to_string())

    best = rows[f'linear @ cost<={TARGET_COST:.0%}']
    gap = 100 * (exp_acc - best['acc'])
    print(f"\nlinear probe at <= {TARGET_COST:.0%} cost: {100 * best['acc']:.1f}% vs always-expensive "
          f"{100 * exp_acc:.1f}%, gap {gap:.1f} pts ({'meets' if gap <= TARGET_GAP else 'misses'} the {TARGET_GAP:g}-pt target).")

    plot(probe, rand, length, points, OUT / 'accuracy_vs_cost.png')


if __name__ == '__main__':
    main()
