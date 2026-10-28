"""
MLP probe router: frozen query embeddings -> shared hidden layer -> P(model answers correctly) for each model.
Reads results/labels.csv and results/questions.csv. Trained with binary cross entropy and Adam.

First routing rule routes to the cheapest model whose predicted probability exceeds some threshold t, otherwise
routes to the most expensive model. Second routing rule routes to the expensive model if
P(expensive answers correct) - P(cheap answers correct) > t. We sweep t over all possible values and compare against baselines.
"""
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

# Settings
ROOT = Path(__file__).parent / 'results'
OUT = ROOT / 'mlp'
SEED = 0
SPLIT = {'train': 0.6, 'val': 0.2, 'test': 0.2}
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

MODELS = ['llama-8b', 'llama-70b'] # cheapest to most expensive
TARGET_GAP, TARGET_COST = 3.0, 0.40

GRID = {'hidden': [32, 64, 128], 'lr': [1e-3, 3e-3], 'weight_decay': [0.0, 1e-4, 1e-3], 'dropout': [0.1, 0.3]}
SEEDS = [0, 1, 2]
EPOCHS, PATIENCE, BATCH = 300, 20, 32
STEP = 0.01

# Data
def load():
    """
    Build the per-question table: joins the pilot's correctness bits and costs with the 
    MMLU-Pro answer options, text to embed, question splits, and response latencies.
    """
    labels = pd.read_csv(ROOT / 'labels.csv').set_index('qid')
    Q = pd.read_csv(ROOT / 'questions.csv').set_index('qid')
    df = labels.join(Q[['options']])
    df['text'] = [query_text(r) for r in df.itertuples()]
    return df.join(splits(df)).join(latencies())


def query_text(r):
    """
    Return text to embed for one question (MMLU-Pro questions include answer options).
    """
    if r.dataset != 'mmlupro':
        return r.question
    opts = '\n'.join(f'{"ABCDEFGHIJ"[i]}. {o}' for i, o in enumerate(json.loads(r.options)))
    return f'{r.question}\n\n{opts}'


def splits(df):
    """
    Return each question's split (train / val / test). Saved to results/splits.csv.
    """
    path = ROOT / 'splits.csv'
    if not path.exists():
        rng = np.random.default_rng(SEED)
        rows = []
        for _, g in df.groupby('dataset'):
            n = len(g)
            n_train, n_val = round(n * SPLIT['train']), round(n * SPLIT['val'])
            names = ['train'] * n_train + ['val'] * n_val + ['test'] * (n - n_train - n_val)
            rows += list(zip(g.index[rng.permutation(n)], names))
        pd.DataFrame(rows, columns=['qid', 'split']).to_csv(path, index=False)
    return pd.read_csv(path).set_index('qid')


def latencies():
    """
    Return each model's response time per question.
    """
    lat = pd.concat([pd.read_json(ROOT / f'{m}.jsonl', lines=True).set_index('qid')['latency'].rename(f'latency_{m}')
                     for m in MODELS], axis=1)
    return lat


def embeddings(df):
    """
    Return one 384-dimensional embedding per question. Cached in results/embeddings.npz.
    """
    path = ROOT / 'embeddings.npz'
    if path.exists():
        z = np.load(path, allow_pickle=True)
        cached = dict(zip(z['qid'], z['X']))
        if set(df.index) <= set(cached):
            return np.stack([cached[q] for q in df.index])
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer(EMBED_MODEL, device='cpu').encode(df['text'].tolist(), batch_size=64)
    np.savez(path, qid=df.index.to_numpy(dtype=object), X=X)
    return X


def embed_latency(texts):
    """
    Return time to embed each text.
    """
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device='cpu')
    enc.encode(texts[:2])
    out = []
    for t in texts:
        t0 = time.perf_counter()
        enc.encode([t])
        out.append(time.perf_counter() - t0)
    return np.array(out)


def bce_summed(logits, Y):
    """
    Computes training loss: binary cross-entropy for each output, summed over the models and
    averaged over the questions in batch.
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction='none').sum(1).mean()


def train(X, Y, Xv, Yv, hidden, lr, weight_decay, dropout, seed):
    """
    Train one MLP probe and return (network, best validation loss).

    The network is Linear -> ReLU -> Dropout -> Linear with one output (logit) per model, trained
    with Adam on mini-batches of the training set (X, Y).
    """
    torch.manual_seed(seed)
    X, Y, Xv, Yv = (torch.tensor(a, dtype=torch.float32) for a in (X, Y, Xv, Yv))
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float('inf'), None, 0
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(X), generator=g).split(BATCH):
            opt.zero_grad()
            bce_summed(net(X[b]), Y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = bce_summed(net(Xv), Yv).item()
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def predict(nets, X):
    """
    Return P(model answers correctly) for every row of X.
    """
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(X)).numpy() for n in nets], axis=0)


def route(P, t):
    """First routing rule: send each question to the cheapest model whose predicted
    P(correct) exceeds t, otherwise to the most expensive model if none does.
    """
    k = P.shape[1]
    above = (P[:, :k - 1] > t) | (t <= 0)
    return np.where(above.any(1), above.argmax(1), k - 1)


def route_gain(P, t):
    """
    Second routing rule (gain rule): send a question to the expensive model only when it is
    predicted to beat the cheap model by more than t, i.e. P(expensive) - P(cheap) > t.
    """
    gain = P[:, 1] - P[:, 0]
    return ((gain > t) | (t <= -1)).astype(int)


GAIN_TS = np.round(np.arange(-1, 1 + STEP / 2, STEP), 6)


def oracle_choice(C):
    """
    Perfect router given the true correctness bits C. Routes to cheapest model that 
    answers correctly, or the cheapest model when no model answers correctly.
    """
    return np.where(C.any(1), C.argmax(1), 0)


def metrics(choice, C, K, L, router_latency=0.0):
    """
    Score one routing decision per question.
    """
    idx = np.arange(len(C))
    orc = oracle_choice(C)
    any_ok = C.any(1)
    out = {
        'task_acc': C[idx, choice].mean(),
        'cost_frac': K[idx, choice].sum() / K[:, -1].sum(),
        'routing_acc': (choice == orc).mean(),
        'latency_s': (L[idx, choice] + router_latency).mean(),
        'under_routed': ((choice < orc) & any_ok).mean(),
        'over_routed': ((choice > orc) & any_ok).mean(),
        'none_solve_sent_expensive': ((choice > 0) & ~any_ok).mean(),
    }
    for j, m in enumerate(MODELS):
        out[f'frac_{m}'] = (choice == j).mean()
    return out


def sweep(P, C, K, L, router_latency=0.0, rule=route, ts=None):
    """Apply routing rule at every threshold in ts to get router's accuracy-vs-cost curve."""
    if ts is None:
        ts = np.round(np.arange(0, 1 + STEP / 2, STEP), 6)
    return pd.DataFrame([{'t': t, **metrics(rule(P, t), C, K, L, router_latency)} for t in ts])


def pick_t(curve, acc_ref):
    """Return the cheapest threshold on a curve whose accuracy is within TARGET_GAP points of acc_ref."""
    ok = curve[curve.task_acc >= acc_ref - TARGET_GAP / 100 - 1e-12]
    return None if ok.empty else float(ok.sort_values(['cost_frac', 't']).iloc[0].t)


def oracle_gap(curve, C, K):
    """
    Average accuracy shortfall of a router's curve below the oracle curve.
    """
    L0 = np.zeros_like(K)
    orc = metrics(oracle_choice(C), C, K, L0)
    cheap = metrics(np.zeros(len(C), int), C, K, L0)
    xs = np.linspace(cheap['cost_frac'], 1.0, 401)
    best = np.interp(xs, [cheap['cost_frac'], orc['cost_frac'], 1.0],
                     [cheap['task_acc'], orc['task_acc'], max(orc['task_acc'], C[:, -1].mean())])
    c = curve.sort_values('cost_frac')
    router = np.array([c.task_acc[c.cost_frac <= x + 1e-12].max() if (c.cost_frac <= x + 1e-12).any()
                       else cheap['task_acc'] for x in xs])
    return float(np.mean(best - router))


def main():
    """Run MLP probe experiment and write results/mlp/.

    1. Load pilot data, embed questions, and standardize embeddings.
    2. Search hyperparameters by validation loss, train best settings with every seed.
    3. Measure router's latency per test question.
    4. Sweep thresholds for routing rules.
    5. Compute test baselines: always routes to cheap, always routes to expensive, oracle, random, length heuristic.
    6. Pick rules' thresholds on validation; apply to test.
    7. Save results.
    """
    OUT.mkdir(exist_ok=True)
    df = load()
    X = embeddings(df)
    C = df[MODELS].to_numpy(int)
    K = df[[f'cost_{m}' for m in MODELS]].to_numpy(float)
    L = df[[f'latency_{m}' for m in MODELS]].to_numpy(float)
    S = {s: (df.split == s).to_numpy() for s in SPLIT}
    print({s: int(m.sum()) for s, m in S.items()})

    mu, sd = X[S['train']].mean(0), X[S['train']].std(0) + 1e-6
    Z = (X - mu) / sd
    tr, va, te = S['train'], S['val'], S['test']

    grid = []
    for values in itertools.product(*GRID.values()):
        hp = dict(zip(GRID, values))
        _, v = train(Z[tr], C[tr], Z[va], C[va], seed=0, **hp)
        grid.append({**hp, 'val_bce': v})
    grid = pd.DataFrame(grid).sort_values('val_bce')
    hp = grid.iloc[0][list(GRID)].to_dict()
    hp['hidden'] = int(hp['hidden'])
    print('best hyper-parameters:', hp)
    nets = [train(Z[tr], C[tr], Z[va], C[va], seed=s, **hp)[0] for s in SEEDS]
    P = predict(nets, Z)

    emb_lat = embed_latency(df.text[te].tolist())
    t0 = time.perf_counter()
    for _ in range(100):
        predict(nets, Z[:1])
    router_lat = emb_lat + (time.perf_counter() - t0) / 100

    curves = {s: sweep(P[S[s]], C[S[s]], K[S[s]], L[S[s]], router_lat if s == 'test' else router_lat.mean())
              for s in ('val', 'test')}
    gain_curves = {s: sweep(P[S[s]], C[S[s]], K[S[s]], L[S[s]], router_lat if s == 'test' else router_lat.mean(),
                            rule=route_gain, ts=GAIN_TS)
                   for s in ('val', 'test')}

    Ct, Kt, Lt = C[te], K[te], L[te]
    n = len(Ct)
    base = {'always_cheapest': metrics(np.zeros(n, int), Ct, Kt, Lt),
            'always_most_expensive': metrics(np.full(n, len(MODELS) - 1), Ct, Kt, Lt),
            'oracle': metrics(oracle_choice(Ct), Ct, Kt, Lt)}
    rand = pd.concat([sweep(np.random.default_rng(s).uniform(size=(n, len(MODELS))), Ct, Kt, Lt)
                      for s in range(20)]).groupby('t', as_index=False).mean()
    train_len = np.sort(df.text[tr].str.len().to_numpy())
    pct = np.searchsorted(train_len, df.text[te].str.len().to_numpy(), side='right') / (len(train_len) + 1)
    length = sweep(np.stack([1 - pct, np.ones(n)], 1), Ct, Kt, Lt)

    acc_exp_val = C[va, -1].mean()

    def operating_point(curve_val, rule):
        """
        Pick a threshold on the validation curve with pick_t and return test metrics at that
        threshold, plus accuracy gap to always-most-expensive (in points), and whether the target is met
        """
        t = pick_t(curve_val, acc_exp_val)
        if t is None:
            return None
        m = metrics(rule(P[te], t), Ct, Kt, Lt, router_lat)
        m['t'] = t
        m['acc_gap_pts'] = 100 * (base['always_most_expensive']['task_acc'] - m['task_acc'])
        m['meets_target'] = bool(m['acc_gap_pts'] <= TARGET_GAP + 1e-9 and m['cost_frac'] <= TARGET_COST)
        return m

    at = operating_point(curves['val'], route)
    at_gain = operating_point(gain_curves['val'], route_gain)
    t_star = at['t'] if at else None

    from sklearn.metrics import roc_auc_score
    summary = {
        'split_sizes': {s: int(m.sum()) for s, m in S.items()},
        'hyperparameters': hp,
        'test_auroc_cheap_head': float(roc_auc_score(Ct[:, 0], P[te, 0])) if 0 < Ct[:, 0].sum() < n else None,
        'baselines_test': base,
        'mlp_test_at_t_star': at,
        'mlp_gain_rule_test_at_t_star': at_gain,
        'mlp_test_mean_gap_to_oracle_curve': oracle_gap(curves['test'], Ct, Kt),
        'mlp_gain_rule_test_mean_gap_to_oracle_curve': oracle_gap(gain_curves['test'], Ct, Kt),
        'random_test_mean_gap_to_oracle_curve': oracle_gap(rand, Ct, Kt),
        'length_test_mean_gap_to_oracle_curve': oracle_gap(length, Ct, Kt),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    pd.concat([c.assign(split=s, rule=rule) for rule, cs in (('threshold', curves), ('gain', gain_curves))
               for s, c in cs.items()]).to_csv(OUT / 'curves.csv', index=False)
    grid.to_csv(OUT / 'grid.csv', index=False)

    orc = oracle_choice(Ct)
    queries = pd.DataFrame({'qid': df.index[te], 'dataset': df.dataset[te].values, 'subject': df.subject[te].values,
                            'p_cheap_correct': P[te, 0], 'p_expensive_correct': P[te, 1],
                            'oracle': np.array(MODELS)[orc]})
    for rule, fn, m in (('threshold', route, at), ('gain', route_gain, at_gain)):
        if m is None:
            continue
        ch = fn(P[te], m['t'])
        queries[f'chosen_{rule}'] = np.array(MODELS)[ch]
        queries[f'outcome_{rule}'] = np.select([~Ct.any(1), ch == orc, ch < orc],
                                               ['no model solves', 'optimal', 'under-routed'], 'over-routed')
    queries.to_csv(OUT / 'test_queries.csv', index=False)

    plot(curves['test'], gain_curves['test'], rand, length, base, n)

    print(f"\ntest n={n}")
    for k, v in base.items():
        print(f"  {k:24s} acc {v['task_acc']:.3f}  cost {v['cost_frac']:.3f}")
    for label, m in (('MLP, threshold rule', at), ('MLP, gain rule', at_gain)):
        if m:
            print(f"  {label:24s} acc {m['task_acc']:.3f}  cost {m['cost_frac']:.3f}  t*={m['t']:.2f}  "
                  f"gap {m['acc_gap_pts']:.1f} pts  routing acc {m['routing_acc']:.3f}  "
                  f"to 70B {m['frac_llama-70b']:.2f}  meets target: {m['meets_target']}")
        else:
            print(f'  {label}: no threshold on validation came within the target gap')
    print(f"  mean gap to oracle curve: MLP threshold {summary['mlp_test_mean_gap_to_oracle_curve']:.3f}, "
          f"MLP gain {summary['mlp_gain_rule_test_mean_gap_to_oracle_curve']:.3f}, "
          f"length {summary['length_test_mean_gap_to_oracle_curve']:.3f}, "
          f"random {summary['random_test_mean_gap_to_oracle_curve']:.3f}")


def plot(mlp, mlp_gain, rand, length, base, n):
    """
    Plot the test-set accuracy-vs-cost figure and save it to results/mlp/accuracy_vs_cost.png.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.grid(True, color='#e6e5e0')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.plot(mlp.cost_frac, mlp.task_acc, color='#eb6834', lw=2, label='MLP probe, threshold rule')
    ax.plot(mlp_gain.cost_frac, mlp_gain.task_acc, color='#2a78d6', lw=2, label='MLP probe, gain rule')
    ax.plot(length.cost_frac, length.task_acc, color='#eda100', lw=2, label='Length heuristic')
    ax.plot(rand.cost_frac, rand.task_acc, color='#8a8a85', lw=2, ls='--', label='Random')
    for name, marker, label in [('always_cheapest', 'o', 'Always 8B'), ('always_most_expensive', 's', 'Always 70B'),
                                ('oracle', '*', 'Oracle')]:
        p = base[name]
        ax.scatter(p['cost_frac'], p['task_acc'], marker=marker, s=200 if marker == '*' else 50,
                   color='#1f1f1e', edgecolor='white', lw=1.5, zorder=5, label=label)
    ax.set_xlabel('Total cost (fraction of always-70B)')
    ax.set_ylabel('Task accuracy')
    ax.set_title(f'MLP probe: accuracy vs cost (test, n={n})', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'accuracy_vs_cost.png')


if __name__ == '__main__':
    main()
