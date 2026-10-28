"""Pilot: oracle headroom for Llama 3.1 8B vs Llama 3.3 70B on MATH and MMLU-Pro.
Queries both models on N questions per dataset (resuming from results/*.jsonl),
grades them, prints the oracle headroom table from the proposal, and writes
results/labels.csv, the training table for the linear and MLP probes.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify
from openai import OpenAI

ROOT = Path(__file__).parent / 'results'
ROOT.mkdir(exist_ok=True)
DEEPINFRA_API_KEY = "ASDASD"
SEED, N = 0, 200
MAX_TOKENS = 1024
MAX_WORKERS = 16

# DeepInfra model id and UsD per 1M input tokens USD per 1M output tokens
MODELS = {
    'llama-8b':  ('meta-llama/Meta-Llama-3.1-8B-Instruct', 0.02, 0.03),
    'llama-70b': ('meta-llama/Llama-3.3-70B-Instruct',     0.10, 0.32),
}
CHEAP, EXPENSIVE = 'llama-8b', 'llama-70b'

# Hypothesis target from the proposal which is within 3 pts of always expensive at less than 40% of its cost
TARGET_GAP, TARGET_COST = 3.0, 0.40

MATH_SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra','number_theory', 'prealgebra', 'precalculus']


# questions
def load_questions():
   # Question table indexed by qid
    qpath = ROOT / 'questions.csv'
    if not qpath.exists():
        m = concatenate_datasets([load_dataset('EleutherAI/hendrycks_math', s, split='test')
                                  for s in MATH_SUBJECTS]).shuffle(seed=SEED).select(range(N))
        p = load_dataset('TIGER-Lab/MMLU-Pro', split='test').shuffle(seed=SEED).select(range(N))
        pd.concat([
            pd.DataFrame({'qid': [f'math_{i}' for i in range(N)], 'dataset': 'math',
                          'question': m['problem'], 'subject': m['type'],
                          'truth': m['solution'], 'options': None}),
            pd.DataFrame({'qid': [f'mmlupro_{i}' for i in range(N)], 'dataset': 'mmlupro',
                          'question': p['question'], 'subject': p['category'],
                          'truth': p['answer'], 'options': [json.dumps(o) for o in p['options']]}),
        ], ignore_index=True).to_csv(qpath, index=False)
    return pd.read_csv(qpath).set_index('qid')


# prompt takes rwo and returns a prompt to be sent to model
def prompt(r):
    if r.dataset == 'math':
        return (f"Solve this problem. Reason briefly, then put the final answer on the last line "
                f"as \\boxed{{answer}}.\n\n{r.question}")
    opts = '\n'.join(f'{"ABCDEFGHIJ"[i]}. {o}' for i, o in enumerate(json.loads(r.options)))
    return (f"Answer this multiple choice question. Reason briefly, then end with a line "
            f"\"Answer: X\".\n\n{r.question}\n\n{opts}")


def grade(r, text):
    # Return (pred, truth, correct) for a model response
    if r.dataset == 'math':
        # parse() finds the last \boxed{} and returns [sympy_obj, latex_str] or [] if none
        truth, pred = parse(r.truth), parse(text)
        if len(pred) > 1:
            pred_s = pred[1]
        else:
            pred_s = None

        if len(truth) > 1:
            truth_s = truth[1]
        else:
            truth_s = None
        return pred_s, truth_s, int(bool(pred) and bool(truth) and verify(truth, pred))
    m = re.findall(r'[Aa]nswer\s*[:\-]?\s*\(?([A-J])\)?', text)
    pred = m[-1].upper() if m else None
    return pred, r.truth, int(pred == r.truth)


def ask(client, key, r):
    # API call and Runs in a worker thread
    mid, pin, pout = MODELS[key]
    mid, pin, pout = MODELS[key]
    t0 = time.time()
    # builds the prompt and sends it to the model
    resp = client.chat.completions.create(model=mid, messages=[{'role': 'user', 'content': prompt(r)}],temperature=0, max_tokens=MAX_TOKENS)
    u = resp.usage
    return {'qid': r.name, 'model': key, 'response': resp.choices[0].message.content or '',
            'in_tok': u.prompt_tokens, 'out_tok': u.completion_tokens,
            'cost': getattr(u, 'estimated_cost', None) or (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6,
            'latency': time.time() - t0,
            'truncated': resp.choices[0].finish_reason == 'length'}


def run(client, key, Q):
    # load completed question IDs from this model's results file
    path = ROOT / f'{key}.jsonl'
    # read completed IDs if the results file exists
    if path.exists():
        done = set()
        with open(path) as f:
            for line in f:
                done.add(json.loads(line)['qid'])
    else:
        # start with no completed IDs for a new results file
        done = set()
    # collect questions tha still need a response
    todo = []
    for q in Q.index:
        if q not in done:
            todo.append(q)

    print(f'{key}: {len(done)} done, {len(todo)} to go')
    if not todo:
        return
    # run requests and save results as they finish by opening thread pool and writing to the results file to make faster
    with open(path, 'a') as f, ThreadPoolExecutor(MAX_WORKERS) as ex:
        for n, rec in enumerate(ex.map(lambda q: ask(client, key, Q.loc[q]), todo), 1):
            # grade on the main thread because found math_verify's timeout uses signals
            rec['pred'], rec['truth'], rec['correct'] = grade(Q.loc[rec['qid']], rec['response'])
            f.write(json.dumps(rec) + '\n')
            f.flush()
    print(f'{key}: complete')


def analyze(bits, cost):
    # headroom shares plus oracle accuracy and cost for one subset of questions
    c, e, cc = bits[CHEAP], bits[EXPENSIVE], cost.loc[bits.index]
    # oracle sends query to expensive model only when just the expensive model solves it;
    # queries that neither model can solve go to the cheap model to save cost
    to_expensive = (c == 0) & (e == 1)
    oracle_cost = cc[CHEAP][~to_expensive].sum() + cc[EXPENSIVE][to_expensive].sum()
    return pd.Series({
        'n': len(bits),
        'cheap_acc': c.mean(), 'expensive_acc': e.mean(), 'oracle_acc': (c | e).mean(),
        'both_solve': ((c == 1) & (e == 1)).mean(),
        'only_expensive_solves': ((c == 0) & (e == 1)).mean(),
        'only_cheap_solves': ((c == 1) & (e == 0)).mean(),
        'neither_solves': ((c == 0) & (e == 0)).mean(),
        'oracle_cost_frac': oracle_cost / cc[EXPENSIVE].sum(),
        'oracle_frac_routed_cheap': (~to_expensive).mean(),
    })


def main():
    Q = load_questions()
    client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url='https://api.deepinfra.com/v1/openai')
    for key in [EXPENSIVE, CHEAP]:
        run(client, key, Q)

    # everything below reads the JSONL logs using only questions both models answered
    log = pd.concat([pd.read_json(ROOT / f'{k}.jsonl', lines=True) for k in MODELS], ignore_index=True)
    bits = log.pivot(index='qid', columns='model', values='correct').dropna().astype(int)
    cost = log.pivot(index='qid', columns='model', values='cost').loc[bits.index]
    bits['dataset'] = Q.loc[bits.index, 'dataset'].values

    # Headroom table per dataset and overall
    res = pd.concat([bits.groupby('dataset').apply(lambda b: analyze(b, cost)), analyze(bits, cost).to_frame('overall').T])
    res.to_csv(ROOT / 'pilot_results.csv')
    print(res.round(3).to_string())

    o = res.loc['overall']
    gap = 100 * (o.expensive_acc - o.oracle_acc)
    print('\nTarget is reachable by the oracle.' if gap <= TARGET_GAP and o.oracle_cost_frac <= TARGET_COST
          else '\nTarget NOT reachable by the oracle on this sample. Report routers by distance to the oracle curve.')

    # Training table
    (Q.loc[bits.index, ['dataset', 'subject', 'question']]
     .join(bits.drop(columns='dataset'))
     .join(cost.add_prefix('cost_'))
     .to_csv(ROOT / 'labels.csv'))


if __name__ == '__main__':
    main()
