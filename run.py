"""Train and test SAGE-EEG and twelve missing-electrode reconstruction baselines."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import protocol
from protocol import RATIOS, mask_plan, split_plan, infer, ridge_fit, train_deep, atomic_json
from models import NAMES, make_model
from metrics import NAMES as METRICS
from committee import build_experts, SAGEEEG
from sage_training import train_sage

BASE = Path(__file__).resolve().parent


def fingerprint(x, subjects):
    digest = hashlib.sha256()
    for start in range(0, len(x), 128):
        digest.update(np.ascontiguousarray(x[start:start + 128]).tobytes())
    digest.update(subjects.tobytes())
    return digest.hexdigest()


def summarize(results):
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in results.glob('*/*/fold_*/mask_*/metrics.json')]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(results / 'per_fold_metrics.csv', index=False)
    summary = df.groupby(['dataset', 'model', 'missing_ratio'])[METRICS].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(results / 'summary_mean_std.csv')


def load_model(name, out):
    state = torch.load(out / 'checkpoint.pt', weights_only=True, map_location='cuda')
    if name == 'SAGE-EEG':
        model = SAGEEEG(build_experts(state['absolute'], state['gain'], state['library']), state['weights'], state['channels'])
    elif name == 'Ridge':
        model = protocol.RidgeGPU(state['state_dict']['weight'])
    else:
        model = make_model(name, state['channels'], state['times']).cuda()
    model.load_state_dict(state['state_dict'])
    return model.cuda().eval()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'test', 'summarize'])
    p.add_argument('--data-dir', type=Path, default=BASE / 'data')
    p.add_argument('--output-dir', type=Path, default=BASE / 'results')
    p.add_argument('--datasets', nargs='+', default=['PhysioNet', 'EEG-MCD', 'DEAP'])
    p.add_argument('--models', nargs='+', choices=['SAGE-EEG'] + NAMES, default=['SAGE-EEG'] + NAMES)
    p.add_argument('--n-splits', type=int, default=3)
    p.add_argument('--original-folds', action='store_true', help='Five folds for Ridge/MLP_DAE/ConvDAE; three for the others')
    p.add_argument('--folds', nargs='+', type=int)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--min-epochs', type=int, default=15)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=0, help='0 uses the RTX 5070 Ti per-model defaults')
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001, help='Baseline learning rate; SAGE calibrators use .003')
    p.add_argument('--seed', type=int, default=20261009)
    args = p.parse_args()
    if args.epochs < 1 or args.n_splits < 2 or args.batch_size < 0 or args.eval_batch_size < 1:
        p.error('Invalid epoch, fold or batch size')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.RESULTS = args.output_dir
    if args.mode == 'summarize':
        summarize(args.output_dir)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. Install a CUDA-enabled PyTorch build for your GPU.')
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('GPU:', torch.cuda.get_device_name(0), torch.__version__, flush=True)
    for dataset in args.datasets:
        x = np.load(args.data_dir / f'{dataset}_X.npy', mmap_mode='r')
        subjects = np.load(args.data_dir / f'{dataset}_metadata.npz', allow_pickle=False)['subjects']
        if x.ndim != 3 or x.shape[-1] != 256 or len(x) != len(subjects) or x.shape[1] < 4:
            raise ValueError('Expected X[N,C,256], C>=4, and subjects[N] at 128 Hz')
        if not np.isfinite(x).all():
            raise ValueError('Dataset contains nonfinite samples')
        digest = fingerprint(x, subjects)
        raw = torch.tensor(np.array(x), dtype=torch.float32, device='cuda')
        for name in args.models:
            splits = (5 if name in ['Ridge', 'MLP_DAE', 'ConvDAE'] else 3) if args.original_folds else args.n_splits
            if args.folds is not None and any(f < 0 or f >= splits for f in args.folds):
                raise ValueError(f'{name}: --folds must be between 0 and {splits - 1}')
            for fold, train, val, test in split_plan(dataset, subjects, args.seed, splits):
                if args.folds is not None and fold not in args.folds:
                    continue
                out = args.output_dir / dataset / name / f'fold_{fold}'
                seed = args.seed + fold * 100000
                hidden_val = [mask_plan(len(val), raw.shape[1], seed + 421, r) for r in RATIOS]
                if args.mode == 'train':
                    if (out / 'checkpoint.pt').exists():
                        raise FileExistsError(f'{out}: checkpoint exists; use test mode or a fresh output directory')
                    out.mkdir(parents=True, exist_ok=True)
                    ids = torch.as_tensor(train, device='cuda')
                    mean = raw[ids].mean((0, 2), keepdim=True)
                    std = raw[ids].std((0, 2), unbiased=False, keepdim=True).clamp_min(1e-8)
                    np.savez(out / 'normalization.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy())
                    np.savez(out / 'split.npz', train=train, validation=val, test=test)
                    z = (raw - mean) / std
                    if name == 'SAGE-EEG':
                        model = train_sage(z, train, val, subjects, seed, out, args)
                    elif name == 'Ridge':
                        model, _ = ridge_fit(z, train, val, hidden_val, subjects, seed, out)
                    else:
                        model, _ = train_deep(name, z, train, val, hidden_val, subjects, seed, out, args)
                    atomic_json(out / 'run.json', {'dataset': dataset, 'model': name, 'dataset_sha256': digest,
                                'n_splits': splits, 'seed': args.seed, 'fold': fold, 'torch': torch.__version__,
                                'gpu': torch.cuda.get_device_name(0), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
                else:
                    info = json.loads((out / 'run.json').read_text(encoding='utf-8'))
                    if info['dataset_sha256'] != digest or info['n_splits'] != splits or info['seed'] != args.seed:
                        raise ValueError('Dataset or split protocol differs from checkpoint training')
                    saved_split = np.load(out / 'split.npz')
                    for key, indices in [('train', train), ('validation', val), ('test', test)]:
                        np.testing.assert_array_equal(saved_split[key], indices)
                    norm = np.load(out / 'normalization.npz')
                    z = (raw - torch.tensor(norm['mean'], device='cuda')) / torch.tensor(norm['std'], device='cuda')
                    model = load_model(name, out)
                model.eval()
                for ratio in RATIOS:
                    target = out / f'mask_{int(ratio * 100)}'
                    target.mkdir(exist_ok=True)
                    hidden = mask_plan(len(test), raw.shape[1], seed + 642, ratio)
                    score, values = infer(model, z, test, hidden, subjects, args.eval_batch_size, seed=seed + 913)
                    np.savez_compressed(target / 'per_window.npz', metrics=values, hidden=hidden,
                                        sample_indices=test, subjects=subjects[test], metric_names=np.array(METRICS))
                    score.update(dataset=dataset, model=name, fold=fold, n_splits=splits, missing_ratio=ratio)
                    atomic_json(target / 'metrics.json', score)
                    print(dataset, name, fold, ratio, score['NMSE'], flush=True)
                del model, z
                gc.collect()
                torch.cuda.empty_cache()
        del raw
        gc.collect()
        torch.cuda.empty_cache()
    summarize(args.output_dir)


if __name__ == '__main__':
    main()
