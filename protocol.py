"""Subject splits, random electrode masks, baseline optimization and scoring."""
from pathlib import Path
import json, random, time
import numpy as np
import torch
import h5py
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from models import make_model, masked_mse
from metrics import NAMES as METRICS, aggregate, per_sample
RATIOS = [0.25, 0.5, 0.75]
RESULTS = Path('results')
BATCHES = {'MLP_DAE': 512, 'ConvDAE': 256, 'UNet1D': 128, 'BiLSTM_AE': 256, 'TCN_AE': 256, 'Transformer': 128, 'GPVAE': 128, 'SAITS': 256, 'TimesNet': 64, 'iTransformer': 256, 'TimeMixer': 256}

def splits_folder(dataset, n_splits):
    return RESULTS / dataset / f'splits_{n_splits}fold'

def batch_for(name, args, evaluation=False):
    batch = args.batch_size or BATCHES.get(name, 256)
    return min(batch, args.eval_batch_size) if evaluation else batch

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)

def mask_plan(n, c, seed, ratio=None):
    rng = np.random.default_rng(seed)
    rank = np.argsort(rng.random((n, c)), axis=1)
    chosen = rng.choice(RATIOS, n) if ratio is None else np.full(n, ratio)
    hidden = np.zeros((n, c), dtype=bool)
    for i, k in enumerate(np.maximum(1, (chosen * c).astype(int))):
        hidden[i, rank[i, :k]] = True
    return hidden

def split_plan(name, subjects, seed, n_splits=5):
    folder = splits_folder(name, n_splits)
    folder.mkdir(parents=True, exist_ok=True)
    for fold, (tv, test) in enumerate(GroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(subjects)), groups=subjects)):
        tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed + fold).split(np.zeros(len(tv)), groups=subjects[tv]))
        train = tv[tr]
        val_all = tv[va]
        rng = np.random.default_rng(seed + fold + 300)
        val = np.concatenate([np.sort(rng.choice(val_all[subjects[val_all] == s], min(64, np.sum(subjects[val_all] == s)), replace=False)) for s in np.unique(subjects[val_all])])
        for a, b in [(train, val_all), (train, test), (val_all, test)]:
            assert not set(subjects[a]) & set(subjects[b])
        path = folder / f'fold_{fold}.npz'
        if path.exists():
            old = np.load(path)
            assert all((np.array_equal(old[k], v) for k, v in [('train', train), ('validation', val), ('validation_all', val_all), ('test', test)])), 'Protocol mismatch: use a new result directory'
        else:
            np.savez(path, train=train, validation=val, validation_all=val_all, test=test)
        yield (fold, train, val, test)

def infer(model, z, idx, hidden, subjects, batch_size, samples=1, save_path=None, seed=0, nmse_only=False):
    all_values = []
    model.eval()
    n, c, t = z.shape
    k = int(hidden[0].sum())
    hidden_indices = np.argsort(~hidden, axis=1)[:, :k]
    hfile = None
    if save_path is not None:
        hfile = h5py.File(save_path, 'w')
        out = hfile.create_dataset('hidden_reconstruction', shape=(len(idx), k, t), dtype='float32', compression='lzf')
        channels = hfile.create_dataset('hidden_channels', data=hidden_indices)
        hfile.create_dataset('sample_indices', data=idx)
    try:
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            for start in range(0, len(idx), batch_size):
                end = min(start + batch_size, len(idx))
                target = z[torch.as_tensor(idx[start:end], device='cuda')]
                h = torch.as_tensor(hidden[start:end], device='cuda')
                m = (~h).float().unsqueeze(-1).expand_as(target)
                observed = target * m
                pred = model(observed, m)
                if nmse_only:
                    difference = pred.double() - target.double()
                    energy = target.double().square().sum(-1).clamp_min(1e-12)
                    channel_nmse = difference.square().sum(-1) / energy
                    values = ((channel_nmse * h).sum(1) / h.sum(1)).cpu().numpy()[:, None]
                else:
                    which = torch.as_tensor(hidden_indices[start:end], device='cuda')
                    rows = torch.arange(len(target), device='cuda')[:, None]
                    p = pred[rows, which].float().cpu().numpy()
                    y = target[rows, which].cpu().numpy()
                    values = per_sample(p, y, np.ones(p.shape[:2], dtype=bool))
                all_values.append(values)
                if hfile is not None:
                    out[start:end] = p
        values = np.concatenate(all_values)
        if hfile is not None:
            hfile.create_dataset('per_window_metrics', data=values)
            hfile.attrs['metric_names'] = json.dumps(METRICS)
            hfile.attrs['evaluation'] = 'hidden channels only; normalized by training subjects statistics'
    finally:
        if hfile is not None:
            hfile.close()
    if nmse_only:
        return ({'NMSE': float(np.mean([values[subjects[idx] == s].mean() for s in np.unique(subjects[idx])]))}, values)
    return (aggregate(values, subjects[idx]), values)

class RidgeGPU(torch.nn.Module):

    def __init__(self, w):
        super().__init__()
        self.register_buffer('weight', w)

    def forward(self, x, m):
        f = torch.cat([x, m, torch.ones_like(x[:, :1])], 1).transpose(1, 2)
        return (f @ self.weight.T).transpose(1, 2)

def ridge_fit(z, train, val, hidden_val, subjects, seed, out):
    c = z.shape[1]
    d = 2 * c + 1
    cov = torch.zeros(d, d, device='cuda', dtype=torch.float64)
    cross = torch.zeros(d, c, device='cuda', dtype=torch.float64)
    count = 0
    hidden = mask_plan(len(train), c, seed)
    for start in range(0, len(train), 64):
        ids = train[start:start + 64]
        target = z[torch.as_tensor(ids, device='cuda')]
        m = torch.as_tensor(~hidden[start:start + 64], device='cuda').float().unsqueeze(-1).expand_as(target)
        features = torch.cat([target * m, m, torch.ones_like(target[:, :1])], 1).transpose(1, 2).reshape(-1, d).double()
        y = target.transpose(1, 2).reshape(-1, c).double()
        cov += features.T @ features
        cross += features.T @ y
        count += len(features)
    cov /= count
    cross /= count
    best = None
    history = []
    for alpha in [0.0001, 0.01, 1.0]:
        weight = torch.zeros(c, d, device='cuda', dtype=torch.float64)
        for channel in range(c):
            keep = torch.tensor([i for i in range(d) if i not in [channel, c + channel]], device='cuda')
            a = cov[keep][:, keep].clone()
            reg = torch.eye(len(keep), device='cuda', dtype=torch.float64) * alpha
            reg[-1, -1] = 0
            weight[channel, keep] = torch.linalg.solve(a + reg, cross[keep, channel])
        model = RidgeGPU(weight.float())
        scores = [infer(model, z, val, h, subjects, 64, nmse_only=True)[0]['NMSE'] for h in hidden_val]
        objective = float(np.mean(scores))
        history.append({'alpha': alpha, 'val_NMSE': objective})
        if best is None or objective < best[0]:
            best = (objective, alpha, model)
    torch.save({'state_dict': best[2].state_dict(), 'alpha': best[1]}, out / 'checkpoint.pt')
    atomic_json(out / 'history.json', history)
    return (best[2], {'alpha': best[1], 'best_val_NMSE': best[0], 'parameters': int(best[2].weight.numel()), 'device': 'cuda:0'})

def train_deep(name, z, train, val, hidden_val, subjects, seed, out, args):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = make_model(name, z.shape[1], z.shape[2]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    batch = batch_for(name, args)
    best = np.inf
    wait = 0
    history = []
    best_epoch = None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        permutation = np.random.default_rng(seed + epoch).permutation(len(train))
        hidden = mask_plan(len(train), z.shape[1], seed + epoch * 1009)
        total = torch.zeros((), device='cuda')
        for start in range(0, len(train), batch):
            rel = permutation[start:start + batch]
            target = z[torch.as_tensor(train[rel], device='cuda')]
            h = torch.as_tensor(hidden[rel], device='cuda')
            m = (~h).float().unsqueeze(-1).expand_as(target)
            x = target * m
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, m, target) if hasattr(model, 'loss') else masked_mse(model(x, m), target, m)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += loss.detach() * len(rel)
        val_scores = [infer(model, z, val, h, subjects, batch_for(name, args, evaluation=True), samples=1, seed=seed + 811, nmse_only=True)[0]['NMSE'] for h in hidden_val]
        objective = float(np.mean(val_scores))
        training_loss = float(total) / len(train)
        elapsed = time.perf_counter() - epoch_start
        history.append({'epoch': epoch + 1, 'training_loss': training_loss, 'val_NMSE': objective, 'val_NMSE_by_ratio': val_scores, 'seconds': elapsed, 'batch_size': batch})
        print(name, 'epoch', epoch + 1, 'batch', batch, 'loss', round(training_loss, 6), 'validation NMSE', round(objective, 6), 'seconds', round(elapsed, 2), flush=True)
        if objective < best - 1e-06:
            best = objective
            wait = 0
            best_epoch = epoch + 1
            torch.save({'state_dict': model.state_dict(), 'epoch': best_epoch, 'name': name, 'channels': z.shape[1], 'times': z.shape[2]}, out / 'checkpoint.pt')
        else:
            wait += 1
        atomic_json(out / 'history.json', history)
        if epoch + 1 >= args.min_epochs and wait >= args.patience:
            break
    state = torch.load(out / 'checkpoint.pt', weights_only=True)
    model.load_state_dict(state['state_dict'])
    config = getattr(model, 'configuration', {'architecture': 'Classical reconstruction baseline implemented in models.py'})
    return (model, {'parameters': sum((p.numel() for p in model.parameters())), 'device': torch.cuda.get_device_name(0), 'best_epoch': best_epoch, 'epochs_run': len(history), 'best_val_NMSE': best, 'batch_size': batch, 'configuration': config, 'architecture': str(model), 'peak_vram_bytes': torch.cuda.max_memory_allocated()})
