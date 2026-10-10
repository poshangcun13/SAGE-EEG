"""Fit the final SAGE-EEG fields, calibrators, library and convex mixture."""
import random
import numpy as np
import torch
from field import AbsoluteField, estimate_fields, relative_loss
from gain_field import GainTransport, fit_gain_fields, exact_relative_loss
from library import fit_library
from committee import build_experts, SAGEEEG
from calibration import quadratic, solve
from protocol import RATIOS, mask_plan, infer, atomic_json


def fit_component(model, loss_fn, z, train, val, subjects, seed, out, args, batch, initial=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    masks = [mask_plan(len(val), z.shape[1], seed + 421, r) for r in RATIOS]
    best, wait, history, state = float('inf'), 0, [], None
    for epoch in range(0 if initial else 1, args.epochs + 1):
        if epoch:
            model.train()
            order = np.random.default_rng(seed + epoch - 1).permutation(len(train))
            hidden = mask_plan(len(train), z.shape[1], seed + (epoch - 1) * 1009)
            for start in range(0, len(train), batch):
                rel = order[start:start + batch]
                y = z[torch.as_tensor(train[rel], device=z.device)]
                mask = torch.as_tensor(~hidden[rel], device=z.device).float().unsqueeze(-1).expand_as(y)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(y * mask, mask), y, mask)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite SAGE-EEG training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
        scores = [infer(model, z, val, h, subjects, min(batch, args.eval_batch_size), nmse_only=True)[0]['NMSE'] for h in masks]
        objective = float(np.mean(scores))
        history.append({'epoch': epoch, 'validation_NMSE': objective, 'by_ratio': scores})
        print(out.name, epoch, objective, flush=True)
        if objective < best - 1e-6:
            best, wait = objective, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        atomic_json(out, history)
        if epoch >= args.min_epochs and wait >= args.patience:
            break
    model.load_state_dict(state)
    return state


def train_sage(z, train, val, subjects, seed, out, args):
    def seed_all():
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    seed_all()
    covariance, bands = estimate_fields(z, train, 'robust', True)
    absolute_model = AbsoluteField(covariance, bands).cuda()
    absolute = {'covariance': covariance, 'bands': bands,
                'state_dict': fit_component(absolute_model, relative_loss, z, train, val, subjects, seed,
                                            out / 'absolute_history.json', args, args.batch_size or 128)}
    del absolute_model
    seed_all()
    priors = fit_gain_fields(z, train)
    gain_model = GainTransport(priors).cuda()
    gain = {'priors': priors,
            'state_dict': fit_component(gain_model, exact_relative_loss, z, train, val, subjects, seed,
                                       out / 'gain_history.json', args, args.batch_size or 256, initial=True)}
    del gain_model
    covariance, bands = fit_library(z, train, subjects)
    library = {'covariance': covariance, 'bands': bands}
    experts = [model.cuda().eval() for model in build_experts(absolute, gain, library)]
    weights, records = [], []
    for ratio in RATIOS:
        hidden = mask_plan(len(val), z.shape[1], seed + 421, ratio)
        a, b, c = quadratic(experts, z, val, hidden, subjects)
        w, risk = solve(a, b, c)
        weights.append(w)
        records.append({'missing_ratio': ratio, 'weights': w.tolist(), 'validation_NMSE': risk})
    weights = torch.tensor(np.array(weights), device=z.device)
    model = SAGEEEG(experts, weights, z.shape[1]).cuda().eval()
    atomic_json(out / 'calibration.json', records)
    torch.save({'absolute': absolute, 'gain': gain, 'library': library,
                'weights': weights, 'channels': z.shape[1], 'state_dict': model.state_dict()}, out / 'checkpoint.pt')
    return model
