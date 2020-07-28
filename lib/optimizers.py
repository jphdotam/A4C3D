import os
import math
import itertools as it

import torch
from torch.optim import Adam, AdamW
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from lib.schedulers import JamesFlatCosineLR


class RangerLars(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=.5, k=5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]

        super().__init__(params, defaults)

        # look ahead params
        for group in self.param_groups:
            group["step_counter"] = 0

        self.alpha = alpha
        self.k = k

        # lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                             for group in self.param_groups]

        # don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):

        loss = None
        # if closure is not None:
        #    loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RangerLars does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                update = torch.zeros_like(p_data_fp32)
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    update.addcdiv_(radam_step_size, exp_avg, denom)
                else:
                    update.add_(radam_step_size, exp_avg)

                if group['weight_decay'] != 0:
                    update.add_(group['weight_decay'], p_data_fp32)

                radam_norm = update.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                trust_ratio = max(0, min(10, trust_ratio))

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                p_data_fp32.add_(-update * trust_ratio * group['lr'])
                p.data.copy_(p_data_fp32)

        # look ahead tracking and updating if latest batch = k
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                # at k interval: take the difference of (RAdam params - LookAhead params) * LookAhead alpha param
                q.data.add_(self.alpha, p.data - q.data)
                # update RAdam weights with the interpolated weights
                p.data.copy_(q.data)

        return loss


def load_optimizer(model, cfg, state, steps_per_epoch=None):
    resuming = cfg['resume'].get('path', False) is not False
    resetting_epoch = cfg['resume'].get('epoch', 0) == 1 and resuming
    resetting_optimizer = cfg['resume'].get('reset_optimizer', False) is not False
    resetting_lr = cfg['resume'].get('reset_lr', False) is not False

    # Create optimizer
    opt = cfg['training']['optimizer']['type']
    lr = cfg['training']['optimizer']['lr']
    wd = cfg['training']['optimizer']['weight_decay']
    if opt == 'adam':
        optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    elif opt == 'adamw':
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    elif opt == 'rangerlars':
        optimizer = RangerLars((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer {opt}")

    # Load optimizer weights if in state dict
    if (opt_path := state.get('optimizer', None)) and not resetting_optimizer:
        optimizer.load_state_dict(opt_path)
        if resetting_lr:
            optimizer.lr = lr
            print(f"Loaded optimizer from state dict and LR reset to {lr}")
        else:
            print(f"Loaded optimizer from state dict")  # ; lr is {optimizer.lr}")

    # SCHEDULERS
    schedtype = cfg['training']['scheduler']['type']

    # Load scheduler if in state dict AND if we're not resetting the epoch or optimizer
    if (scheduler := state.get('scheduler', None)) and not resetting_epoch and not resetting_optimizer:
        print(f"Loaded scheduler from state dict: {scheduler}")
        return optimizer, scheduler

    # Otherwise create scheduler if needed
    elif schedtype:
        if schedtype == 'flatcosine':
            anneal_start = cfg['training'].get('anneal_start_iters', False)
            if anneal_start is False:
                anneal_start = math.floor(cfg['training']['n_epochs'] / 2 * steps_per_epoch)
                scheduler = JamesFlatCosineLR(optimizer,
                                              T_max_from_annealing=math.floor(cfg['training']['n_epochs'] / 2 * steps_per_epoch),
                                              anneal_start_iter=anneal_start)
                print(f"Using flat cosine annealing (annealing from iter {anneal_start} - epoch {anneal_start / steps_per_epoch})")
        if schedtype == 'one_cycle':
            assert steps_per_epoch
            div_factor = cfg['training']['scheduler'].get('one_cycle_div_factor', 25)
            final_div_factor = cfg['training']['scheduler'].get('one_cycle_final_div_factor', 25)
            scheduler = OneCycleLR(optimizer,
                                   max_lr=cfg['training']['lr'],
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=cfg['training']['n_epochs'],
                                   div_factor=div_factor,
                                   final_div_factor=final_div_factor)
            print(f"Using One-cycle scheduler")

        elif schedtype == 'ReduceLROnPlateau':
            patience = cfg['training']['scheduler'].get('patience', 5)
            factor = cfg['training']['scheduler'].get('factor', 0.2)
            scheduler = ReduceLROnPlateau(optimizer,
                                          mode='min',
                                          factor=factor,
                                          patience=patience,
                                          verbose=True)
            print(f"Using ReduceLROnPlateau with factor {factor} and patients {patience}")

            # If we are resuming but not resetting the epoch to 1, user should be warned we aren't continuing the scheduler
        if resuming and not resetting_epoch and not resetting_optimizer:
            print(f"WARNING: Resuming training from a checkpoint without resetting the epochs or optimzier, and yet no"
                  f"scheduler found - creating new scheduler")
    else:
        scheduler = None
    return optimizer, scheduler