"""Graph-retrieval-enhanced diffusion completion.

``GraphMSDiffusion`` subclasses the author's ``MSDiffusion`` and changes ONLY how
the diffusion denoiser is conditioned. The original self-based conditioning uses
an item's own *other* modalities. Here we additionally retrieve, for each item
and target modality, a set of semantically-relevant anchor items from the item
graph (see ``graph_retrieval.build_neighbors``) and fuse their target-modality
information into the conditioning signal.

Because MoDiCF's U-Net already cross-attends to the condition tensor, we keep the
condition shape identical to the original ([B, embed_channel, embed_size]) and
simply produce a *graph-aware* condition instead of a purely self-based one. This
makes the change fully drop-in: the diffusion/U-Net stack is untouched.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mDiffusion import MSDiffusion


class CondFusion(nn.Module):
    """Gated fusion of the self-based condition and the neighbor-based condition.

    Both inputs are [B, C, E]. A learned gate (per channel/position) interpolates
    between them, letting the model fall back to self-conditioning when retrieved
    neighbors are unhelpful and lean on neighbors when they carry strong signal.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, self_cond, nbr_cond):
        g = self.gate(torch.cat([self_cond, nbr_cond], dim=1))
        return g * self_cond + (1.0 - g) * nbr_cond


class GraphMSDiffusion(MSDiffusion):
    def __init__(self, *args, retrieval_k=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_k = retrieval_k
        # one fusion head per modality
        for m in range(self.n_modalities):
            self.models['fusion_' + str(m)] = CondFusion(self.embed_channel)

        # populated via set_neighbors / set_full_features
        self.nbr_idx = None     # list[m] LongTensor [n_items, K]
        self.nbr_sim = None     # list[m] FloatTensor [n_items, K] (raw scores, -inf pad)
        self._full_feats = None  # list[m] Tensor [n_items, d_m] used to encode anchors

    # ------------------------------------------------------------------ setup
    def set_neighbors(self, nbr_idx, nbr_sim):
        self.nbr_idx = nbr_idx
        self.nbr_sim = nbr_sim

    def set_full_features(self, data):
        """Reference to the full per-modality feature matrices used to encode anchors.

        In MoDiCF's joint training the completed features are refreshed every epoch,
        so this should be called whenever the feature tensors are replaced.
        """
        self._full_feats = data

    def _ready(self):
        return self.nbr_idx is not None and self._full_feats is not None

    # -------------------------------------------------------------- conditions
    def _neighbor_condition(self, m, gids):
        """Encode and aggregate the target-modality features of item ``gids`` anchors.

        gids: LongTensor [n] of global item ids whose modality-``m`` condition we build.
        Returns [n, embed_channel, embed_size].
        """
        idx = self.nbr_idx[m][gids]            # [n, K]
        sim = self.nbr_sim[m][gids]            # [n, K]
        n, K = idx.shape

        feats = self._full_feats[m][idx.reshape(-1)]          # [n*K, d_m]
        enc = self.models[self.modalities[m] + '_enc'](feats)  # [n*K, C, E]
        C, E = enc.shape[1], enc.shape[2]
        enc = enc.reshape(n, K, C, E)

        w = torch.softmax(sim, dim=1)          # [n, K]; -inf pads -> 0
        # rows that are entirely -inf (shouldn't happen after fallback) -> uniform
        bad = torch.isnan(w).any(dim=1)
        if bad.any():
            w[bad] = 1.0 / K
        w = w.view(n, K, 1, 1)
        return (enc * w).sum(dim=1)            # [n, C, E]

    def _fused_condition(self, m, embeds, ids, gids):
        """Build the graph-aware condition for modality ``m`` over rows ``ids``.

        embeds: list of [N, C, E] encoded modality embeddings for the current data.
        ids:    LongTensor/ndarray row indices (into the current batch) being conditioned.
        gids:   global item ids corresponding to ``ids`` (for neighbor lookup).
        """
        other = [embeds[j][ids, :, :] for j in range(self.n_modalities) if j != m]
        other = torch.cat(other, dim=1)
        self_cond = self.models['condition_' + str(m)](other)

        if not self._ready():
            return self_cond
        nbr_cond = self._neighbor_condition(m, gids)
        return self.models['fusion_' + str(m)](self_cond, nbr_cond)

    def _global_ids(self, data, item_idx):
        n = data[0].shape[0]
        if item_idx is None:
            return torch.arange(n, device=data[0].device)
        if not torch.is_tensor(item_idx):
            item_idx = torch.as_tensor(np.asarray(item_idx), device=data[0].device)
        return item_idx.to(data[0].device).long()

    # -------------------------------------------------------------- overrides
    def forward(self, data, indicator, print_progress=False, min_data=None, max_data=None,
                item_idx=None):
        dtype = data[0].dtype
        device = data[0].device
        gids_all = self._global_ids(data, item_idx)

        embeds = [self.models[self.modalities[m] + '_enc'](data[m]) for m in range(self.n_modalities)]

        diff_loss = torch.tensor(0.0, dtype=dtype, device=device)
        rec_loss = torch.tensor(0.0, dtype=dtype, device=device)
        for m in range(self.n_modalities):
            m_complete_id = np.where(indicator[:, m] == 1)[0]
            if len(m_complete_id) == 0:
                continue
            ids = torch.from_numpy(m_complete_id).long().to(device)
            m_condition = self._fused_condition(m, embeds, ids, gids_all[ids])

            m_embeds = embeds[m][ids, :, :]
            diff_loss += self.models[self.modalities[m] + '_diffusion'](m_embeds, condition=m_condition)

            rec_dec = self.models[self.modalities[m] + '_dec'](m_embeds)
            rec_loss += F.mse_loss(rec_dec, data[m][m_complete_id, :])

        return diff_loss, rec_loss

    def complete(self, data, indicator, target_data, min_data=None, max_data=None, item_idx=None):
        dtype = data[0].dtype
        device = data[0].device
        gids_all = self._global_ids(data, item_idx)

        batch_ind_sum = np.sum(indicator, axis=1)
        incomplete_id = np.where(batch_ind_sum < self.n_modalities)[0]
        if len(incomplete_id) == 0:
            return data, torch.tensor(0., dtype=dtype, device=device)

        embeds = [self.models[self.modalities[m] + '_enc'](data[m]) for m in range(self.n_modalities)]

        rec_data = [data[m].detach().cpu().numpy() for m in range(self.n_modalities)]
        rec_loss = torch.tensor(0.0, dtype=dtype, device=device)
        for m in range(self.n_modalities):
            m_incomplete_id = np.where(indicator[:, m] == 0)[0]
            if len(m_incomplete_id) == 0:
                continue
            ids = torch.from_numpy(m_incomplete_id).long().to(device)
            m_condition = self._fused_condition(m, embeds, ids, gids_all[ids])
            m_embeds = embeds[m][ids, :, :]

            rec_diff = self.models[self.modalities[m] + '_diffusion'].sample(
                shape=m_embeds.shape, condition=m_condition, print_progress=False,
                min_data=min_data[m], max_data=max_data[m])
            rec_dec = self.models[self.modalities[m] + '_dec'](rec_diff)
            rec_data[m][m_incomplete_id, :] = rec_dec.detach().cpu().numpy()
            rec_loss += F.mse_loss(rec_dec, target_data[m][m_incomplete_id, :])

        return rec_data, rec_loss.item()

    def complete_train(self, data, indicator, target_data, min_data=None, max_data=None, item_idx=None):
        dtype = data[0].dtype
        device = data[0].device
        gids_all = self._global_ids(data, item_idx)

        embeds = [self.models[self.modalities[m] + '_enc'](data[m]) for m in range(self.n_modalities)]

        out_data = [data[m].clone().detach() for m in range(self.n_modalities)]
        rec_train_loss = torch.tensor(0.0, dtype=dtype, device=device)
        rec_eval_loss = torch.tensor(0.0, dtype=dtype, device=device)
        n = data[0].shape[0]
        all_ids = torch.arange(n, device=device)
        for m in range(self.n_modalities):
            m_condition = self._fused_condition(m, embeds, all_ids, gids_all)
            m_embeds = embeds[m]

            rec_diff = self.models[self.modalities[m] + '_diffusion'].sample(
                shape=m_embeds.shape, condition=m_condition, print_progress=False,
                min_data=min_data[m], max_data=max_data[m])
            rec_dec = self.models[self.modalities[m] + '_dec'](rec_diff)

            m_complete_id = np.where(indicator[:, m] == 1)[0]
            if len(m_complete_id) != 0:
                rec_train_loss += F.mse_loss(rec_dec[m_complete_id, :], target_data[m][m_complete_id, :])

            with torch.no_grad():
                m_incomplete_id = np.where(indicator[:, m] == 0)[0]
                if len(m_incomplete_id) != 0:
                    rec_eval_loss += F.mse_loss(rec_dec[m_incomplete_id, :], target_data[m][m_incomplete_id, :])
                    out_data[m][m_incomplete_id, :] = rec_dec[m_incomplete_id, :].clone().detach()

        return out_data, rec_train_loss, rec_eval_loss
