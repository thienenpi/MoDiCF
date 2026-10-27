"""Item-graph neighbor retrieval for graph-retrieval-enhanced modality completion.

This module implements the *Modality-Aware Subgraph Retrieval* idea from
"Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion"
(GRE-MC, docs/2605.00670v1.pdf), adapted as an anchor-retrieval step that feeds
the diffusion completer in MoDiCF.

For every item and every modality ``m`` that may need to be reconstructed, we
retrieve a fixed set of ``K`` *anchor* items that

  1. actually observe modality ``m`` (so they carry genuine signal about it), and
  2. are semantically similar to the query item in the modalities the query
     itself observes, optionally boosted by item-item co-interaction structure
     derived from the user-item graph (``train_mat``).

The returned neighbor index/weights are consumed by ``GraphMSDiffusion``.
"""

import os
import numpy as np
import scipy.sparse as sp
import torch


def _l2norm_rows(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def build_item_cooccurrence(ui_graph):
    """Build a normalized item-item co-interaction matrix from a user-item csr matrix.

    Two items are connected if they share at least one interacting user (GRE-MC Eq. 1).
    The raw co-interaction counts are symmetrically degree-normalized to [0, 1]-ish
    weights so the graph signal is comparable to cosine similarity.
    """
    R = ui_graph.tocsr().astype(np.float32)
    # binarize interactions
    R.data = np.ones_like(R.data)
    Aii = (R.T @ R).tocsr()          # [n_items, n_items], counts of shared users
    Aii.setdiag(0.0)
    Aii.eliminate_zeros()

    deg = np.asarray(Aii.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg + 1e-8, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    Aii_norm = (D @ Aii @ D).tocsr()
    return Aii_norm


def _neighbor_cache_path(path, dataset, MR, seed, K, beta):
    fname = f"retrieval_neighbors_seed{seed}_MR{MR}_K{K}_beta{beta}.npz"
    return os.path.join(path, fname)


def build_neighbors(dataset, data_path, incomplete_data, indicator, modalities,
                    ui_graph, K=10, beta=0.3, chunk=2048, seed=0, MR=0.4,
                    use_cache=True, device=torch.device("cpu")):
    """Precompute per-modality top-K anchor neighbors for every item.

    Args:
        incomplete_data: list of [n_items, d_m] feature arrays/tensors that are
            actually observable (missing entries already imputed by the chosen
            ``--complete`` strategy). Ground-truth ``ori_data`` is NOT used.
        indicator: [n_items, n_modalities] int array; 1 == modality observed.
        ui_graph: scipy csr user-item matrix (train_mat).
        K: number of anchors per item.
        beta: weight of the graph co-interaction signal added to cosine similarity.

    Returns:
        nbr_idx:  list (len n_modalities) of LongTensor [n_items, K] global item ids.
        nbr_sim:  list (len n_modalities) of FloatTensor [n_items, K] raw scores
                  (softmaxed inside the model). Padding slots carry -inf.
    """
    n_modalities = len(modalities)
    n_items = indicator.shape[0]

    cache_file = _neighbor_cache_path(data_path, dataset, MR, seed, K, beta)
    if use_cache and os.path.exists(cache_file):
        blob = np.load(cache_file)
        nbr_idx = [torch.from_numpy(blob[f"idx_{m}"]).long().to(device) for m in range(n_modalities)]
        nbr_sim = [torch.from_numpy(blob[f"sim_{m}"]).float().to(device) for m in range(n_modalities)]
        print(f"[retrieval] Loaded cached neighbor index from {cache_file}")
        return nbr_idx, nbr_sim

    print(f"[retrieval] Building neighbor index (K={K}, beta={beta}) ...")

    ind = np.asarray(indicator).astype(bool)               # [N, M]
    # per-modality L2-normalized features (rows for unobserved modalities are zeroed
    # so they contribute no spurious similarity).
    feats = []
    for m in range(n_modalities):
        f = incomplete_data[m]
        if torch.is_tensor(f):
            f = f.detach().float().cpu()
        else:
            f = torch.tensor(np.asarray(f), dtype=torch.float32)
        f = _l2norm_rows(f)
        f[~torch.from_numpy(ind[:, m])] = 0.0
        feats.append(f.to(device))

    Aii = build_item_cooccurrence(ui_graph) if beta > 0 else None

    nbr_idx, nbr_sim = [], []
    NEG = float("-inf")

    for m in range(n_modalities):
        pool = np.where(ind[:, m])[0]                      # candidates observing modality m
        pool_t = torch.from_numpy(pool).long().to(device)
        # candidate features for each ranking modality r (only where candidate observes r)
        cand_feats = {r: feats[r][pool_t] for r in range(n_modalities) if r != m}

        idx_out = np.full((n_items, K), -1, dtype=np.int64)
        sim_out = np.full((n_items, K), NEG, dtype=np.float32)

        for start in range(0, n_items, chunk):
            end = min(start + chunk, n_items)
            q = torch.arange(start, end, device=device)
            b = end - start

            score = torch.zeros(b, pool.shape[0], device=device)
            count = torch.zeros(b, pool.shape[0], device=device)
            for r in range(n_modalities):
                if r == m:
                    continue
                s = feats[r][q] @ cand_feats[r].T          # [b, |pool|] cosine sim
                # valid only where both query and candidate observe modality r
                qobs = torch.from_numpy(ind[start:end, r]).float().to(device).unsqueeze(1)
                cobs = torch.from_numpy(ind[pool, r]).float().to(device).unsqueeze(0)
                mask = qobs * cobs
                score += s * mask
                count += mask
            sem = score / count.clamp(min=1.0)             # mean cosine over shared modalities

            if Aii is not None:
                g = torch.from_numpy(Aii[start:end][:, pool].toarray()).float().to(device)
                total = sem + beta * g
            else:
                total = sem

            # exclude self if the query itself is in the pool
            self_pos = (pool_t.unsqueeze(0) == q.unsqueeze(1))
            total = total.masked_fill(self_pos, NEG)
            # queries with no shared-modality evidence and no graph link -> leave as NEG
            no_evidence = (count.sum(dim=1) == 0)
            if Aii is None:
                total[no_evidence] = NEG

            kk = min(K, total.shape[1])
            topv, topi = torch.topk(total, kk, dim=1)
            gids = pool_t[topi]                            # map pool position -> global id
            idx_out[start:end, :kk] = gids.cpu().numpy()
            sim_out[start:end, :kk] = topv.cpu().numpy()

        # Fallback: any row with no valid anchor falls back to itself (weight 0 -> self
        # modality features; harmless, equals a degenerate self-based condition).
        no_valid = ~np.isfinite(sim_out).any(axis=1)
        if no_valid.any():
            rows = np.where(no_valid)[0]
            idx_out[rows, 0] = rows
            sim_out[rows, 0] = 0.0
        # remaining -inf padding slots get a placeholder index (self) so gather is safe;
        # their -inf weight zeroes them out in the model's softmax.
        pad = idx_out < 0
        idx_out[pad] = np.repeat(np.arange(n_items)[:, None], K, axis=1)[pad]

        nbr_idx.append(torch.from_numpy(idx_out).long().to(device))
        nbr_sim.append(torch.from_numpy(sim_out).float().to(device))
        print(f"[retrieval]   modality '{modalities[m]}': pool={pool.shape[0]} items, "
              f"{int(no_valid.sum())} fallback-to-self")

    if use_cache:
        np.savez(cache_file,
                 **{f"idx_{m}": nbr_idx[m].cpu().numpy() for m in range(n_modalities)},
                 **{f"sim_{m}": nbr_sim[m].cpu().numpy() for m in range(n_modalities)})
        print(f"[retrieval] Cached neighbor index to {cache_file}")

    return nbr_idx, nbr_sim
