"""
Frequency-domain contrastive learning utilities.

Implements:
  - RegionContrastMemory: per-class deque-based memory bank.
  - compute_region_centers: compute per-class mean feature vectors.
  - contrastive_loss: InfoNCE-style loss using region centers and memory banks.

Design is inspired by the multi-scale region contrastive memory in AMR-CMB but
simplified to 2D single-scale usage.
"""

import collections
import torch
import torch.nn.functional as F


class RegionContrastMemory:
    """Per-class memory bank storing region-center feature vectors.

    Each class maintains a fixed-size FIFO queue (``collections.deque``).
    Entries are L2-normalised feature vectors of dimension ``feat_dim``.

    Args:
        num_classes: Number of segmentation classes (0 … num_classes-1).
        feat_dim:    Feature dimensionality (number of channels).
        queue_size:  Maximum number of entries per class queue.
    """

    def __init__(self, num_classes: int, feat_dim: int, queue_size: int = 200):
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.queue_size  = queue_size
        # One deque per class; each element is a 1-D CPU tensor [feat_dim]
        self.queues = [collections.deque(maxlen=queue_size)
                       for _ in range(num_classes)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_centers(self, centers_dict: dict,
                            confidence_scores: dict = None,
                            conf_threshold: float = 0.9):
        """Push new region-center vectors into the per-class queues.

        Args:
            centers_dict:      Dict mapping class_id -> Tensor [M, C] where M
                               is the number of new center vectors for that class.
            confidence_scores: Optional dict mapping class_id -> Tensor [M] of
                               per-center confidence values (0–1).  When provided,
                               only centers whose confidence ≥ ``conf_threshold``
                               are enqueued.
            conf_threshold:    Minimum confidence to accept a center (default 0.9).
        """
        for cls_id, centers in centers_dict.items():
            if centers is None or centers.numel() == 0:
                continue
            # Optionally filter by confidence
            if confidence_scores is not None and cls_id in confidence_scores:
                conf = confidence_scores[cls_id]
                keep = conf >= conf_threshold
                if keep.sum() == 0:
                    continue
                centers = centers[keep]
            # L2-normalise and move to CPU for storage
            centers = F.normalize(centers, dim=1).detach().cpu()
            for vec in centers:
                self.queues[cls_id].append(vec)

    def get_memory(self, cls_id: int):
        """Return all stored feature vectors for ``cls_id`` as a Tensor.

        Returns:
            Tensor of shape ``[N, C]`` where N is the current queue length,
            or ``None`` if the queue is empty.
        """
        q = self.queues[cls_id]
        if len(q) == 0:
            return None
        return torch.stack(list(q), dim=0)  # [N, C]

    def size(self, cls_id: int) -> int:
        """Current number of entries stored for ``cls_id``."""
        return len(self.queues[cls_id])


# ---------------------------------------------------------------------------
# Region-center computation
# ---------------------------------------------------------------------------

def compute_region_centers(feature_map: torch.Tensor,
                           labels: torch.Tensor,
                           num_classes: int):
    """Compute per-class mean feature vectors (region centers) over a batch.

    For every sample in the batch, and for every class present in the label
    map, we average the feature vectors of all spatial positions that belong
    to that class.  The resulting centers are aggregated across the batch.

    Args:
        feature_map: Float tensor of shape ``[B, C, h, w]``.
        labels:      Long tensor of shape  ``[B, H, W]``.  If the spatial
                     resolution differs from ``feature_map``, labels are
                     downsampled with nearest-neighbor interpolation.
        num_classes: Total number of classes.

    Returns:
        Dict mapping class_id (int) -> Tensor ``[M, C]`` or ``None``.
        M is the number of samples in the batch that contain that class.
    """
    B, C, h, w = feature_map.shape

    # Resize labels to match feature map resolution (nearest-neighbor)
    if labels.shape[-2:] != (h, w):
        labels_ds = F.interpolate(
            labels.float().unsqueeze(1), size=(h, w),
            mode='nearest').squeeze(1).long()
    else:
        labels_ds = labels

    centers = {cls_id: [] for cls_id in range(num_classes)}

    for b in range(B):
        feat = feature_map[b]          # [C, h, w]
        lbl  = labels_ds[b]            # [h, w]
        for cls_id in range(num_classes):
            mask = (lbl == cls_id)     # [h, w] bool
            if mask.sum() == 0:
                continue
            # Mean over masked spatial positions -> [C]
            center = feat[:, mask].mean(dim=1)
            centers[cls_id].append(center)

    # Stack per-class lists into tensors
    result = {}
    for cls_id in range(num_classes):
        vecs = centers[cls_id]
        if len(vecs) == 0:
            result[cls_id] = None
        else:
            result[cls_id] = torch.stack(vecs, dim=0)  # [M, C]

    return result


# ---------------------------------------------------------------------------
# InfoNCE-style contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(anchor_centers: dict,
                     memory: RegionContrastMemory,
                     tau: float = 0.15,
                     max_neg: int = 128):
    """Compute InfoNCE contrastive loss between anchor centers and memory bank.

    For each class ``c``:
      - Anchors   : current batch region centers for class ``c``.
      - Positives : stored memory vectors for class ``c``.
      - Negatives : stored memory vectors for all other classes (sampled to
                    at most ``max_neg`` entries).

    The loss for one anchor ``a`` is:
        -log( sum_pos exp(cos(a,p)/τ) / (sum_pos exp(cos(a,p)/τ)
                                          + sum_neg exp(cos(a,n)/τ)) )

    Args:
        anchor_centers: Dict mapping class_id -> Tensor ``[M, C]``.
        memory:         ``RegionContrastMemory`` instance.
        tau:            Temperature (default 0.15).
        max_neg:        Maximum number of negative samples to use per loss
                        computation (default 128).

    Returns:
        Scalar loss tensor (returns 0 when no valid pair exists).
    """
    device = None
    # Detect device from anchors
    for v in anchor_centers.values():
        if v is not None:
            device = v.device
            break
    if device is None:
        return torch.tensor(0.0, requires_grad=False)

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for cls_id in range(memory.num_classes):
        anchors = anchor_centers.get(cls_id, None)
        if anchors is None or anchors.numel() == 0:
            continue
        pos_mem = memory.get_memory(cls_id)
        if pos_mem is None:
            continue  # No positives yet – skip this class

        # Gather negatives from all other classes
        neg_list = []
        for neg_cls in range(memory.num_classes):
            if neg_cls == cls_id:
                continue
            neg_mem = memory.get_memory(neg_cls)
            if neg_mem is not None:
                neg_list.append(neg_mem)
        if len(neg_list) == 0:
            continue  # No negatives yet – skip
        neg_mem_all = torch.cat(neg_list, dim=0)  # [N_neg, C]

        # Subsample negatives if too many
        if neg_mem_all.shape[0] > max_neg:
            idx = torch.randperm(neg_mem_all.shape[0])[:max_neg]
            neg_mem_all = neg_mem_all[idx]

        # Move memory to anchor device
        pos_mem     = pos_mem.to(device)
        neg_mem_all = neg_mem_all.to(device)

        # L2-normalise anchors (memory is already normalised)
        anchors_n = F.normalize(anchors, dim=1)  # [M, C]
        pos_n     = F.normalize(pos_mem, dim=1)  # [P, C]
        neg_n     = F.normalize(neg_mem_all, dim=1)  # [N, C]

        # Similarity: [M, P] and [M, N]
        sim_pos = torch.mm(anchors_n, pos_n.T) / tau   # [M, P]
        sim_neg = torch.mm(anchors_n, neg_n.T) / tau   # [M, N]

        # Numerator: log-sum-exp over positives (per anchor)
        log_num = torch.logsumexp(sim_pos, dim=1)       # [M]

        # Denominator: log-sum-exp over both positives and negatives
        sim_all = torch.cat([sim_pos, sim_neg], dim=1)  # [M, P+N]
        log_den = torch.logsumexp(sim_all, dim=1)        # [M]

        loss_cls = (log_den - log_num).mean()
        total_loss = total_loss + loss_cls
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)

    return total_loss / count
