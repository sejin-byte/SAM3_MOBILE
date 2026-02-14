"""MPS-native Greedy Matcher — replaces scipy Hungarian assignment.

Computes cost matrix from mask IoU, box L1, and logit similarity entirely
on-device using torch tensor operations. No CPU roundtrip required.
"""

import torch
import torch.nn.functional as F


class GreedyMatcher:
    """Greedy bipartite matching between student and teacher predictions.

    Cost = alpha * (1 - mask_iou) + beta * box_l1 + gamma * (1 - logit_sim)

    All operations stay on the input tensors' device (MPS/CUDA/CPU).
    """

    def __init__(
        self,
        mask_iou_weight: float = 1.0,
        box_l1_weight: float = 1.0,
        logit_sim_weight: float = 1.0,
    ):
        self.mask_iou_weight = mask_iou_weight
        self.box_l1_weight = box_l1_weight
        self.logit_sim_weight = logit_sim_weight

    @staticmethod
    def compute_mask_iou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Dice/IoU between two sets of masks.

        Args:
            masks_a: [Na, H, W] predicted masks (logits or probabilities)
            masks_b: [Nb, H, W] target masks

        Returns:
            [Na, Nb] IoU matrix
        """
        a = masks_a.sigmoid().flatten(1)  # [Na, HW]
        b = masks_b.sigmoid().flatten(1)  # [Nb, HW]

        # Dice coefficient as IoU proxy
        intersection = torch.mm(a, b.t())  # [Na, Nb]
        sum_a = a.sum(dim=1, keepdim=True)  # [Na, 1]
        sum_b = b.sum(dim=1, keepdim=True)  # [1, Nb]
        union = sum_a + sum_b.t() - intersection
        iou = intersection / (union + 1e-6)
        return iou

    @staticmethod
    def compute_box_l1(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
        """Pairwise L1 distance between box sets.

        Args:
            boxes_a: [Na, 4]
            boxes_b: [Nb, 4]

        Returns:
            [Na, Nb] L1 distance matrix (normalized by 4)
        """
        # Expand for broadcasting
        diff = (boxes_a.unsqueeze(1) - boxes_b.unsqueeze(0)).abs()  # [Na, Nb, 4]
        return diff.sum(dim=-1) / 4.0  # normalize by coordinate count

    @staticmethod
    def compute_logit_similarity(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        """Pairwise sigmoid similarity between logit sets.

        Args:
            logits_a: [Na]
            logits_b: [Nb]

        Returns:
            [Na, Nb] similarity matrix in [0, 1]
        """
        prob_a = logits_a.sigmoid().unsqueeze(1)  # [Na, 1]
        prob_b = logits_b.sigmoid().unsqueeze(0)  # [1, Nb]
        # 1 - |prob_a - prob_b| as similarity
        return 1.0 - (prob_a - prob_b).abs()

    def compute_cost_matrix(
        self,
        student_masks: torch.Tensor,
        teacher_masks: torch.Tensor,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined cost matrix.

        Args:
            student_masks: [Ns, H, W]
            teacher_masks: [Nt, H, W]
            student_boxes: [Ns, 4]
            teacher_boxes: [Nt, 4]
            student_logits: [Ns]
            teacher_logits: [Nt]

        Returns:
            [Ns, Nt] cost matrix (lower = better match)
        """
        # Resize student masks to teacher size if needed
        if student_masks.shape[-2:] != teacher_masks.shape[-2:]:
            student_masks = F.interpolate(
                student_masks.unsqueeze(0),
                size=teacher_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        mask_iou = self.compute_mask_iou(student_masks, teacher_masks)
        box_l1 = self.compute_box_l1(student_boxes, teacher_boxes)
        logit_sim = self.compute_logit_similarity(student_logits, teacher_logits)

        cost = (
            self.mask_iou_weight * (1.0 - mask_iou)
            + self.box_l1_weight * box_l1
            + self.logit_sim_weight * (1.0 - logit_sim)
        )
        return cost

    @staticmethod
    def greedy_assign(cost_matrix: torch.Tensor, num_matches: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy assignment: iteratively pick minimum cost pairs.

        Args:
            cost_matrix: [Ns, Nt] cost matrix
            num_matches: max pairs to match (default: min(Ns, Nt))

        Returns:
            (student_indices, teacher_indices) — both [num_matches]
        """
        ns, nt = cost_matrix.shape
        if num_matches is None:
            num_matches = min(ns, nt)
        else:
            num_matches = min(num_matches, ns, nt)

        if num_matches <= 0:
            device = cost_matrix.device
            empty = torch.zeros(0, device=device, dtype=torch.long)
            return empty, empty

        # Work on a copy
        costs = cost_matrix.clone()
        student_indices = []
        teacher_indices = []

        for _ in range(num_matches):
            # Find global minimum
            flat_idx = costs.reshape(-1).argmin()
            si = flat_idx // nt
            ti = flat_idx % nt

            student_indices.append(si)
            teacher_indices.append(ti)

            # Mask used row and column with large value
            costs[si, :] = float("inf")
            costs[:, ti] = float("inf")

        device = cost_matrix.device
        return (
            torch.tensor(student_indices, device=device, dtype=torch.long),
            torch.tensor(teacher_indices, device=device, dtype=torch.long),
        )

    @torch.no_grad()
    def match(
        self,
        student_masks: torch.Tensor,
        teacher_masks: torch.Tensor,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        max_matches: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full matching pipeline: cost matrix → greedy assignment.

        All tensors are for a single sample (no batch dim).

        Returns:
            (student_indices, teacher_indices) for matched pairs
        """
        cost_matrix = self.compute_cost_matrix(
            student_masks, teacher_masks,
            student_boxes, teacher_boxes,
            student_logits, teacher_logits,
        )
        return self.greedy_assign(cost_matrix, num_matches=max_matches)
