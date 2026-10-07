"""D4 对称增强：4×4 棋盘的旋转与镜像在规则上完全等价。

两类用法必须区分，混用会让策略标签指错方向：

* **价值样本**（决策局面 + afterstate + 标量回报）：局面本身与标量标签都不含方向语义，
  因此可以对每个样本独立施加任意 D4 变换。
* **策略样本**（决策局面 + 四方向访问分布）：方向标签必须与棋盘一起变换。
  这里沿用旧版验证过的口径——`torch.rot90` 是逆时针旋转，水平翻转作用在旋转之后。
"""

from __future__ import annotations

import torch

BOARD_CELLS = 16


def _sample_transforms(
    count: int, rng: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = torch.randint(0, 4, (count,), generator=rng, device=device)
    reflections = torch.randint(0, 2, (count,), generator=rng, device=device)
    return rotations, reflections


def apply_boards(boards: torch.Tensor, rotations: torch.Tensor, reflections: torch.Tensor) -> torch.Tensor:
    shaped = boards.reshape(-1, 4, 4)
    candidates = torch.stack([torch.rot90(shaped, turns, dims=(-2, -1)) for turns in range(4)], dim=1)
    selected = torch.gather(candidates, 1, rotations[:, None, None, None].expand(-1, 1, 4, 4)).squeeze(1)
    flipped = torch.where(reflections[:, None, None].bool(), selected.flip(-1), selected)
    return flipped.reshape(-1, BOARD_CELLS)


def transform_policy_targets(
    legal_mask: torch.Tensor, visits: torch.Tensor, rotations: torch.Tensor, reflections: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """把四方向标签旋转到与变换后的棋盘一致。

    动作顺序是 UP=0, RIGHT=1, DOWN=2, LEFT=3（见 ADR 0004）。逆时针旋转一次等价于
    动作编号减一，水平镜像把 LEFT/RIGHT 对调。
    """
    new_mask = torch.zeros_like(legal_mask)
    new_visits = torch.zeros_like(visits)
    for action in range(4):
        rotated = (action - rotations) % 4
        reflected = torch.where(reflections.bool(), (-rotated) % 4, rotated)
        index = reflected[:, None]
        new_mask.scatter_(1, index, torch.gather(legal_mask, 1, torch.full_like(index, action)))
        new_visits.scatter_(1, index, torch.gather(visits, 1, torch.full_like(index, action)))
    return new_mask, new_visits


def augment_value(afterstates: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """价值样本：只变换棋盘。"""
    rotations, reflections = _sample_transforms(afterstates.shape[0], rng, afterstates.device)
    return apply_boards(afterstates, rotations, reflections)


def augment_policy(
    decisions: torch.Tensor,
    legal_mask: torch.Tensor,
    visits: torch.Tensor,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """策略样本：棋盘、合法掩码、访问分布使用**同一个**变换。"""
    rotations, reflections = _sample_transforms(decisions.shape[0], rng, decisions.device)
    boards = apply_boards(decisions, rotations, reflections)
    mask, target = transform_policy_targets(legal_mask, visits, rotations, reflections)
    return boards, mask, target


def augment_joint(
    decisions: torch.Tensor,
    afterstates: torch.Tensor,
    legal_mask: torch.Tensor,
    visits: torch.Tensor,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """联合样本：决策局面与 afterstate 属于同一步，必须用同一个变换，
    否则价值标签会落在另一个局面上。"""
    rotations, reflections = _sample_transforms(decisions.shape[0], rng, decisions.device)
    mask, target = transform_policy_targets(legal_mask, visits, rotations, reflections)
    return (
        apply_boards(decisions, rotations, reflections),
        apply_boards(afterstates, rotations, reflections),
        mask,
        target,
    )
