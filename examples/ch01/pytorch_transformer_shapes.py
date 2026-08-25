"""A tiny, runnable shape lab for the Transformer attention chapter.

Run:
    python examples/ch01/pytorch_transformer_shapes.py

The script intentionally prints every important tensor shape. It is a learning
aid, not an optimized attention kernel.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    """Return a broadcastable causal mask with shape [1, 1, S, S]."""
    mask = torch.full(
        (1, 1, seq_len, seq_len),
        float("-inf"),
        device=device,
    )
    return torch.triu(mask, diagonal=1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention for q/k/v laid out as [B, N, S, D]."""
    head_dim = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    print(f"scores = QK^T / sqrt(D): {tuple(scores.shape)}")

    if mask is not None:
        scores = scores + mask
        print(f"scores + broadcast mask: {tuple(scores.shape)}")

    probabilities = F.softmax(scores.float(), dim=-1).type_as(q)
    print(f"softmax probabilities: {tuple(probabilities.shape)}")

    output = torch.matmul(probabilities, v)
    print(f"probabilities @ V: {tuple(output.shape)}")
    return output, probabilities


class ShapeOnlyMultiHeadAttention(nn.Module):
    """Small self-attention module whose purpose is to expose shape changes."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        print(f"after view [B,S,N,D]: {tuple(x.shape)}")
        x = x.transpose(1, 2)
        print(f"after transpose [B,N,S,D]: {tuple(x.shape)}")
        return x

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        print(f"before merging [B,S,N,D]: {tuple(x.shape)}")
        print(f"is contiguous after transpose: {x.is_contiguous()}")
        x = x.contiguous().view(batch_size, seq_len, self.hidden_size)
        print(f"after contiguous + view [B,S,H]: {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"input x [B,S,H]: {tuple(x.shape)}")

        q_projected = self.wq(x)
        k_projected = self.wk(x)
        v_projected = self.wv(x)
        print(f"projected Q/K/V [B,S,H]: {tuple(q_projected.shape)}")

        q = self.split_heads(q_projected)
        k = self.split_heads(k_projected)
        v = self.split_heads(v_projected)

        mask = causal_mask(x.size(1), device=x.device)
        print(f"causal mask [1,1,S,S]: {tuple(mask.shape)}")
        attention_output, probabilities = scaled_dot_product_attention(
            q, k, v, mask
        )

        merged = self.merge_heads(attention_output)
        output = self.wo(merged)
        print(f"output projection [B,S,H]: {tuple(output.shape)}")
        return output, probabilities


def embedding_demo() -> None:
    vocab_size = 20
    hidden_size = 8
    token_ids = torch.tensor([[2, 5, 9, 3], [1, 7, 4, 6]])
    embedding = nn.Embedding(vocab_size, hidden_size)
    embedded = embedding(token_ids)

    print("\n=== Embedding ===")
    print(f"token ids [B,S]: {tuple(token_ids.shape)}; dtype={token_ids.dtype}")
    print(f"embedded [B,S,H]: {tuple(embedded.shape)}")


def attention_demo() -> None:
    torch.manual_seed(7)
    batch_size, seq_len, hidden_size, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention = ShapeOnlyMultiHeadAttention(hidden_size, num_heads)

    print("\n=== Multi-head self-attention ===")
    output, probabilities = attention(x)
    print(f"final output: {tuple(output.shape)}")

    row_sums = probabilities.sum(dim=-1)
    print(f"softmax row sums shape: {tuple(row_sums.shape)}")
    print("all row sums are approximately 1:", bool(torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-6
    )))


if __name__ == "__main__":
    embedding_demo()
    attention_demo()
