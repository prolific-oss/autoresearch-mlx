"""
Text generation script for autoresearch-mlx checkpoints.
Loads a saved checkpoint and samples completions from a prompt.

Usage:
    uv run generate.py --prompt "The history of machine learning"
    uv run generate.py --checkpoint checkpoints/5efc7aa.safetensors --prompt "Once upon a time"
    uv run generate.py --list-checkpoints
"""

import argparse
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from prepare import Tokenizer


# ---------------------------------------------------------------------------
# Model definition (mirrors train.py — must stay in sync with architecture)
# ---------------------------------------------------------------------------

def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    indices = mx.arange(seq_len)
    blocked = indices[None, :] > indices[:, None]
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


def create_sliding_window_mask(seq_len, window_size, dtype=mx.float32):
    indices = mx.arange(seq_len)
    causal = indices[None, :] > indices[:, None]
    too_far = (indices[:, None] - indices[None, :]) >= window_size
    blocked = causal | too_far
    return mx.where(blocked, mx.array(float("-inf"), dtype=dtype), mx.array(0.0, dtype=dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config["n_head"]
        self.n_kv_head = config["n_kv_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config["n_layer"])
            else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)

    def __call__(self, x, ve, mask):
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        scale = 1.0 / math.sqrt(self.head_dim)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=False)
        self.c_proj = nn.Linear(4 * config["n_embd"], config["n_embd"], bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.blocks = [Block(config, i) for i in range(config["n_layer"])]
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.resid_lambdas = mx.ones((config["n_layer"],), dtype=mx.float32)
        self.x0_lambdas = mx.zeros((config["n_layer"],), dtype=mx.float32)
        head_dim = config["n_embd"] // config["n_head"]
        kv_dim = config["n_kv_head"] * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config["vocab_size"], kv_dim)
            for i in range(config["n_layer"])
            if has_ve(i, config["n_layer"])
        }
        self._mask_cache = {}

    def _compute_window_sizes(self, config):
        pattern = config["window_pattern"].upper()
        long_window = config["sequence_len"]
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config["n_layer"]):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def _get_masks(self, seq_len):
        unique_windows = set(self.window_sizes)
        for window_size in unique_windows:
            key = (seq_len, window_size)
            if key not in self._mask_cache:
                if window_size >= seq_len:
                    self._mask_cache[key] = create_additive_causal_mask(seq_len)
                else:
                    self._mask_cache[key] = create_sliding_window_mask(seq_len, window_size)
        return [self._mask_cache[(seq_len, window_size)] for window_size in self.window_sizes]

    def __call__(self, idx):
        _, seq_len = idx.shape
        masks = self._get_masks(seq_len)
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, masks[i])
        x = norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)
        return logits


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_token(logits, temperature, top_k):
    """Sample next token from logits (1D array of shape [vocab_size])."""
    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.shape[0])
        # find the kth-largest value and mask everything below it
        sorted_logits = mx.sort(logits)
        threshold = sorted_logits[-k]
        logits = mx.where(logits < threshold, mx.full(logits.shape, float("-inf")), logits)

    return int(mx.random.categorical(logits.reshape(1, -1)).item())


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, seed):
    mx.random.seed(seed)

    bos = tokenizer.get_bos_token_id()
    prompt_ids = tokenizer.encode(prompt, prepend=bos)
    ids = mx.array([prompt_ids], dtype=mx.int32)
    max_seq_len = model.config["sequence_len"]

    print(prompt, end="", flush=True)

    generated = 0
    for _ in range(max_new_tokens):
        if ids.shape[1] >= max_seq_len:
            break

        logits = model(ids)          # (1, seq_len, vocab_size)
        mx.eval(logits)
        next_logits = logits[0, -1]  # (vocab_size,)

        next_id = sample_token(next_logits, temperature, top_k)
        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        token_str = tokenizer.decode([next_id])
        print(token_str, end="", flush=True)
        generated += 1

    print()
    return generated


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def list_checkpoints(checkpoint_dir="checkpoints"):
    if not os.path.isdir(checkpoint_dir):
        print(f"No checkpoints directory found at '{checkpoint_dir}'.")
        return []

    configs = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith("_config.json"))
    if not configs:
        print("No checkpoints found. Train a model first with: uv run train.py")
        return []

    rows = []
    for cfg_file in configs:
        cfg_path = os.path.join(checkpoint_dir, cfg_file)
        with open(cfg_path) as f:
            cfg = json.load(f)
        commit = cfg.get("commit", cfg_file.replace("_config.json", ""))
        val_bpb = cfg.get("val_bpb", "n/a")
        weights = cfg_file.replace("_config.json", ".safetensors")
        weights_path = os.path.join(checkpoint_dir, weights)
        exists = "yes" if os.path.exists(weights_path) else "MISSING"
        rows.append((commit, val_bpb, exists, weights_path))

    print(f"{'COMMIT':<12} {'VAL_BPB':<12} {'WEIGHTS':<8} PATH")
    print("-" * 60)
    for commit, val_bpb, exists, path in rows:
        bpb_str = f"{val_bpb:.6f}" if isinstance(val_bpb, float) else str(val_bpb)
        print(f"{commit:<12} {bpb_str:<12} {exists:<8} {path}")

    return rows


def resolve_checkpoint(checkpoint_arg, checkpoint_dir="checkpoints"):
    """Resolve a commit hash or full path to (weights_path, config_path)."""
    if checkpoint_arg is None:
        # pick the checkpoint with the lowest val_bpb
        configs = sorted(
            f for f in os.listdir(checkpoint_dir) if f.endswith("_config.json")
        )
        if not configs:
            raise FileNotFoundError("No checkpoints found. Run 'uv run train.py' first.")
        best_cfg, best_bpb = None, float("inf")
        for cfg_file in configs:
            with open(os.path.join(checkpoint_dir, cfg_file)) as f:
                cfg = json.load(f)
            bpb = cfg.get("val_bpb", float("inf"))
            if bpb < best_bpb:
                best_bpb = bpb
                best_cfg = cfg_file
        commit = best_cfg.replace("_config.json", "")
        print(f"No checkpoint specified — using best: {commit} (val_bpb={best_bpb:.6f})")
        weights_path = os.path.join(checkpoint_dir, f"{commit}.safetensors")
        config_path = os.path.join(checkpoint_dir, best_cfg)
    elif os.path.isfile(checkpoint_arg):
        weights_path = checkpoint_arg
        config_path = checkpoint_arg.replace(".safetensors", "_config.json")
    else:
        # treat as commit hash
        weights_path = os.path.join(checkpoint_dir, f"{checkpoint_arg}.safetensors")
        config_path = os.path.join(checkpoint_dir, f"{checkpoint_arg}_config.json")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    return weights_path, config_path


def load_model(weights_path, config_path):
    with open(config_path) as f:
        config = json.load(f)

    model = GPT(config)

    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    return model, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text from an autoresearch-mlx checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .safetensors file, or commit hash (default: best val_bpb)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints (default: checkpoints/)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to complete")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature: 0=greedy, 1=unscaled (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling: 0=disabled (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all available checkpoints and exit")
    args = parser.parse_args()

    if args.list_checkpoints:
        list_checkpoints(args.checkpoint_dir)
        return

    weights_path, config_path = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)
    print(f"Loading checkpoint: {weights_path}")
    model, config = load_model(weights_path, config_path)
    print(f"Model: {config['n_layer']}L × {config['n_embd']}d | val_bpb={config.get('val_bpb', 'n/a')}")

    tokenizer = Tokenizer.from_directory()

    print(f"\n{'─' * 60}")
    n = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"{'─' * 60}")
    print(f"Generated {n} tokens.")


if __name__ == "__main__":
    main()
