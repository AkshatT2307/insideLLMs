"""
hooks.py — Forward-hook based activation capture for transformer models.

Registers hooks on self_attn, mlp, and the full decoder layer to capture
Δa (attention output pre-residual), Δm (MLP output pre-residual), and
x^{ℓ+1} (full layer output post-residual).

Pooling (last-token and mean-over-non-pad) happens *inside* the hook to
avoid storing full (batch, seq, hidden) tensors — only pooled (batch, d)
vectors are kept on CPU.
"""

import torch
from typing import Dict, List, Optional


class ActivationStore:
    """
    Manages forward hooks for capturing and pooling activations.

    Usage
    -----
    >>> store = ActivationStore(num_layers=28)
    >>> store.register_hooks(model)
    >>> store.set_mask(attention_mask)          # before each forward pass
    >>> model(input_ids=..., attention_mask=...) # hooks fire automatically
    >>> results = store.get_batch_results()     # {comp/mode: (B,L,d)}
    >>> store.remove_hooks()                    # cleanup
    """

    COMPONENTS = ("attn", "mlp", "layer")
    MODES = ("last", "mean")

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.attention_mask: Optional[torch.Tensor] = None
        self._pooled: Dict = {}          # (component, mode, layer_idx) → (B, d) cpu fp16
        self._handles: List = []

    # ── public API ───────────────────────────────────────────────────────

    def set_mask(self, mask: torch.Tensor) -> None:
        """
        Provide the attention mask for the *current* batch.

        Parameters
        ----------
        mask : Tensor of shape (batch, seq_len)
            1 for real tokens, 0 for padding tokens.
        """
        self.attention_mask = mask

    def register_hooks(self, model: torch.nn.Module, model_info: Optional[dict] = None) -> None:
        """
        Register forward hooks on every decoder layer of the model.

        Parameters
        ----------
        model : nn.Module
            The loaded transformer model.
        model_info : dict, optional
            Architecture info from ``model_loader.load_model_from_config``.
            If None, falls back to ``model.model.layers`` (Qwen/LLaMA-style).
        """
        if model_info is not None:
            # Use architecture registry for model-agnostic layer resolution
            arch = model_info["arch_config"]
            layers_attr = arch["layers_attr"]
            attn_attr = arch["attn_attr"]
            mlp_attr = arch["mlp_attr"]

            # Resolve layers list
            obj = model
            for attr in layers_attr:
                obj = getattr(obj, attr)
            layers = obj
        else:
            # Legacy fallback: Qwen/LLaMA-style
            layers = model.model.layers
            attn_attr = "self_attn"
            mlp_attr = "mlp"

        if len(layers) != self.num_layers:
            raise ValueError(
                f"Model has {len(layers)} layers but ActivationStore "
                f"was initialised with num_layers={self.num_layers}"
            )

        for i, layer in enumerate(layers):
            # Δaℓ — self-attention output (before residual add)
            attn_module = getattr(layer, attn_attr)
            h1 = attn_module.register_forward_hook(
                self._make_hook("attn", i, output_is_tuple=True)
            )
            # Δmℓ — MLP output (before residual add)
            mlp_module = getattr(layer, mlp_attr)
            h2 = mlp_module.register_forward_hook(
                self._make_hook("mlp", i, output_is_tuple=False)
            )
            # x^{ℓ+1} — full decoder layer output (after both residual adds)
            h3 = layer.register_forward_hook(
                self._make_hook("layer", i, output_is_tuple=True)
            )
            self._handles.extend([h1, h2, h3])

        print(
            f"  Registered {len(self._handles)} hooks "
            f"(3 per layer × {self.num_layers} layers)"
        )

    def get_batch_results(self) -> Dict[str, torch.Tensor]:
        """
        Assemble pooled per-layer activations into stacked tensors.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``"attn/last"``, ``"attn/mean"``, ``"mlp/last"``,
            ``"mlp/mean"``, ``"layer/last"``, ``"layer/mean"``.
            Each value has shape ``(batch, num_layers, hidden_dim)`` in
            float16 on CPU.

        Raises
        ------
        RuntimeError
            If any hook failed to capture its activation.
        """
        results: Dict[str, torch.Tensor] = {}

        for comp in self.COMPONENTS:
            for mode in self.MODES:
                stack = []
                for i in range(self.num_layers):
                    key = (comp, mode, i)
                    if key not in self._pooled:
                        raise RuntimeError(
                            f"Missing activation for {comp}/{mode}/layer_{i}. "
                            f"Hook may not have fired."
                        )
                    stack.append(self._pooled[key])
                # list of (B, d) → (B, L, d)
                results[f"{comp}/{mode}"] = torch.stack(stack, dim=1)

        self._pooled.clear()
        return results

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        print("  All hooks removed.")

    # ── internals ────────────────────────────────────────────────────────

    def _pool(self, act: torch.Tensor):
        """
        Pool a (batch, seq, d) activation tensor into last-token and
        mean-of-non-pad-token vectors.

        The attention mask is broadcast to the activation's device so
        this works even with ``device_map='auto'``.
        """
        # Move mask to same device as activation
        mask = self.attention_mask.to(device=act.device, dtype=torch.float32)
        mask = mask.unsqueeze(-1)  # (B, S, 1)

        # Last token — with left-padding the rightmost token is always real
        last = act[:, -1, :].to(dtype=torch.float16).cpu()

        # Mean — weighted by mask to exclude pad tokens
        mean = (act.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        mean = mean.to(dtype=torch.float16).cpu()

        return last, mean

    def _make_hook(self, component: str, layer_idx: int, output_is_tuple: bool):
        """
        Return a hook function that pools the output and stores it.

        Parameters
        ----------
        component : str
            One of ``"attn"``, ``"mlp"``, ``"layer"``.
        layer_idx : int
            Index of the decoder layer (0-based).
        output_is_tuple : bool
            Legacy hint. Output is now inspected dynamically to support both
            single Tensors and tuples.
        """

        def hook_fn(module, input, output):
            act = output
            while isinstance(act, tuple):
                act = act[0]
            last, mean = self._pool(act)
            self._pooled[(component, "last", layer_idx)] = last
            self._pooled[(component, "mean", layer_idx)] = mean

        return hook_fn
