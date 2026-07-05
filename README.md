# MyTorch: A Deep Learning Library Built from Scratch, with a Transformer-Based Speech Recognition System

A NumPy/PyTorch deep learning stack built module-by-module in four stages — from manually-derived backpropagation through linear layers up to a GPT-2-style Transformer decoder — then used to build an attention-based encoder-decoder for end-to-end speech recognition.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-attention-lightgrey)

## Project structure

Each stage builds directly on the previous one. The autograd fundamentals from the core library stage underpin the convolutional layers in the CNN stage, which underpin the RNN/CTC machinery in the sequence-model stage, which underpin the attention primitives in the Transformer stage. That Transformer decoder is then reused as a pretrained initialization for the speech recognition decoder in Part 2.

## Part 1 — MyTorch: NumPy Deep Learning Library

Built from scratch, mirroring PyTorch's API — no autograd graph, no `nn.Module` under the hood. Every layer implements its own explicit `forward()`/`backward()` pair from hand-derived closed-form gradients, validated against a reference PyTorch implementation via `np.allclose` at each stage.

<details>
<summary><strong>Stage 1 — Core: Linear layers, activations, losses, SGD, BatchNorm</strong></summary>

**Built:** `Linear` (forward `Z = AW^T + b` + manual backward for `dLdA/dLdW/dLdb`), activations (Identity, Sigmoid, Tanh, ReLU, GELU, Swish, Softmax), losses (MSE, CrossEntropy), `SGD` with momentum (per-parameter velocity buffers, decoupled from layers, PyTorch-style), and `BatchNorm1d` with separate train/eval behavior. Composed into 3 test MLPs (0/1/4 hidden layers).

**Design decisions:** cache-then-differentiate pattern (forward stores intermediates needed for backward, modules otherwise stateless); correctness driven by careful shape-matching since there's no autograd graph to catch errors automatically.

**Hardest parts:** deriving Softmax+CrossEntropy backward without double-counting the Jacobian; BatchNorm backward, since batch statistics create cross-sample dependencies that make `dLdZ` non-elementwise.

**Validated against:** PyTorch reference tensors/`.backward()`, `np.allclose` at `atol=1e-4`, verified component-by-component then on full MLPs.
</details>

<details>
<summary><strong>Stage 2 — CNNs: Conv1d/2d, pooling, resampling, MLP↔CNN equivalence</strong></summary>

**Built:** `Conv1d`/`Conv2d` (stride-1 core + strided wrapper), `MaxPool2d`/`MeanPool2d`, `Upsample`/`Downsample`, `Flatten`, a full composed CNN, and a scanning-MLP-to-CNN weight conversion proving a weight-shared MLP is equivalent to a CNN.

**Backprop approach:** sliding-window + `np.tensordot` (not im2col) — forward tensordots each input window against all filters; `dLdW` correlates input windows against the upstream gradient; `dLdA` uses the full-convolution trick (zero-pad `dLdZ`, flip `W`, tensordot again). Stride/padding handled by composing the stride-1 core with separate pad/downsample layers.

**Hardest parts:** a `dLdb` shape bug — `(C,1)` instead of `(C,)` — silently failed validation tests despite correct values, tracked down to a one-line fix; MaxPool backward required caching per-window argmax indices to route gradients correctly; the MLP→CNN weight conversion was the single hardest piece of this stage.

**Validated against:** per-layer reference tests; Conv1d passed fully on first attempt, Conv2d required the `dLdb` fix above.
</details>

<details>
<summary><strong>Stage 3 — Sequence models: RNN, GRU, CTC loss & decoding</strong></summary>

**Built:** RNN cell (single-step forward/backward + multi-layer BPTT classifier), GRU cell (reset/update/candidate gates, all six weight matrices derived by hand), full CTC forward-backward algorithm (blank-extended targets, α/β dynamic programming with skip-connections, posterior γ, gradient backprop into logits), and greedy + beam-search CTC decoders.

**Hardest parts:** BPTT hidden-state caching (an off-by-one silently broke all downstream gradients); GRU backward's shared intermediate terms across three gates; CTC's skip-connection indexing (only skip a blank when surrounding symbols differ) while keeping α/β numerically stable; beam search path-merging without losing the eventual best path.

**Validated against:** a reference test suite scoring each component independently (RNN forward/backward, BPTT, GRU, CTC extend/forward/backward, greedy + beam decoding).
</details>

<details>
<summary><strong>Stage 4 — Decoder-only Transformer for causal language modeling</strong></summary>

**Built:** arbitrary-dimension `Linear`/`Softmax`, `ScaledDotProductAttention`, `MultiHeadAttention`, then a full GPT-2-style decoder-only Transformer — causal + padding masking, sinusoidal positional encoding, self-attention/feed-forward sublayers stacked into `SelfAttentionDecoderLayer`s — plus greedy decoding and a full training/validation/generation loop (`LMTrainer`).

**Task:** autoregressive next-token prediction (character-level), evaluated by per-character perplexity.

**Result:** **2.76 validation perplexity.**

**Hardest parts:** composing causal + padding masks correctly across batched variable-length sequences; numerically stable log-softmax for generation; building checkpointing/LR scheduling with minimal framework scaffolding.
</details>

## Part 2 — Attention-Based Speech Recognition

An end-to-end, Listen-Attend-Spell-style ASR system — converting log-mel filterbank speech features directly to character-level transcripts — built on the same attention stack developed in Part 1.

**Data:** LibriSpeech-derived (`train-clean-100` for training, `dev-clean`/`test-clean` for validation/test); evaluated on Character Error Rate (CER).

**Architecture:** pre-norm Transformer encoder-decoder — `SelfAttentionEncoderLayer` stack for the speech encoder, `CrossAttentionLayer` + `CrossAttentionDecoderLayer` for the text decoder attending over encoder outputs. ~30M parameters. Supports both greedy and beam-search decoding.

**Training setup:**

| Component | Choice |
|---|---|
| Loss | Cross-entropy with causal masking (no label leakage) |
| Precision | Mixed-precision with gradient accumulation |
| LR control | Custom `create_optimizer` with layer-wise freezing/unfreezing |
| Initialization | **Pretrained Transformer decoder from Part 1**, frozen then progressively unfrozen |
| Tracking | Weights & Biases |
| Curriculum | Staged: progressive layer unfreezing + ramped-up regularization/dropout |

**Result: 8.78% CER.**

**Key decisions & hardest bug:**

- **Biggest lever:** reusing the pretrained Part 1 decoder as the ASR decoder's initialization (freeze-then-unfreeze) gave a strong head start over training the encoder-decoder from scratch.
- **Hardest bug — beam search, multi-day debug:** (1) the `was_finished` mask wasn't reordered alongside beam sequences after `topk` selection, silently corrupting scores; (2) finished beams kept accumulating log-probs past EOS instead of freezing, producing NaNs. Fixed by tracking `was_finished` pre-reorder, reordering it in lockstep with `beam_indices`, and using `torch.where` to freeze already-finished beam scores.
- **What I'd do differently:** add shape/value invariant assertions to beam search from the start (e.g. "finished mask and score tensor must share beam ordering at every step") — would have caught the reordering bug immediately instead of after days of print-statement debugging.

## Results summary

| Stage | Metric | Score |
|---|---|---|
| MyTorch library (core, CNN, RNN/CTC) | Reference-implementation validation | All components verified against PyTorch reference |
| Language Model | Validation perplexity | **2.76** |
| Speech Recognition | Character Error Rate | **8.78%** |

## Tech stack

`Python` · `NumPy` (from-scratch autodiff-free backprop) · `PyTorch` · `Transformers / Multi-Head Attention` · `CTC` · `RNN/GRU` · `Mixed-precision training` · `Weights & Biases`

---
*Part of a [broader project series](../) covering MLP-based speech classification, CNN-based face recognition, and CTC-based speech recognition.*
