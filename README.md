# llm-from-scratch

Building a GPT-style, character-level language model from scratch in PyTorch to understand transformer internals: tokenization, self-attention, training and generation.

The model is trained on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1.1M characters) and learns to generate Shakespeare-like text one character at a time. The approach follows Andrej Karpathy's *"Let's build GPT: from scratch, in code, spelled out"*.

## Files

| File | What it is |
|---|---|
| `get_gpt.ipynb` | Exploration notebook: data loading, tokenizer, batching, the first bigram model, and the "mathematical trick" behind self-attention |
| `bigram.py` | The training script, grown step by step from a bigram model into a full transformer |
| `test.ipynb` | Checks that the Colab GPU (Tesla T4) is available |

## Setup

```bash
pip install torch
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python bigram.py
```

The final configuration is large (6 layers, 384-dim embeddings, 256-character context), so train it on a GPU. I ran it on a Google Colab T4. The script automatically uses CUDA when it's available.

## How I built it

### 1. Data and tokenizer (`get_gpt.ipynb`)

- Loaded the 1,115,394-character dataset and found **65 unique characters**, which became the vocabulary.
- Built a character-level tokenizer: `stoi` / `itos` lookup tables with `encode` and `decode` functions.
- Encoded the whole text into a `torch.long` tensor and split it **90% train / 10% validation**.
- Wrote `get_batch()`, which samples random chunks of `block_size` characters. The targets are the same chunk shifted one character to the right, so every chunk gives `block_size` training examples (predict the next character from 1, 2, …, `block_size` characters of context).

### 2. Baseline: bigram model

- A single `nn.Embedding(vocab_size, vocab_size)` table: each character directly predicts the logits of the next character. It has no context beyond the current token.
- Before training, the loss was **4.88**. That's a bit above the uniform-guess loss of `-ln(1/65) ≈ 4.17`, as expected for random initial weights. The output was gibberish.
- After 10,000 steps of AdamW (lr `1e-3`, batch 32), the train loss dropped to **~2.38**. The output started to look like words and line breaks, but without meaning.

### 3. The self-attention trick

- Explored how a token can "talk to" the tokens before it: first by averaging previous tokens with a loop, then with a lower-triangular matrix multiply, and finally with `masked_fill(-inf)` + `softmax`. That last version is exactly how causal attention weights are computed.

### 4. Single self-attention head

- Added **position embeddings** (`nn.Embedding(block_size, n_embd)`), so the model knows where each token sits, not just what it is.
- Added one `Head` with key, query and value projections, a causal mask and scaled dot-product attention.
- Tokens can now gather information from earlier context instead of only looking at the current character.

### 5. Multi-head attention

- Replaced the single head with `MultiHeadAttention`: 4 heads of 8 dimensions running in parallel, with their outputs concatenated.
- Several smaller heads can each attend to a different kind of pattern.

### 6. Feed-forward layer

- Added a `FeedForward` MLP (Linear → ReLU → Linear) after attention. Attention is the *communication* step; the MLP is the *computation* step, where each token processes what it gathered.

### 7. Transformer blocks: residuals, projection and LayerNorm

- Wrapped attention and feed-forward into a `Block` and stacked several of them.
- Stacking blocks made the network deep enough to be hard to optimize, so I added:
  - **Residual connections** (`x = x + sa(x)`, `x = x + ffwd(x)`), which let gradients flow straight through the network.
  - A **projection layer** after multi-head attention, back into the residual stream.
  - A 4× wider inner layer in the feed-forward network.
  - **Pre-norm LayerNorm** (`ln1`, `ln2`) before attention and feed-forward, plus a final `ln_f`.

### 8. Scaling up with dropout

- Scaled the model up and added **dropout (0.2)** on the attention weights, the attention projection and the feed-forward output, so the larger model doesn't overfit.
- Final hyperparameters:

| Hyperparameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` (context length) | 256 |
| `n_embd` | 384 |
| `n_head` | 6 (64 dims per head) |
| `n_layer` | 6 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 |
| `max_iters` | 5000 |

## Validation loss across iterations

Over all iterations, the **val loss dropped from ~4.4 to ~1.4**.

| # | Change | Val loss |
|---|---|---|
| 0 | Untrained model (random weights) | ~4.4 |
| 1 | Bigram baseline | ~2.8 |
| 2 | + position embeddings + single attention head | ~2.4 |
| 3 | + multi-head attention (4 × 8) + feed-forward layer | ~2.3 |
| 4 | + stacked blocks, residuals, projection, LayerNorm | ~2.07 |
| 5 | Scaled up + dropout (final) | ~1.4 |

The train loss decreased gradually alongside the val loss at every step.

### Overfitting, and how dropout fixed it

After scaling the model up but **before adding dropout**, the model started to overfit: the train loss kept falling while the val loss stayed noticeably higher, so the model was memorizing the training text instead of generalizing. Adding **dropout (0.2)** and lowering the learning rate to 3e-4 closed that gap, which brought the val loss down to **~1.4**.

### What reduced the val loss

- **Context instead of a single character.** Self-attention lets each prediction use up to `block_size` earlier characters instead of just one.
- **More heads, more patterns.** Multi-head attention captures several kinds of relationships at once for the same compute.
- **Thinking time per token.** The feed-forward layer processes the information that attention gathered.
- **Making depth trainable.** Residual connections and LayerNorm let a stack of blocks train stably. Without them, adding layers didn't help.
- **Scale.** A longer context (8 → 256), wider embeddings (32 → 384) and more layers gave the model much more capacity.
- **Regularization.** Dropout (0.2) together with a lower learning rate (1e-3 → 3e-4) kept the larger model from overfitting, so the train and val losses stayed close.

## Bugs I hit along the way

- `idx[:, -block_size]` in `generate()` selected one column instead of slicing the context window, which crashed generation. The fix was `idx[:, -block_size:]`.
- `n_embd / 4` produced a float head size, so `nn.Linear` failed. The fix was integer division, `n_embd // 4`.
- `n_embd = 32` with `n_head = 6` gave a 5-dimensional head, and 6 × 5 = 30 didn't match the 32-dimensional projection layer (`mat1 and mat2 shapes cannot be multiplied`). The fix was to pick an `n_embd` divisible by `n_head` (384 / 6 = 64).

## Next steps

- Apply the final `ln_f` in `forward()` and dropout after the attention projection. Both are defined in the script but not used yet.
- Scale the attention scores by `head_size**-0.5` instead of `n_embd**-0.5`.
- Save checkpoints and log losses to a file.
- Move from character-level to sub-word (BPE) tokenization.
