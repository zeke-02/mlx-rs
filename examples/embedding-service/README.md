# Embedding Service Example

This example demonstrates how to use mlx-rs to load a Qwen3 embedding model from a local directory and extract text embeddings.

## Overview

This example shows how to:

- Load a Qwen3 embedding model from a local directory
- Apply chat templates to format input text (if available)
- Extract embeddings from the model's hidden states
- Display the resulting embedding vectors

## Usage

### Required Files

Your model directory must contain:

- `tokenizer.json` - Required for tokenization
- `config.json` - Required for model configuration
- `model.safetensors` - Required for model weights (or `model.safetensors.index.json` for sharded models)
- `tokenizer_config.json` - Optional, only needed for chat template support

### Command-line Options

- `--prompt`: The text to generate embeddings for (default: "hello")
- `--seed`: Random seed for reproducibility (default: 0)
- `--model-dir`: Path to local model directory (required)

### Example

```bash
cargo run --release -p embedding-service -- --model-dir /path/to/model --prompt "hello world"
```

## How It Works

1. **Model Loading**: The example loads the model from a local directory using the `load_qwen3_model()` function from the mlx-lm crate.

2. **Chat Template**: If the model has a `tokenizer_config.json` file with a chat template, it's applied to format the input text appropriately using the `apply_chat_template()` function from mlx-lm-utils.

3. **Tokenization**: The input text is tokenized using the model's tokenizer.

4. **Embedding Extraction**: Instead of generating text, we extract the hidden states from the model by calling `model.model.forward()` directly, bypassing the language modeling head. We use the last token's hidden state as the embedding vector.

5. **Normalization**: The embedding is normalized to unit length (L2 norm = 1), making cosine similarity equivalent to dot product.

6. **Output**: The embedding vector is displayed, showing the first 10 dimensions, total dimensions, and L2 norm.

## Key Differences from Text Generation

Unlike text generation examples (like mistral), this example:

- Performs a single forward pass instead of iterative generation
- Extracts hidden states from `model.model.forward()` instead of using the full model with the language modeling head
- Returns embedding vectors instead of generated tokens
- Normalizes embeddings to unit length for similarity comparisons

## Related Python Example

This example is based on the Python mlx-lm embedding example:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
