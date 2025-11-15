# ModernBERT Example

This example demonstrates how to use ModernBERT models with MLX-RS for generating text embeddings.

## Architecture

ModernBERT is an improved BERT architecture with several key differences from standard BERT:

- **RoPE (Rotary Position Embeddings)**: Uses rotary position embeddings instead of absolute position embeddings
- **Pre-quantized weights**: Supports 8-bit quantization with group size 64 for efficient inference
- **Fused QKV projection**: More efficient attention computation with a single projection matrix
- **No token type embeddings**: Simplified architecture for single-segment inputs
- **Pooled output**: Returns both sequence output and CLS token pooling

## Model Structure

```
ModernBert
├── embeddings (ModernBertEmbeddings)
│   ├── tok_embeddings (QuantizedEmbedding)
│   └── norm (LayerNorm)
├── layers (ModernBertEncoder)
│   └── layers[0..21] (ModernBertLayer)
│       ├── attn (ModernBertAttention)
│       │   ├── Wqkv (QuantizedLinear) - Fused Q/K/V projection
│       │   ├── Wo (QuantizedLinear) - Output projection
│       │   └── rope (RoPE)
│       ├── mlp (ModernBertMLP)
│       │   ├── Wi (QuantizedLinear) - Input projection
│       │   └── Wo (QuantizedLinear) - Output projection
│       └── mlp_norm (LayerNorm)
└── final_norm (LayerNorm)
```

## Usage

### Loading a Model

```rust
use model::{ModernBert, ModernBertInput, ModelArgs};
use std::path::PathBuf;

fn load_model(model_dir: &PathBuf) -> Result<ModernBert> {
    // Load config
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    // Create model
    let mut model = ModernBert::new(&model_args)?;

    // Load weights
    let safetensors_path = model_dir.join("model.safetensors");
    model.load_safetensors(&safetensors_path)?;

    Ok(model)
}
```

### Running Inference

```rust
use mlx_rs::Array;

fn extract_embeddings(model: &mut ModernBert, input_ids: &Array) -> Result<Array> {
    let input = ModernBertInput::from(input_ids);
    let output = model.forward(input)?;
    
    // Use pooler_output (CLS token) for sentence embeddings
    Ok(output.pooler_output)
}
```

### Complete Example

```bash
# Run the example (update model path in main.rs first)
cargo run --release
```

## Configuration

The model configuration is loaded from `config.json`. Key parameters include:

- `vocab_size`: Vocabulary size (typically 50368)
- `hidden_size`: Hidden dimension size (768 for base model)
- `num_hidden_layers`: Number of transformer layers (22 for base model)
- `num_attention_heads`: Number of attention heads (12 for base model)
- `intermediate_size`: FFN intermediate size (1152 for base model)
- `local_rope_theta`: RoPE base frequency (10000.0)
- `layer_norm_eps`: Layer normalization epsilon (1e-05)

## Quantization

ModernBERT models are typically distributed with pre-quantized weights:

- **Bits**: 8-bit quantization
- **Group size**: 64 elements per quantization group
- **Format**: Weights stored as (weight, scales, biases) triplets in safetensors

The model automatically loads these pre-quantized weights using `QuantizedLinear` and `QuantizedEmbedding` layers.

## Output Format

The model returns a `ModernBertOutput` struct with:

- `last_hidden_state`: Full sequence output `[batch_size, seq_length, hidden_size]`
- `pooler_output`: CLS token embedding `[batch_size, hidden_size]`

For sentence embeddings, use `pooler_output` which extracts the first token (CLS) from the sequence.

## Performance Tips

1. **Use eval()**: Call `mlx_rs::transforms::eval()` to trigger computation
2. **Batch processing**: Process multiple texts together for better throughput
3. **Pre-quantized models**: Use 8-bit quantized models for faster inference
4. **Normalize embeddings**: Apply L2 normalization for similarity search tasks

## Example Output

```
[INFO] Using model directory: "path/to/modernbert-model"
[INFO] Loading tokenizer...
[INFO] Loading model...
[INFO] Input text: Hello, this is a test sentence for ModernBERT embeddings.
[INFO] Tokens: [101, 7592, 1010, ...]
[INFO] Extracting embeddings...
[INFO] Embedding shape: [768]
[INFO] L2 norm: 12.345678
[INFO] First 10 dimensions of normalized embedding:
  dim[0]: 0.123456
  dim[1]: -0.234567
  ...
[SUCCESS] ModernBERT model loaded and inference completed successfully!
```

## References

- ModernBERT paper: [Link to paper when available]
- Original implementation: [HuggingFace ModernBERT](https://huggingface.co/models?search=modernbert)
