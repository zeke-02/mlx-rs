use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::scaled_dot_product_attention,
    linalg::norm_l2,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub hidden_activation: String,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    pub rope_local_base_freq: f32,
    pub rope_theta: f32,
    pub sliding_window: i32,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub use_bidirectional_attention: bool,
    pub query_pre_attn_scalar: i32,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum Gemma3Error {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error("Invalid number of layers: {0}")]
    InvalidNumLayers(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Creates a bidirectional sliding window attention mask.
/// Each position can attend to tokens within ±(window_size/2) positions.
fn create_bidirectional_window_mask(seq_length: i32, window_size: i32) -> Result<Array, Exception> {
    // Create position indices [0, 1, 2, ..., seq_length-1]
    let positions = Array::arange::<_, i32>(None, seq_length, None)?;

    // Expand to [seq_length, 1] and [1, seq_length] for broadcasting
    let pos_i = positions.expand_dims(1)?; // [seq_length, 1]
    let pos_j = positions.expand_dims(0)?; // [1, seq_length]

    // Calculate absolute distance between positions
    let distance = pos_i.subtract(&pos_j)?.abs()?;

    // Create mask: distance <= window_size/2
    let half_window = window_size / 2;
    distance.le(&Array::from_int(half_window))
}

/// Attention type for Gemma3 layers
#[derive(Debug, Clone)]
pub enum AttentionType {
    /// Full attention - attends to all tokens in the sequence
    FullAttention,
    /// Sliding attention - attends only to tokens within a sliding window
    SlidingAttention { window_size: i32 },
}

// ============================================================================
// Gemma3Attention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3Attention {
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    scale: f32,
    attention_type: AttentionType,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    q_norm: nn::RmsNorm,

    #[param]
    k_norm: nn::RmsNorm,

    local_rope: nn::Rope,
    global_rope: nn::Rope,
}

impl Gemma3Attention {
    pub fn new(args: &ModelArgs, attention_type: AttentionType) -> Result<Self, Gemma3Error> {
        let num_attention_heads = args.num_attention_heads;
        let num_key_value_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;

        // Q projection: hidden_size -> num_heads * head_dim
        let q_proj = nn::LinearBuilder::new(args.hidden_size, num_attention_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;

        // K projection: hidden_size -> num_kv_heads * head_dim
        let k_proj = nn::LinearBuilder::new(args.hidden_size, num_key_value_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;

        // V projection: hidden_size -> num_kv_heads * head_dim
        let v_proj = nn::LinearBuilder::new(args.hidden_size, num_key_value_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;

        // O projection: num_heads * head_dim -> hidden_size
        let o_proj = nn::LinearBuilder::new(num_attention_heads * head_dim, args.hidden_size)
            .bias(args.attention_bias)
            .build()?;

        // Q/K normalization
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        // RoPE with local base frequency (for sliding attention)
        let local_rope = nn::RopeBuilder::new(head_dim)
            .base(args.rope_local_base_freq)
            .build()
            .expect("Infallible");

        // RoPE with global base frequency (for full attention)
        let global_rope = nn::RopeBuilder::new(head_dim)
            .base(args.rope_theta)
            .build()
            .expect("Infallible");

        let scale = args.query_pre_attn_scalar as f32;

        Ok(Self {
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            scale,
            attention_type,
            q_proj: MaybeQuantized::new(q_proj),
            k_proj: MaybeQuantized::new(k_proj),
            v_proj: MaybeQuantized::new(v_proj),
            o_proj: MaybeQuantized::new(o_proj),
            q_norm,
            k_norm,
            local_rope,
            global_rope,
        })
    }
}

pub struct Gemma3AttentionInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for Gemma3AttentionInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        Gemma3AttentionInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<Gemma3AttentionInput<'_>> for Gemma3Attention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma3AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let Gemma3AttentionInput {
            hidden_states,
            attention_mask,
        } = input;

        let batch_size = hidden_states.shape()[0];
        let seq_length = hidden_states.shape()[1];

        // Project Q, K, V
        let mut queries = self.q_proj.forward(hidden_states)?;
        let mut keys = self.k_proj.forward(hidden_states)?;
        let mut values = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch_size, seq_length, num_heads, head_dim]
        queries = queries.reshape(&[
            batch_size,
            seq_length,
            self.num_attention_heads,
            self.head_dim,
        ])?;
        keys = keys.reshape(&[
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        ])?;
        values = values.reshape(&[
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        ])?;

        // Apply Q/K normalization
        queries = self.q_norm.forward(&queries)?;
        keys = self.k_norm.forward(&keys)?;

        // Transpose to [batch_size, num_heads, seq_length, head_dim]
        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        values = values.transpose_axes(&[0, 2, 1, 3])?;

        // Apply appropriate RoPE based on attention type
        let (queries, keys) = match &self.attention_type {
            AttentionType::FullAttention => {
                let q = self.global_rope.forward(&queries)?;
                let k = self.global_rope.forward(&keys)?;
                (q, k)
            }
            AttentionType::SlidingAttention { .. } => {
                let q = self.local_rope.forward(&queries)?;
                let k = self.local_rope.forward(&keys)?;
                (q, k)
            }
        };

        // Note: scaled_dot_product_attention automatically handles grouped query attention
        // No need to manually repeat keys/values to match the number of query heads

        // Compute attention with appropriate mask
        let context_layer = match &self.attention_type {
            AttentionType::FullAttention => {
                // For full attention, use the provided mask (if any)
                let mask_opt = attention_mask.map(|m| m.into());
                scaled_dot_product_attention(queries, &keys, &values, 1.0 / self.scale, mask_opt)?
            }
            AttentionType::SlidingAttention { window_size } => {
                // For sliding attention, create or combine with window mask
                let window_mask = create_bidirectional_window_mask(seq_length, *window_size)?;

                if let Some(user_mask) = attention_mask {
                    // Combine user mask with window mask using logical AND
                    let combined_mask = user_mask.logical_and(&window_mask)?;
                    scaled_dot_product_attention(
                        queries,
                        &keys,
                        &values,
                        1.0 / self.scale,
                        &combined_mask,
                    )?
                } else {
                    scaled_dot_product_attention(
                        queries,
                        &keys,
                        &values,
                        1.0 / self.scale,
                        &window_mask,
                    )?
                }
            }
        };

        // Transpose back to [batch_size, seq_length, num_heads, head_dim]
        let context_layer = context_layer.transpose_axes(&[0, 2, 1, 3])?;

        // Reshape to [batch_size, seq_length, num_heads * head_dim]
        let context_layer = context_layer.reshape(&[
            batch_size,
            seq_length,
            self.num_attention_heads * self.head_dim,
        ])?;

        // Output projection
        self.o_proj.forward(&context_layer)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
    }
}

// ============================================================================
// Gemma3MLP
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3MLP {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
}

impl Gemma3MLP {
    pub fn new(args: &ModelArgs) -> Result<Self, Gemma3Error> {
        let gate_proj = nn::LinearBuilder::new(args.hidden_size, args.intermediate_size)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(args.hidden_size, args.intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(args.intermediate_size, args.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::new(gate_proj),
            up_proj: MaybeQuantized::new(up_proj),
            down_proj: MaybeQuantized::new(down_proj),
        })
    }
}

impl Module<&Array> for Gemma3MLP {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, hidden_states: &Array) -> Result<Self::Output, Self::Error> {
        // gate_proj -> GELU -> multiply with up_proj -> down_proj
        let gate = self.gate_proj.forward(hidden_states)?;
        let gate = nn::gelu_approximate(&gate)?; // gelu_pytorch_tanh variant
        let up = self.up_proj.forward(hidden_states)?;
        let hidden_states = gate.multiply(&up)?;
        self.down_proj.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

// ============================================================================
// Gemma3Layer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3Layer {
    #[quantizable]
    #[param]
    self_attn: Gemma3Attention,

    #[quantizable]
    #[param]
    mlp: Gemma3MLP,

    #[param]
    input_layernorm: nn::RmsNorm,

    #[param]
    post_attention_layernorm: nn::RmsNorm,

    #[param]
    pre_feedforward_layernorm: nn::RmsNorm,

    #[param]
    post_feedforward_layernorm: nn::RmsNorm,
}

impl Gemma3Layer {
    pub fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, Gemma3Error> {
        // Determine attention type based on layer_types config
        let layer_type = args
            .layer_types
            .get(layer_idx as usize)
            .map(|s| s.as_str())
            .unwrap_or("sliding_attention");

        let attention_type = match layer_type {
            "full_attention" => AttentionType::FullAttention,
            _ => AttentionType::SlidingAttention {
                window_size: args.sliding_window,
            },
        };

        let self_attn = Gemma3Attention::new(args, attention_type)?;
        let mlp = Gemma3MLP::new(args)?;

        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let pre_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_feedforward_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }
}

pub struct Gemma3LayerInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for Gemma3LayerInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        Gemma3LayerInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<Gemma3LayerInput<'_>> for Gemma3Layer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma3LayerInput<'_>) -> Result<Self::Output, Self::Error> {
        let Gemma3LayerInput {
            hidden_states,
            attention_mask,
        } = input;

        // Attention block with multiple norms
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attention_input = Gemma3AttentionInput {
            hidden_states: &normed,
            attention_mask,
        };
        let attn_output = self.self_attn.forward(attention_input)?;
        let attn_output = self.post_attention_layernorm.forward(&attn_output)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // MLP block with multiple norms
        let normed = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        let mlp_output = self.post_feedforward_layernorm.forward(&mlp_output)?;
        hidden_states.add(&mlp_output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attn.training_mode(mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

// ============================================================================
// Gemma3Model
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3Model {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    layers: Vec<Gemma3Layer>,

    #[param]
    norm: nn::RmsNorm,
}

impl Gemma3Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Gemma3Error> {
        if args.vocab_size <= 0 {
            return Err(Gemma3Error::InvalidVocabSize(args.vocab_size));
        }
        if args.num_hidden_layers <= 0 {
            return Err(Gemma3Error::InvalidNumLayers(args.num_hidden_layers));
        }

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| Gemma3Layer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens: MaybeQuantized::new(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct Gemma3ModelInput<'a> {
    pub input_ids: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for Gemma3ModelInput<'a> {
    fn from(input_ids: &'a Array) -> Self {
        Gemma3ModelInput {
            input_ids,
            attention_mask: None,
        }
    }
}

impl Module<Gemma3ModelInput<'_>> for Gemma3Model {
    type Output = Array;
    type Error = Gemma3Error;

    fn forward(&mut self, input: Gemma3ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        use mlx_rs::transforms::eval;

        let Gemma3ModelInput {
            input_ids,
            attention_mask,
        } = input;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Pass through all layers
        // For very deep models (>20 layers), periodically evaluate to prevent
        // excessive graph growth. The documentation recommends graph sizes of
        // "a few tens to many thousands of operations" per evaluation.
        let eval_interval = if self.layers.len() > 20 {
            10
        } else {
            usize::MAX
        };

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_input = Gemma3LayerInput {
                hidden_states: &hidden_states,
                attention_mask,
            };
            hidden_states = layer.forward(layer_input)?;

            // Periodically evaluate for deep models to prevent memory bloat
            if (idx + 1) % eval_interval == 0 {
                eval([&hidden_states])?;
            }
        }

        // Final normalization
        self.norm.forward(&hidden_states).map_err(Into::into)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
        self.norm.training_mode(mode);
    }
}

// ============================================================================
// EmbeddingGemma (Top-level model with dense projections)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct EmbeddingGemma {
    #[quantizable]
    #[param]
    model: Gemma3Model,

    #[quantizable]
    #[param(rename = "dense.0")]
    dense_0: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param(rename = "dense.1")]
    dense_1: MaybeQuantized<nn::Linear>,
}

impl EmbeddingGemma {
    pub fn new(args: &ModelArgs) -> Result<Self, Gemma3Error> {
        let model = Gemma3Model::new(args)?;

        // Dense projection layers: 768 -> 3072 -> 768
        let dense_0 = nn::LinearBuilder::new(args.hidden_size, args.hidden_size * 4)
            .bias(false)
            .build()?;
        let dense_1 = nn::LinearBuilder::new(args.hidden_size * 4, args.hidden_size)
            .bias(false)
            .build()?;

        let mut model = Self {
            model,
            dense_0: MaybeQuantized::new(dense_0),
            dense_1: MaybeQuantized::new(dense_1),
        };

        // If quantization config is present, quantize the model
        if let Some(quant_config) = &args.quantization {
            model.quantize_model(quant_config.group_size, quant_config.bits)?;
        }

        Ok(model)
    }

    /// Quantize all quantizable layers in the model
    pub fn quantize_model(&mut self, group_size: i32, bits: i32) -> Result<(), Gemma3Error> {
        use mlx_rs::module::ModuleParameters;
        use mlx_rs::transforms::eval;

        // Quantize embed_tokens
        self.model.embed_tokens = self
            .model
            .embed_tokens
            .clone()
            .quantize_with(|m| nn::quantize(m, group_size, bits))?;

        // Quantize all layers
        for layer in &mut self.model.layers {
            // Quantize attention projections
            layer.self_attn.q_proj = layer
                .self_attn
                .q_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
            layer.self_attn.k_proj = layer
                .self_attn
                .k_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
            layer.self_attn.v_proj = layer
                .self_attn
                .v_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
            layer.self_attn.o_proj = layer
                .self_attn
                .o_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;

            // Quantize MLP projections
            layer.mlp.gate_proj = layer
                .mlp
                .gate_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
            layer.mlp.up_proj = layer
                .mlp
                .up_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
            layer.mlp.down_proj = layer
                .mlp
                .down_proj
                .clone()
                .quantize_with(|m| nn::quantize(m, group_size, bits))?;
        }

        // Quantize dense layers
        self.dense_0 = self
            .dense_0
            .clone()
            .quantize_with(|m| nn::quantize(m, group_size, bits))?;
        self.dense_1 = self
            .dense_1
            .clone()
            .quantize_with(|m| nn::quantize(m, group_size, bits))?;

        // Evaluate all parameters after quantization to prevent memory bloat
        // from unevaluated cloned arrays. This follows the pattern from the docs
        // where loading weights should be followed by evaluation.
        let params = self.parameters().flatten();
        let param_refs: Vec<&Array> = params.values().copied().collect();
        eval(param_refs)?;

        Ok(())
    }

    /// Mean pooling across the sequence dimension
    pub fn mean_pool(&self, hidden_states: &Array) -> Result<Array, Exception> {
        // hidden_states shape: [batch_size, seq_length, hidden_size]
        // Mean across dimension 1 (seq_length)
        hidden_states.mean_axes(&[1], None)
    }

    /// L2 normalization
    pub fn normalize(&self, embeddings: &Array) -> Result<Array, Exception> {
        // Compute L2 norm along the last dimension
        let axes = [-1];
        let l2_norm = norm_l2(embeddings, Some(&axes[..]), Some(true))?;

        // Normalize
        embeddings.divide(&l2_norm)
    }

    /// Generate embeddings from input tokens
    pub fn generate_embeddings(&mut self, input_ids: &Array) -> Result<Array, Gemma3Error> {
        use mlx_rs::transforms::eval;

        // Forward through transformer
        let model_input = Gemma3ModelInput {
            input_ids,
            attention_mask: None,
        };
        let hidden_states = self.model.forward(model_input)?;

        // Evaluate after transformer forward pass to prevent excessive graph growth
        // This is especially important for deep models with many layers
        eval([&hidden_states])?;

        // Mean pooling
        let pooled = self.mean_pool(&hidden_states)?;

        // Dense projections
        let embeddings = self.dense_0.forward(&pooled)?;
        let embeddings = self.dense_1.forward(&embeddings)?;

        // L2 normalization
        self.normalize(&embeddings).map_err(Into::into)
    }
}

pub struct EmbeddingGemmaInput<'a> {
    pub input_ids: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for EmbeddingGemmaInput<'a> {
    fn from(input_ids: &'a Array) -> Self {
        EmbeddingGemmaInput {
            input_ids,
            attention_mask: None,
        }
    }
}

impl Module<EmbeddingGemmaInput<'_>> for EmbeddingGemma {
    type Output = Array;
    type Error = Gemma3Error;

    fn forward(&mut self, input: EmbeddingGemmaInput<'_>) -> Result<Self::Output, Self::Error> {
        self.generate_embeddings(input.input_ids)
    }

    fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
        self.dense_0.training_mode(mode);
        self.dense_1.training_mode(mode);
    }
}
