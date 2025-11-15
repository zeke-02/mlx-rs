use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::scaled_dot_product_attention,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::IndexOp,
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
    pub intermediate_size: i32,
    pub hidden_activation: String,
    pub layer_norm_eps: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub classifier_bias: bool,
    #[serde(default)]
    pub decoder_bias: bool,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default = "default_local_rope_theta")]
    pub local_rope_theta: f32,
    #[serde(default = "default_global_rope_theta")]
    pub global_rope_theta: f32,
    #[serde(default)]
    pub global_attn_every_n_layers: i32,
    #[serde(default)]
    pub local_attention: i32,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_local_rope_theta() -> f32 {
    10000.0
}

fn default_global_rope_theta() -> f32 {
    160000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum ModernBertError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Creates a bidirectional sliding window attention mask.
///
/// Each position can attend to tokens within ±(window_size/2) positions.
/// Returns a boolean mask of shape [seq_length, seq_length] where True means
/// the attention is allowed.
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

// ============================================================================
// ModernBertEmbeddings
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBertEmbeddings {
    #[quantizable]
    #[param(rename = "tok_embeddings")]
    tok_embeddings: MaybeQuantized<nn::Embedding>,

    #[param]
    norm: nn::LayerNorm,
}

impl ModernBertEmbeddings {
    pub fn new(args: &ModelArgs) -> Result<Self, ModernBertError> {
        let tok_embeddings = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            tok_embeddings: MaybeQuantized::new(tok_embeddings),
            norm,
        })
    }
}

impl Module<&Array> for ModernBertEmbeddings {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input_ids: &Array) -> Result<Self::Output, Self::Error> {
        let embeddings = self.tok_embeddings.forward(input_ids)?;
        self.norm.forward(&embeddings)
    }

    fn training_mode(&mut self, mode: bool) {
        self.tok_embeddings.training_mode(mode);
        self.norm.training_mode(mode);
    }
}

// ============================================================================
// ModernBertAttention
// ============================================================================

/// Attention type for ModernBERT layers
#[derive(Debug, Clone)]
pub enum AttentionType {
    /// Global attention - attends to all tokens in the sequence
    Global,
    /// Local attention - attends only to tokens within a sliding window
    Local { window_size: i32 },
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBertAttention {
    num_attention_heads: i32,
    attention_head_size: i32,
    all_head_size: i32,
    scale: f32,
    attention_type: AttentionType,

    #[quantizable]
    #[param(rename = "Wqkv")]
    wqkv: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param(rename = "Wo")]
    wo: MaybeQuantized<nn::Linear>,

    local_rope: nn::Rope,
    global_rope: nn::Rope,
}

impl ModernBertAttention {
    pub fn new(args: &ModelArgs, attention_type: AttentionType) -> Result<Self, ModernBertError> {
        let num_attention_heads = args.num_attention_heads;
        let attention_head_size = args.hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        // Fused QKV projection
        let wqkv = nn::LinearBuilder::new(args.hidden_size, 3 * all_head_size)
            .bias(args.attention_bias)
            .build()?;

        // Output projection
        let wo = nn::LinearBuilder::new(all_head_size, args.hidden_size)
            .bias(args.attention_bias)
            .build()?;

        // RoPE with local theta
        let local_rope = nn::RopeBuilder::new(attention_head_size)
            .base(args.local_rope_theta)
            .build()
            .expect("Infallible");

        // RoPE with global theta
        let global_rope = nn::RopeBuilder::new(attention_head_size)
            .base(args.global_rope_theta)
            .build()
            .expect("Infallible");

        let scale = (attention_head_size as f32).sqrt();

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            scale,
            attention_type,
            wqkv: MaybeQuantized::new(wqkv),
            wo: MaybeQuantized::new(wo),
            local_rope,
            global_rope,
        })
    }
}

pub struct ModernBertAttentionInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for ModernBertAttentionInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        ModernBertAttentionInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<ModernBertAttentionInput<'_>> for ModernBertAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: ModernBertAttentionInput<'_>,
    ) -> Result<Self::Output, Self::Error> {
        let ModernBertAttentionInput {
            hidden_states,
            attention_mask,
        } = input;

        let batch_size = hidden_states.shape()[0];
        let seq_length = hidden_states.shape()[1];

        // Fused QKV projection
        let qkv = self.wqkv.forward(hidden_states)?;

        // Reshape to [batch_size, seq_length, 3, num_heads, head_size]
        let qkv = qkv.reshape(&[
            batch_size,
            seq_length,
            3,
            self.num_attention_heads,
            self.attention_head_size,
        ])?;

        // Split into Q, K, V and transpose to [batch_size, num_heads, seq_length, head_size]
        let queries = qkv
            .index((.., .., 0, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = qkv
            .index((.., .., 1, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = qkv
            .index((.., .., 2, .., ..))
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply appropriate RoPE based on attention type
        let (queries, keys) = match &self.attention_type {
            AttentionType::Global => {
                let q = self.global_rope.forward(&queries)?;
                let k = self.global_rope.forward(&keys)?;
                (q, k)
            }
            AttentionType::Local { .. } => {
                let q = self.local_rope.forward(&queries)?;
                let k = self.local_rope.forward(&keys)?;
                (q, k)
            }
        };

        // Compute attention with appropriate mask
        let context_layer = match &self.attention_type {
            AttentionType::Global => {
                // For global attention, use the provided mask (if any)
                let mask_opt = attention_mask.map(|m| m.into());
                scaled_dot_product_attention(queries, &keys, &values, 1.0 / self.scale, mask_opt)?
            }
            AttentionType::Local { window_size } => {
                // For local attention, create or combine with window mask
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

        // Transpose back to [batch_size, seq_length, num_heads, head_size]
        let context_layer = context_layer.transpose_axes(&[0, 2, 1, 3])?;

        // Reshape to [batch_size, seq_length, all_head_size]
        let context_layer = context_layer.reshape(&[batch_size, seq_length, self.all_head_size])?;

        // Output projection
        self.wo.forward(&context_layer)
    }

    fn training_mode(&mut self, mode: bool) {
        self.wqkv.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

// ============================================================================
// ModernBertMLP
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBertMLP {
    #[quantizable]
    #[param(rename = "Wi")]
    wi: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param(rename = "Wo")]
    wo: MaybeQuantized<nn::Linear>,
}

impl ModernBertMLP {
    pub fn new(args: &ModelArgs) -> Result<Self, ModernBertError> {
        let wi = nn::LinearBuilder::new(args.hidden_size, args.intermediate_size)
            .bias(args.mlp_bias)
            .build()?;

        let wo = nn::LinearBuilder::new(args.intermediate_size, args.hidden_size)
            .bias(args.mlp_bias)
            .build()?;

        Ok(Self {
            wi: MaybeQuantized::new(wi),
            wo: MaybeQuantized::new(wo),
        })
    }
}

impl Module<&Array> for ModernBertMLP {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, hidden_states: &Array) -> Result<Self::Output, Self::Error> {
        let hidden_states = self.wi.forward(hidden_states)?;
        let hidden_states = nn::gelu(&hidden_states)?;
        self.wo.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.wi.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

// ============================================================================
// ModernBertLayer
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBertLayer {
    #[quantizable]
    #[param]
    attn: ModernBertAttention,

    #[quantizable]
    #[param]
    mlp: ModernBertMLP,

    #[param]
    mlp_norm: nn::LayerNorm,
}

impl ModernBertLayer {
    pub fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, ModernBertError> {
        // Determine attention type based on layer index
        // Layer indices are 0-based, but the config uses 1-based counting
        // Every Nth layer (1, 4, 7, ...) uses global attention if global_attn_every_n_layers = 3
        let attention_type = if args.global_attn_every_n_layers > 0
            && (layer_idx + 1) % args.global_attn_every_n_layers == 0
        {
            AttentionType::Global
        } else {
            AttentionType::Local {
                window_size: args.local_attention,
            }
        };

        let attn = ModernBertAttention::new(args, attention_type)?;
        let mlp = ModernBertMLP::new(args)?;
        let mlp_norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            attn,
            mlp,
            mlp_norm,
        })
    }
}

pub struct ModernBertLayerInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for ModernBertLayerInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        ModernBertLayerInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<ModernBertLayerInput<'_>> for ModernBertLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModernBertLayerInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModernBertLayerInput {
            hidden_states,
            attention_mask,
        } = input;

        // Attention with residual (no pre-norm for attention in ModernBERT)
        let attention_input = ModernBertAttentionInput {
            hidden_states,
            attention_mask,
        };
        let attention_output = self.attn.forward(attention_input)?;
        let hidden_states = hidden_states.add(&attention_output)?;

        // MLP with pre-norm and residual
        let normed = self.mlp_norm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.attn.training_mode(mode);
        self.mlp.training_mode(mode);
        self.mlp_norm.training_mode(mode);
    }
}

// ============================================================================
// ModernBertEncoder
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBertEncoder {
    #[quantizable]
    #[param]
    layers: Vec<ModernBertLayer>,
}

impl ModernBertEncoder {
    pub fn new(args: &ModelArgs) -> Result<Self, ModernBertError> {
        let layers = (0..args.num_hidden_layers)
            .map(|layer_idx| ModernBertLayer::new(args, layer_idx))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { layers })
    }
}

pub struct ModernBertEncoderInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for ModernBertEncoderInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        ModernBertEncoderInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<ModernBertEncoderInput<'_>> for ModernBertEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModernBertEncoderInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModernBertEncoderInput {
            hidden_states,
            attention_mask,
        } = input;

        let mut hidden_states = hidden_states.clone();

        for layer in &mut self.layers {
            let layer_input = ModernBertLayerInput {
                hidden_states: &hidden_states,
                attention_mask,
            };
            hidden_states = layer.forward(layer_input)?;
        }

        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
    }
}

// ============================================================================
// ModernBert (Main Model)
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModernBertOutput {
    pub last_hidden_state: Array,
    pub pooler_output: Array,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct ModernBert {
    #[quantizable]
    #[param]
    embeddings: ModernBertEmbeddings,

    #[quantizable]
    #[param]
    layers: ModernBertEncoder,

    #[param]
    final_norm: nn::LayerNorm,
}

impl ModernBert {
    pub fn new(args: &ModelArgs) -> Result<Self, ModernBertError> {
        if args.vocab_size <= 0 {
            return Err(ModernBertError::InvalidVocabSize(args.vocab_size));
        }

        let embeddings = ModernBertEmbeddings::new(args)?;
        let layers = ModernBertEncoder::new(args)?;
        let final_norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            embeddings,
            layers,
            final_norm,
        })
    }
}

pub struct ModernBertInput<'a> {
    pub input_ids: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for ModernBertInput<'a> {
    fn from(input_ids: &'a Array) -> Self {
        ModernBertInput {
            input_ids,
            attention_mask: None,
        }
    }
}

impl Module<ModernBertInput<'_>> for ModernBert {
    type Output = ModernBertOutput;
    type Error = ModernBertError;

    fn forward(&mut self, input: ModernBertInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModernBertInput {
            input_ids,
            attention_mask,
        } = input;

        // Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Pass through encoder
        let encoder_input = ModernBertEncoderInput {
            hidden_states: &hidden_states,
            attention_mask,
        };
        let hidden_states = self.layers.forward(encoder_input)?;

        // Apply final layer norm
        let last_hidden_state = self.final_norm.forward(&hidden_states)?;

        // Extract CLS token (first token) as pooler output
        let pooler_output = last_hidden_state.index((.., 0, ..));

        Ok(ModernBertOutput {
            last_hidden_state,
            pooler_output,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        self.layers.training_mode(mode);
        self.final_norm.training_mode(mode);
    }
}
