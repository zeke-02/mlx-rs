use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub intermediate_size: i32,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: i32,
    pub type_vocab_size: i32,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    #[serde(default)]
    pub use_cache: bool,
    #[serde(default)]
    pub pad_token_id: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum BertError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

// ============================================================================
// BertEmbeddings
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertEmbeddings {
    hidden_size: i32,
    max_position_embeddings: i32,

    #[quantizable]
    #[param]
    word_embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    position_embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    token_type_embeddings: MaybeQuantized<nn::Embedding>,

    #[param(rename = "LayerNorm")]
    layer_norm: nn::LayerNorm,
}

impl BertEmbeddings {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let word_embeddings = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let position_embeddings =
            nn::Embedding::new(args.max_position_embeddings, args.hidden_size)?;
        let token_type_embeddings = nn::Embedding::new(args.type_vocab_size, args.hidden_size)?;
        let layer_norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            hidden_size: args.hidden_size,
            max_position_embeddings: args.max_position_embeddings,
            word_embeddings: MaybeQuantized::new(word_embeddings),
            position_embeddings: MaybeQuantized::new(position_embeddings),
            token_type_embeddings: MaybeQuantized::new(token_type_embeddings),
            layer_norm,
        })
    }
}

pub struct BertEmbeddingsInput<'a> {
    pub input_ids: &'a Array,
    pub token_type_ids: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertEmbeddingsInput<'a> {
    fn from(input_ids: &'a Array) -> Self {
        BertEmbeddingsInput {
            input_ids,
            token_type_ids: None,
        }
    }
}

impl Module<BertEmbeddingsInput<'_>> for BertEmbeddings {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertEmbeddingsInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertEmbeddingsInput {
            input_ids,
            token_type_ids,
        } = input;

        let seq_length = input_ids.shape()[1];

        // Create position_ids [0, 1, 2, ..., seq_length-1]
        let position_ids = Array::arange::<_, i32>(None, seq_length, None)?;

        // Get embeddings
        let word_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Sum word and position embeddings
        let mut embeddings = word_embeds.add(&position_embeds)?;

        // Add token type embeddings if provided
        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeds = self.token_type_embeddings.forward(token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeds)?;
        } else {
            // Default to all zeros (token type 0)
            let batch_size = input_ids.shape()[0];
            let token_type_ids_default = Array::zeros::<i32>(&[batch_size, seq_length])?;
            let token_type_embeds = self
                .token_type_embeddings
                .forward(&token_type_ids_default)?;
            embeddings = embeddings.add(&token_type_embeds)?;
        }

        // Apply layer norm
        self.layer_norm.forward(&embeddings)
    }

    fn training_mode(&mut self, mode: bool) {
        self.word_embeddings.training_mode(mode);
        self.position_embeddings.training_mode(mode);
        self.token_type_embeddings.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

// ============================================================================
// BertSelfAttention
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertSelfAttention {
    num_attention_heads: i32,
    attention_head_size: i32,
    all_head_size: i32,

    #[quantizable]
    #[param(rename = "self")]
    self_attention: BertSelfAttentionQKV,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertSelfAttentionQKV {
    #[quantizable]
    #[param]
    query: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    key: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    value: MaybeQuantized<nn::Linear>,
}

impl BertSelfAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let num_attention_heads = args.num_attention_heads;
        let attention_head_size = args.hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let query = nn::LinearBuilder::new(args.hidden_size, all_head_size)
            .bias(true)
            .build()?;
        let key = nn::LinearBuilder::new(args.hidden_size, all_head_size)
            .bias(true)
            .build()?;
        let value = nn::LinearBuilder::new(args.hidden_size, all_head_size)
            .bias(true)
            .build()?;

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            self_attention: BertSelfAttentionQKV {
                query: MaybeQuantized::new(query),
                key: MaybeQuantized::new(key),
                value: MaybeQuantized::new(value),
            },
        })
    }

    fn transpose_for_scores(&self, x: &Array) -> Result<Array, Exception> {
        // x shape: [batch_size, seq_length, all_head_size]
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];

        // Reshape to [batch_size, seq_length, num_attention_heads, attention_head_size]
        let x = x.reshape(&[
            batch_size,
            seq_length,
            self.num_attention_heads,
            self.attention_head_size,
        ])?;

        // Transpose to [batch_size, num_attention_heads, seq_length, attention_head_size]
        x.transpose_axes(&[0, 2, 1, 3])
    }
}

pub struct BertSelfAttentionInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertSelfAttentionInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        BertSelfAttentionInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<BertSelfAttentionInput<'_>> for BertSelfAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertSelfAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertSelfAttentionInput {
            hidden_states,
            attention_mask,
        } = input;

        // Compute Q, K, V
        let query_layer = self.self_attention.query.forward(hidden_states)?;
        let key_layer = self.self_attention.key.forward(hidden_states)?;
        let value_layer = self.self_attention.value.forward(hidden_states)?;

        // Transpose for attention computation
        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // Compute attention
        let scale = (self.attention_head_size as f32).sqrt();

        // Note: BERT typically doesn't use causal masking for encoding
        // If attention_mask is provided, it would need to be converted to the proper format
        // For now, we pass the mask directly if provided
        let mask_opt = attention_mask.map(ScaledDotProductAttentionMask::from);
        let context_layer = scaled_dot_product_attention(
            query_layer,
            &key_layer,
            &value_layer,
            1.0 / scale,
            mask_opt,
        )?;

        // Transpose back to [batch_size, seq_length, num_attention_heads, attention_head_size]
        let context_layer = context_layer.transpose_axes(&[0, 2, 1, 3])?;

        // Reshape to [batch_size, seq_length, all_head_size]
        let batch_size = context_layer.shape()[0];
        let seq_length = context_layer.shape()[1];
        context_layer.reshape(&[batch_size, seq_length, self.all_head_size])
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attention.query.training_mode(mode);
        self.self_attention.key.training_mode(mode);
        self.self_attention.value.training_mode(mode);
    }
}

// ============================================================================
// BertSelfOutput
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertSelfOutput {
    #[quantizable]
    #[param]
    dense: MaybeQuantized<nn::Linear>,

    #[param(rename = "LayerNorm")]
    layer_norm: nn::LayerNorm,
}

impl BertSelfOutput {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let dense = nn::LinearBuilder::new(args.hidden_size, args.hidden_size)
            .bias(true)
            .build()?;
        let layer_norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            dense: MaybeQuantized::new(dense),
            layer_norm,
        })
    }
}

pub struct BertSelfOutputInput<'a> {
    pub hidden_states: &'a Array,
    pub input_tensor: &'a Array,
}

impl Module<BertSelfOutputInput<'_>> for BertSelfOutput {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertSelfOutputInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertSelfOutputInput {
            hidden_states,
            input_tensor,
        } = input;

        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.add(input_tensor)?;
        self.layer_norm.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

// ============================================================================
// BertAttention (combines self-attention and output)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertAttention {
    #[quantizable]
    #[param(rename = "self")]
    self_attention: BertSelfAttention,

    #[quantizable]
    #[param]
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        Ok(Self {
            self_attention: BertSelfAttention::new(args)?,
            output: BertSelfOutput::new(args)?,
        })
    }
}

pub struct BertAttentionInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertAttentionInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        BertAttentionInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<BertAttentionInput<'_>> for BertAttention {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertAttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertAttentionInput {
            hidden_states,
            attention_mask,
        } = input;

        let self_attention_input = BertSelfAttentionInput {
            hidden_states,
            attention_mask,
        };
        let self_outputs = self.self_attention.forward(self_attention_input)?;

        let output_input = BertSelfOutputInput {
            hidden_states: &self_outputs,
            input_tensor: hidden_states,
        };
        self.output.forward(output_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.self_attention.training_mode(mode);
        self.output.training_mode(mode);
    }
}

// ============================================================================
// BertIntermediate
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertIntermediate {
    #[quantizable]
    #[param]
    dense: MaybeQuantized<nn::Linear>,
}

impl BertIntermediate {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let dense = nn::LinearBuilder::new(args.hidden_size, args.intermediate_size)
            .bias(true)
            .build()?;

        Ok(Self {
            dense: MaybeQuantized::new(dense),
        })
    }
}

impl Module<&Array> for BertIntermediate {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, hidden_states: &Array) -> Result<Self::Output, Self::Error> {
        let hidden_states = self.dense.forward(hidden_states)?;
        nn::gelu(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
    }
}

// ============================================================================
// BertOutput
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertOutput {
    #[quantizable]
    #[param]
    dense: MaybeQuantized<nn::Linear>,

    #[param(rename = "LayerNorm")]
    layer_norm: nn::LayerNorm,
}

impl BertOutput {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let dense = nn::LinearBuilder::new(args.intermediate_size, args.hidden_size)
            .bias(true)
            .build()?;
        let layer_norm = nn::LayerNormBuilder::new(args.hidden_size)
            .eps(args.layer_norm_eps)
            .affine(true)
            .build()?;

        Ok(Self {
            dense: MaybeQuantized::new(dense),
            layer_norm,
        })
    }
}

pub struct BertOutputInput<'a> {
    pub hidden_states: &'a Array,
    pub input_tensor: &'a Array,
}

impl Module<BertOutputInput<'_>> for BertOutput {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertOutputInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertOutputInput {
            hidden_states,
            input_tensor,
        } = input;

        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.add(input_tensor)?;
        self.layer_norm.forward(&hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.dense.training_mode(mode);
        self.layer_norm.training_mode(mode);
    }
}

// ============================================================================
// BertLayer (encoder layer)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertLayer {
    #[quantizable]
    #[param]
    attention: BertAttention,

    #[quantizable]
    #[param]
    intermediate: BertIntermediate,

    #[quantizable]
    #[param]
    output: BertOutput,
}

impl BertLayer {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        Ok(Self {
            attention: BertAttention::new(args)?,
            intermediate: BertIntermediate::new(args)?,
            output: BertOutput::new(args)?,
        })
    }
}

pub struct BertLayerInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertLayerInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        BertLayerInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<BertLayerInput<'_>> for BertLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertLayerInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertLayerInput {
            hidden_states,
            attention_mask,
        } = input;

        // Self-attention
        let attention_input = BertAttentionInput {
            hidden_states,
            attention_mask,
        };
        let attention_output = self.attention.forward(attention_input)?;

        // Feed-forward
        let intermediate_output = self.intermediate.forward(&attention_output)?;

        let output_input = BertOutputInput {
            hidden_states: &intermediate_output,
            input_tensor: &attention_output,
        };
        self.output.forward(output_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
        self.intermediate.training_mode(mode);
        self.output.training_mode(mode);
    }
}

// ============================================================================
// BertEncoder
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct BertEncoder {
    #[quantizable]
    #[param]
    layer: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        let layers = (0..args.num_hidden_layers)
            .map(|_| BertLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { layer: layers })
    }
}

pub struct BertEncoderInput<'a> {
    pub hidden_states: &'a Array,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertEncoderInput<'a> {
    fn from(hidden_states: &'a Array) -> Self {
        BertEncoderInput {
            hidden_states,
            attention_mask: None,
        }
    }
}

impl Module<BertEncoderInput<'_>> for BertEncoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: BertEncoderInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertEncoderInput {
            hidden_states,
            attention_mask,
        } = input;

        let mut hidden_states = hidden_states.clone();

        for layer in &mut self.layer {
            let layer_input = BertLayerInput {
                hidden_states: &hidden_states,
                attention_mask,
            };
            hidden_states = layer.forward(layer_input)?;
        }

        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.layer
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
    }
}

// ============================================================================
// Bert (main model)
// ============================================================================

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Bert {
    #[quantizable]
    #[param]
    embeddings: BertEmbeddings,

    #[quantizable]
    #[param]
    encoder: BertEncoder,
}

impl Bert {
    pub fn new(args: &ModelArgs) -> Result<Self, BertError> {
        if args.vocab_size <= 0 {
            return Err(BertError::InvalidVocabSize(args.vocab_size));
        }

        Ok(Self {
            embeddings: BertEmbeddings::new(args)?,
            encoder: BertEncoder::new(args)?,
        })
    }
}

pub struct BertInput<'a> {
    pub input_ids: &'a Array,
    pub token_type_ids: Option<&'a Array>,
    pub attention_mask: Option<&'a Array>,
}

impl<'a> From<&'a Array> for BertInput<'a> {
    fn from(input_ids: &'a Array) -> Self {
        BertInput {
            input_ids,
            token_type_ids: None,
            attention_mask: None,
        }
    }
}

impl Module<BertInput<'_>> for Bert {
    type Output = Array;
    type Error = BertError;

    fn forward(&mut self, input: BertInput<'_>) -> Result<Self::Output, Self::Error> {
        let BertInput {
            input_ids,
            token_type_ids,
            attention_mask,
        } = input;

        // Get embeddings
        let embeddings_input = BertEmbeddingsInput {
            input_ids,
            token_type_ids,
        };
        let embedding_output = self.embeddings.forward(embeddings_input)?;

        // Pass through encoder
        let encoder_input = BertEncoderInput {
            hidden_states: &embedding_output,
            attention_mask,
        };
        let sequence_output = self.encoder.forward(encoder_input)?;

        Ok(sequence_output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embeddings.training_mode(mode);
        self.encoder.training_mode(mode);
    }
}
