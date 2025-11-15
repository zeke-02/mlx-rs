# Embedding Gemma 300m mlx

# Safetensors layout

dense(2)

dense.0(3)

dense.0.biases [3 072, 12]

F16
dense.0.scales [3 072, 12]

F16
dense.0.weight [3 072, 192]

U32
dense.1(3)

model(3)

model.embed_tokens(3)

model.embed_tokens.biases [262 144, 12]

F16
model.embed_tokens.scales [262 144, 12]

F16
model.embed_tokens.weight [262 144, 192]

U32
model.layers(24)

model.layers.0(6)

model.layers.0.input_layernorm.weight [768]

F16
model.layers.0.mlp(3)

model.layers.0.mlp.down_proj(3)

model.layers.0.mlp.down_proj.biases [768, 18]

F16
model.layers.0.mlp.down_proj.scales [768, 18]

F16
model.layers.0.mlp.down_proj.weight [768, 288]

U32
model.layers.0.mlp.gate_proj(3)

model.layers.0.mlp.gate_proj.biases [1 152, 12]

F16
model.layers.0.mlp.gate_proj.scales [1 152, 12]

F16
model.layers.0.mlp.gate_proj.weight [1 152, 192]

U32
model.layers.0.mlp.up_proj(3)

model.layers.0.mlp.up_proj.biases [1 152, 12]

F16
model.layers.0.mlp.up_proj.scales [1 152, 12]

F16
model.layers.0.mlp.up_proj.weight [1 152, 192]

U32
model.layers.0.post_attention_layernorm.weight [768]

F16
model.layers.0.post_feedforward_layernorm.weight [768]

F16
model.layers.0.pre_feedforward_layernorm.weight [768]

F16
model.layers.0.self_attn(6)

model.layers.0.self_attn.k_norm.weight [256]

F16
model.layers.0.self_attn.k_proj(3)

model.layers.0.self_attn.k_proj.biases [256, 12]

F16
model.layers.0.self_attn.k_proj.scales [256, 12]

F16
model.layers.0.self_attn.k_proj.weight [256, 192]

U32
model.layers.0.self_attn.o_proj(3)

model.layers.0.self_attn.o_proj.biases [768, 12]

F16
model.layers.0.self_attn.o_proj.scales [768, 12]

F16
model.layers.0.self_attn.o_proj.weight [768, 192]

U32
model.layers.0.self_attn.q_norm.weight [256]

F16
model.layers.0.self_attn.q_proj(3)

model.layers.0.self_attn.q_proj.biases [768, 12]

F16
model.layers.0.self_attn.q_proj.scales [768, 12]

F16
model.layers.0.self_attn.q_proj.weight [768, 192]

U32
model.layers.0.self_attn.v_proj(3)

model.layers.0.self_attn.v_proj.biases [256, 12]

F16
model.layers.0.self_attn.v_proj.scales [256, 12]

F16
model.layers.0.self_attn.v_proj.weight [256, 192]

U32
model.layers.1(6)

model.layers.2(6)

model.layers.3(6)

model.layers.4(6)

model.layers.5(6)

model.layers.6(6)

model.layers.7(6)

model.layers.8(6)

model.layers.9(6)

model.layers.10(6)

model.layers.11(6)

model.layers.12(6)

model.layers.13(6)

model.layers.14(6)

model.layers.15(6)

model.layers.16(6)

model.layers.17(6)

model.layers.18(6)

model.layers.19(6)

model.layers.20(6)

model.layers.21(6)

model.layers.22(6)

model.layers.23(6)

model.norm.weight [768]

F16

# Architecture of Embedding Gemma:

How Embeddings Are Formed

You can use EmbeddingGemma to generate embeddings using frameworks such as Sentence Transformers. Given an input sequence of text, EmbeddingGemma processes it through a series of carefully designed steps to produce a concise vector representation.

SentenceTransformer(
(0): Transformer({'max_seq_length': 2048, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})
(1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
(2): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
(3): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
(4): Normalize()
)

Python

(0): Transformer

An input sequence passes through this encoder-only transformer model. This transformer utilizes bidirectional attention to understand the meaning of each token in the provided context, producing a sequence of 768-dimensional vectors, one for each token in your input sequence.

(1): Pooling

The output of the transformer is a sequence of token embeddings. The pooling layer’s job is to convert this variable-length sequence into a single, fixed-size embedding for the entire input. EmbeddingGemma is using a pooling strategy called “Mean Pooling”. This is the most common approach, where the average of all token embeddings is calculated.

(2): Dense

Next, we apply a linear projection to scale the embedding (768) up to a larger embedding dimension (3072).

(3): Dense

Then we apply another linear projection to scale the learned 3072-dimensional embedding to the final target dimension (768).

(4): Normalize

Finally, we apply Euclidean normalization, enabling efficient similarity comparisons. This is a simpler and cheaper operation compared to the more complex RMSNorm that you might recall from other Gemma models
