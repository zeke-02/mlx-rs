# Bert Model

# Safetensor architecture overview

embeddings(5)

embeddings.position_embeddings.weight [512, 384]

F32
embeddings.token_type_embeddings.weight [2, 384]

F32
embeddings.word_embeddings.weight [30 522, 384]

F32
embeddings.LayerNorm(2)

embeddings.LayerNorm.bias [384]

F32
embeddings.LayerNorm.weight [384]

F32
embeddings.position_ids [1, 512]

I64
encoder.layer(6)

encoder.layer.0(3)

encoder.layer.0.attention(2)

encoder.layer.0.attention.self(3)

encoder.layer.0.attention.self.key(2)

encoder.layer.0.attention.self.key.bias [384]

F32
encoder.layer.0.attention.self.key.weight [384, 384]

F32
encoder.layer.0.attention.self.query(2)

encoder.layer.0.attention.self.query.bias [384]

F32
encoder.layer.0.attention.self.query.weight [384, 384]

F32
encoder.layer.0.attention.self.value(2)

encoder.layer.0.attention.self.value.bias [384]

F32
encoder.layer.0.attention.self.value.weight [384, 384]

F32
encoder.layer.0.attention.output(2)

encoder.layer.0.attention.output.dense(2)

encoder.layer.0.attention.output.dense.bias [384]

F32
encoder.layer.0.attention.output.dense.weight [384, 384]

F32
encoder.layer.0.attention.output.LayerNorm(2)

encoder.layer.0.attention.output.LayerNorm.bias [384]

F32
encoder.layer.0.attention.output.LayerNorm.weight [384]

F32
encoder.layer.0.intermediate.dense(2)

encoder.layer.0.intermediate.dense.bias [1 536]

F32
encoder.layer.0.intermediate.dense.weight [1 536, 384]

F32
encoder.layer.0.output(2)

encoder.layer.0.output.dense(2)

encoder.layer.0.output.dense.bias [384]

F32
encoder.layer.0.output.dense.weight [384, 1 536]

F32
encoder.layer.0.output.LayerNorm(2)

encoder.layer.0.output.LayerNorm.bias [384]

F32
encoder.layer.0.output.LayerNorm.weight [384]

F32
encoder.layer.1(3)

encoder.layer.2(3)

encoder.layer.3(3)

encoder.layer.4(3)

encoder.layer.5(3)

pooler.dense(2)

pooler.dense.bias [384]

F32
pooler.dense.weight
