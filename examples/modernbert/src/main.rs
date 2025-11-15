use mlx_rs::{
    linalg::norm_l2,
    module::{Module, ModuleParametersExt},
    ops::indexing::IndexOp,
    transforms::eval,
    Array,
};
mod model;
use std::path::PathBuf;
use tokenizers::Tokenizer;
type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use model::{ModelArgs, ModernBert, ModernBertInput};

fn get_tokenizer(model_dir: &PathBuf) -> Result<Tokenizer> {
    let tokenizer_filename = model_dir.join("tokenizer.json");
    if !tokenizer_filename.exists() {
        return Err(format!("tokenizer.json not found in {:?}", model_dir).into());
    }
    let t = Tokenizer::from_file(tokenizer_filename)?;
    Ok(t)
}

fn load_model(model_dir: &PathBuf) -> Result<ModernBert> {
    // Verify required files exist
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("config.json not found in {:?}", model_dir).into());
    }

    let file = std::fs::File::open(config_path)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    // Check if we have a single safetensors file
    let single_safetensors = model_dir.join("model.safetensors");
    if !single_safetensors.exists() {
        return Err(format!("model.safetensors not found in {:?}", model_dir).into());
    }

    // Create the model
    let mut model = ModernBert::new(&model_args)?;

    // Load weights from safetensors
    model.load_safetensors(&single_safetensors)?;

    Ok(model)
}

fn extract_embeddings(model: &mut ModernBert, input_ids: &Array) -> Result<Array> {
    // Forward pass through the model
    let bert_input = ModernBertInput::from(input_ids);
    let output = model.forward(bert_input)?;

    // Use the pooler output (CLS token embedding)
    Ok(output.pooler_output)
}

fn main() -> Result<()> {
    mlx_rs::random::seed(42)?;

    // Configuration - modify these variables to test with your model
    let model_dir = PathBuf::from(
        "/Users/zekieldee/Desktop/code/embedding-models/nomicai-modernbert-embed-base-8bit",
    ); // Change this to your model directory path
    let input_text = "Hello, this is a test sentence for ModernBERT embeddings."; // Change this to your test text

    println!("[INFO] Using model directory: {:?}", model_dir);
    println!("[INFO] Loading tokenizer...");
    let tokenizer = get_tokenizer(&model_dir)?;

    println!("[INFO] Loading model...");
    let mut model = load_model(&model_dir)?;

    // Text to encode
    println!("\n[INFO] Input text: {}", input_text);

    // Tokenize the input
    let encoding = tokenizer.encode(input_text, true)?;
    let tokens = encoding.get_ids();
    println!("[INFO] Tokens: {:?}", tokens);

    // Convert to MLX array with batch dimension
    let input_ids = Array::from_slice(tokens, &[1, tokens.len() as i32]);

    // Extract embeddings
    println!("[INFO] Extracting embeddings...");
    let embeddings = extract_embeddings(&mut model, &input_ids)?;

    // Evaluate the computation
    eval([&embeddings])?;

    // Get the embedding as a vector
    let embedding_shape = embeddings.shape();
    println!("[INFO] Embedding shape: {:?}", embedding_shape);

    // Compute L2 norm
    let axes = [-1];
    let l2_norm = norm_l2(&embeddings, Some(&axes[..]), Some(true))?;
    eval([&l2_norm])?;
    println!("[INFO] L2 norm: {:.6}", l2_norm.item::<f32>());

    // Normalize the embedding
    let normalized_embedding = embeddings.divide(&l2_norm)?;
    eval([&normalized_embedding])?;

    // Print first few dimensions
    println!("\n[INFO] First 10 dimensions of normalized embedding:");
    let num_dims = if embedding_shape.len() == 2 {
        10.min(embedding_shape[1])
    } else {
        10.min(embedding_shape[0])
    };

    for i in 0..num_dims {
        let val = if embedding_shape.len() == 2 {
            normalized_embedding.index((0, i)).item::<f32>()
        } else {
            normalized_embedding.index(i).item::<f32>()
        };
        println!("  dim[{}]: {:.6}", i, val);
    }

    println!("\n[SUCCESS] ModernBERT model loaded and inference completed successfully!");

    Ok(())
}
