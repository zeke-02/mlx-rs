use mlx_rs::{module::ModuleParametersExt, transforms::eval, Array};
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod model;
use model::{EmbeddingGemma, ModelArgs};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use clap::Parser;

#[derive(Parser)]
#[command(about = "EmbeddingGemma example - Generate embeddings using Gemma3 embedding model")]
pub struct Cli {
    /// The text to generate embeddings for
    #[clap(long, default_value = "Hello, this is a test sentence.")]
    prompt: String,

    /// The PRNG seed
    #[clap(long, default_value = "42")]
    seed: u64,

    /// Path to local model directory containing tokenizer.json, config.json, and model.safetensors
    #[clap(long)]
    model_dir: PathBuf,

    /// Whether to print the full embedding vector
    #[clap(long, default_value = "false")]
    print_full: bool,
}

fn get_tokenizer(model_dir: &PathBuf) -> Result<Tokenizer> {
    let tokenizer_filename = model_dir.join("tokenizer.json");
    if !tokenizer_filename.exists() {
        return Err(format!("tokenizer.json not found in {:?}", model_dir).into());
    }
    let t = Tokenizer::from_file(tokenizer_filename)?;
    Ok(t)
}

fn load_model(model_dir: &PathBuf) -> Result<EmbeddingGemma> {
    // Verify required files exist
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("config.json not found in {:?}", model_dir).into());
    }

    // Load config
    let file = std::fs::File::open(config_path)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    println!("[INFO] Model configuration:");
    println!("  - vocab_size: {}", model_args.vocab_size);
    println!("  - hidden_size: {}", model_args.hidden_size);
    println!("  - num_hidden_layers: {}", model_args.num_hidden_layers);
    println!(
        "  - num_attention_heads: {}",
        model_args.num_attention_heads
    );
    println!(
        "  - num_key_value_heads: {}",
        model_args.num_key_value_heads
    );
    println!("  - head_dim: {}", model_args.head_dim);
    println!("  - intermediate_size: {}", model_args.intermediate_size);
    println!("  - sliding_window: {}", model_args.sliding_window);

    if let Some(quant_config) = &model_args.quantization {
        println!(
            "  - quantization: {}-bit (group_size: {})",
            quant_config.bits, quant_config.group_size
        );
    }

    // Check if we have a single safetensors file
    let single_safetensors = model_dir.join("model.safetensors");
    if !single_safetensors.exists() {
        return Err(format!("model.safetensors not found in {:?}", model_dir).into());
    }

    // Create the model
    println!("[INFO] Creating model architecture...");
    let mut model = EmbeddingGemma::new(&model_args)?;

    // Load weights from safetensors
    println!("[INFO] Loading weights from safetensors...");
    model.load_safetensors(&single_safetensors)?;

    Ok(model)
}

fn main() -> Result<()> {
    // Parse args
    let cli = Cli::parse();

    mlx_rs::random::seed(cli.seed)?;

    println!("[INFO] Loading model from {:?}...", cli.model_dir);

    // Verify the model directory exists
    if !cli.model_dir.exists() {
        return Err(format!("Model directory not found: {:?}", cli.model_dir).into());
    }

    // Load tokenizer
    println!("[INFO] Loading tokenizer...");
    let tokenizer = get_tokenizer(&cli.model_dir)?;

    // Load model
    let mut model = load_model(&cli.model_dir)?;
    println!("[INFO] Model loaded successfully");

    // Prepare the prompt
    let prompt = cli.prompt.clone();
    println!("\n[INFO] Input text: {}", prompt);

    // Tokenize the prompt
    println!("[INFO] Tokenizing...");
    let encoding = tokenizer.encode(&prompt[..], true)?;
    let tokens = encoding.get_ids();
    println!("[INFO] Number of tokens: {}", tokens.len());

    // Convert to MLX array with batch dimension
    let input_ids = Array::from_slice(tokens, &[1, tokens.len() as i32]);

    // Generate embeddings
    println!("[INFO] Generating embeddings...");
    let embeddings = model.generate_embeddings(&input_ids)?;

    // Evaluate to ensure computation is complete
    eval([&embeddings])?;

    // Get the embedding shape
    let embedding_shape = embeddings.shape();
    println!("\n[RESULT] Embedding shape: {:?}", embedding_shape);

    // Convert to a vector for display
    let embedding_vec: Vec<f32> = embeddings.as_slice::<f32>().to_vec();

    // Verify L2 normalization
    let l2_norm: f32 = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "[RESULT] Embedding L2 norm: {:.6} (should be ~1.0)",
        l2_norm
    );

    // Print embedding dimensions
    if cli.print_full {
        println!("\n[RESULT] Full embedding vector:");
        for (i, val) in embedding_vec.iter().enumerate() {
            println!("  dim[{}]: {:.6}", i, val);
        }
    } else {
        println!("\n[RESULT] First 20 dimensions of embedding:");
        for (i, val) in embedding_vec.iter().take(20).enumerate() {
            println!("  dim[{}]: {:.6}", i, val);
        }
        println!("  ... ({} more dimensions)", embedding_vec.len() - 20);
    }

    println!(
        "\n[INFO] Total embedding dimensions: {}",
        embedding_vec.len()
    );

    // Example: compute similarity with itself (should be 1.0)
    let dot_product: f32 = embedding_vec.iter().map(|x| x * x).sum();
    println!(
        "[INFO] Self-similarity (dot product): {:.6} (should be ~1.0)",
        dot_product
    );

    println!("\n[SUCCESS] Embedding generation completed successfully!");

    Ok(())
}
