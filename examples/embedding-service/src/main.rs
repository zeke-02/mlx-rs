use mlx_lm::{
    cache::ConcatKeyValueCache,
    models::qwen3::{load_qwen3_model, Model, ModelInput},
};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{
    linalg::norm_l2,
    module::Module,
    ops::indexing::{Ellipsis, IndexOp, NewAxis},
    transforms::eval,
    Array,
};
use std::path::PathBuf;

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Embedding service example using Qwen3 embedding model")]
pub struct Cli {
    /// The text to generate embeddings for
    #[clap(long, default_value = "hello")]
    prompt: String,

    /// The PRNG seed
    #[clap(long, default_value = "0")]
    seed: u64,

    /// Path to local model directory containing tokenizer.json, config.json, and model.safetensors
    #[clap(long)]
    model_dir: PathBuf,
}

fn get_tokenizer(model_dir: &PathBuf) -> Result<Tokenizer> {
    let tokenizer_filename = model_dir.join("tokenizer.json");
    if !tokenizer_filename.exists() {
        return Err(format!("tokenizer.json not found in {:?}", model_dir).into());
    }
    let t = Tokenizer::from_file(tokenizer_filename)?;
    Ok(t)
}

fn load_model(model_dir: &PathBuf) -> Result<Model> {
    // Verify required files exist
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(format!("config.json not found in {:?}", model_dir).into());
    }

    // Check if we have a single safetensors file or an index file
    let single_safetensors = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");

    if !single_safetensors.exists() && !index_path.exists() {
        return Err(format!(
            "Neither model.safetensors nor model.safetensors.index.json found in {:?}",
            model_dir
        )
        .into());
    }

    // Load the model (it will find all the files in the directory)
    let model = load_qwen3_model(model_dir)?;

    Ok(model)
}

fn extract_embeddings(model: &mut Model, prompt_tokens: &Array) -> Result<Array> {
    // Create an empty cache for the forward pass
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();

    // Debug: print input shape
    println!("[DEBUG] Input tokens shape: {:?}", prompt_tokens.shape());
    println!("[DEBUG] Model hidden_size: {}", model.args.hidden_size);

    // Create model input
    let input = ModelInput {
        inputs: prompt_tokens,
        mask: None,
        cache: &mut cache,
    };

    // Forward pass through the model to get hidden states
    // We use model.model.forward() to get the hidden states directly,
    // bypassing the language modeling head
    let hidden_states = model.model.forward(input)?;

    // Debug: print hidden states shape
    println!("[DEBUG] Hidden states shape: {:?}", hidden_states.shape());

    // For embeddings, we typically use the last token's hidden state
    // or mean pooling across all tokens. Here we'll use the last token.
    let mut embeddings = hidden_states.index((Ellipsis, -1, ..));

    // Normalize embeddings to unit length (L2 norm = 1)
    // This makes cosine similarity equivalent to dot product
    let axes = [-1];
    let l2_norm = norm_l2(&embeddings, Some(&axes[..]), Some(true))?;
    embeddings = embeddings.divide(&l2_norm)?;

    Ok(embeddings)
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

    let mut tokenizer = get_tokenizer(&cli.model_dir)?;
    let mut model = load_model(&cli.model_dir)?;

    println!("[INFO] Model loaded successfully");
    println!(
        "[INFO] Model config - hidden_size: {}, num_layers: {}",
        model.args.hidden_size, model.args.num_hidden_layers
    );

    // Load chat template if available
    let tokenizer_config_filename = cli.model_dir.join("tokenizer_config.json");
    let chat_template = if tokenizer_config_filename.exists() {
        load_model_chat_template_from_file(tokenizer_config_filename)
            .ok()
            .flatten()
    } else {
        None
    };

    // Prepare the prompt
    let mut prompt = cli.prompt.clone();

    // Apply chat template if available
    if let Some(template) = chat_template {
        println!("[INFO] Applying chat template...");
        let conversations = vec![Conversation {
            role: Role::User,
            content: prompt.clone(),
        }];

        let model_id = cli.model_dir.to_string_lossy().to_string();
        let args = ApplyChatTemplateArgs {
            conversations: [conversations.into()],
            documents: None,
            model_id: &model_id,
            chat_template_id: None,
            add_generation_prompt: Some(true),
            continue_final_message: None,
        };

        let rendered = tokenizer.apply_chat_template(template, args)?;
        if let Some(first) = rendered.first() {
            prompt = first.clone();
        }
    }

    println!("[INFO] Processing prompt: {}", prompt);

    // Tokenize the prompt
    let encoding = tokenizer.encode(&prompt[..], true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);

    println!("[INFO] Extracting embeddings...");

    // Extract embeddings
    let embeddings = extract_embeddings(&mut model, &prompt_tokens)?;

    // Evaluate to ensure computation is complete
    eval([&embeddings])?;

    // Get the embedding vector
    let embedding_shape = embeddings.shape();
    println!("[INFO] Embedding shape: {:?}", embedding_shape);

    // Convert to a vector for display
    let embedding_vec: Vec<f32> = embeddings.as_slice::<f32>().to_vec();

    println!("\n[RESULT] Embedding vector (first 10 dimensions):");
    for (i, val) in embedding_vec.iter().take(10).enumerate() {
        println!("  dim {}: {:.6}", i, val);
    }

    println!(
        "\n[INFO] Total embedding dimensions: {}",
        embedding_vec.len()
    );
    println!("[INFO] Embedding L2 norm: {:.6}", {
        let sum_sq: f32 = embedding_vec.iter().map(|x| x * x).sum();
        sum_sq.sqrt()
    });

    Ok(())
}
