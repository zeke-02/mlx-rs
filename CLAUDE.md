# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mlx-rs provides Rust bindings for Apple's MLX machine learning library. The project is organized as a Cargo workspace with multiple crates that provide different layers of abstraction.

## Building and Testing

### Build Commands
```bash
# Build the entire workspace
cargo build

# Build with specific features
cargo build --features metal,accelerate
cargo build --no-default-features

# Build in release mode
cargo build --release
```

### Testing
**CRITICAL**: MLX is not thread-safe. Always run tests with `--test-threads=1`:

```bash
# Run all tests
cargo test --all -- --test-threads=1

# Run tests for a specific package
cargo test -p mlx-rs -- --test-threads=1
cargo test -p mlx-lm -- --test-threads=1

# Run a specific test
cargo test -p mlx-tests --test test_optimizers -- --test-threads=1
```

### Linting and Formatting
```bash
# Format code
cargo fmt

# Check formatting without modifying files
cargo fmt -- --check

# Run clippy
cargo clippy -- -D warnings
```

### Running Examples
```bash
# Run an example from the examples/ directory
cargo run --example mnist
cargo run --example mistral
```

## Architecture

### Workspace Structure

The project is organized into several specialized crates:

- **mlx-sys**: Low-level FFI bindings to mlx-c (the C API for MLX). Uses bindgen to generate Rust bindings. Builds MLX via CMake in build.rs. Version follows mlx-c versioning.

- **mlx-rs**: Main high-level Rust API for MLX. Provides safe, idiomatic Rust wrappers around mlx-sys. Contains core functionality including:
  - `Array` type and operations
  - Neural network layers (in `nn/` module)
  - Automatic differentiation (in `transforms/` module)
  - Random number generation
  - Linear algebra operations
  - Optimizers (SGD, Adam, AdamW, etc.)

- **mlx-macros**: Procedural macros for code generation, including:
  - `ModuleParameters` - derives parameter management for neural network modules
  - `Quantizable` - derives quantization support
  - Builder pattern generation

- **mlx-internal-macros**: Internal macros for library implementation

- **mlx-lm**: Language model support library. Provides:
  - Pre-built model architectures (currently Qwen3)
  - Text generation utilities
  - KV caching for efficient inference
  - Tokenizer integration

- **mlx-lm-utils**: Utilities for language models (RoPE, attention masks, etc.)

- **mlx-tests**: Integration tests for macro functionality

### Key Architectural Concepts

#### Lazy Evaluation
Operations in MLX are lazy - they construct computation graphs without immediately executing. Arrays must be explicitly evaluated using:
- `array.eval()` - explicit evaluation
- Implicit evaluation happens when: printing arrays, calling `array.item()`, or accessing data with `array.as_slice()`

#### Unified Memory
Arrays exist in shared memory space regardless of device (CPU/GPU). Operations can be performed across devices without explicit data transfers.

#### Automatic Differentiation
The `transforms` module provides gradient computation. **Important**: In Rust, closures cannot implicitly capture arrays for differentiation. Always pass required arrays as function inputs:

```rust
// Correct approach - pass all arrays as inputs
let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
    let w = &inputs[0];
    let x = &inputs[1];
    let y = &inputs[2];
    // ... compute loss
};
let grad = transforms::grad(loss_fn, &[0])(&inputs)?;
```

Do not capture arrays from outer scope as this causes segfaults.

#### Module System
Neural network modules implement the `Module` trait. Use the `ModuleParameters` derive macro to automatically generate parameter management code. Modules can be:
- Frozen/unfrozen for training control
- Quantized using the `Quantizable` derive macro
- Composed into larger architectures

## Feature Flags

- `metal` - enables GPU acceleration via Metal (default: enabled)
- `accelerate` - enables Apple Accelerate framework (default: enabled)
- `safetensors` - enables safetensors format support for mlx-rs

## Platform Requirements

- **Platform**: macOS with Apple Silicon (aarch64-apple-darwin) or iOS
- **Minimum Rust**: 1.82.0
- **Build dependencies**: CMake, C++ compiler
- **Runtime**: Metal framework (for GPU), Accelerate framework

## Development Notes

### Adding New Models to mlx-lm
1. Create a new file in `mlx-lm/src/models/`
2. Define model configuration struct with `serde::Deserialize`
3. Implement model layers using `ModuleParameters` and `Quantizable` macros
4. Add model to `mlx-lm/src/models/mod.rs`

### Working with Submodules
The mlx-sys crate includes mlx-c as a git submodule. When cloning or updating, use:
```bash
git submodule update --init --recursive
```

### Version Management
- mlx-rs, mlx-lm, mlx-macros: Follow MLX versioning (currently 0.25.x)
- mlx-sys: Follows mlx-c versioning (currently 0.2.x)
- Version bumps should maintain this alignment

## Common Patterns

### Creating Arrays
```rust
use mlx_rs::{array, Array};

let a = array!([1, 2, 3, 4]);
let b = Array::from_slice(&[1.0, 2.0, 3.0], &[3]);
```

### Neural Network Module
```rust
use mlx_rs::macros::{ModuleParameters, Quantizable};
use mlx_rs::module::Module;

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct MyLayer {
    weight: Array,
    bias: Array,
}
```

### Gradient Computation
```rust
use mlx_rs::transforms;

let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
    // compute loss from inputs
};
let grad_fn = transforms::grad(loss_fn, &[0]); // differentiate w.r.t. first arg
let gradients = grad_fn(&inputs)?;
```
