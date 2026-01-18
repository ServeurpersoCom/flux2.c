# FLUX.2-klein-4B Pure C Implementation

A dependency-free C implementation of the FLUX.2-klein-4B image generation model. Runs inference using only the C standard library and BLAS (Apple Accelerate on macOS, OpenBLAS on Linux).

On Apple Silicon Macs, Metal GPU acceleration is automatically enabled for faster matrix operations.

## Building

```bash
make
```

To check build configuration:
```bash
make info
```

## Downloading the Model

```bash
pip install huggingface_hub
python download_model.py
```

This downloads the VAE, transformer, and text encoder (~16GB total) to `./flux-klein-model`.

## Usage

Generate an image from a text prompt:

```bash
./flux -d flux-klein-model -p "A fluffy orange cat sitting on a windowsill" -o cat.png
```

Options:
- `-d PATH` - Model directory (required)
- `-p TEXT` - Text prompt for image generation
- `-e PATH` - Pre-computed text embeddings file (alternative to `-p`)
- `-o PATH` - Output image path
- `-W N` - Width (default: 1024)
- `-H N` - Height (default: 1024)
- `-s N` - Sampling steps (default: 4)
- `-S N` - Random seed
- `-v` - Verbose output with progress

## Components

- **Transformer**: FLUX DiT architecture (5 double blocks, 20 single blocks)
- **VAE**: AutoencoderKL for latent decoding
- **Text Encoder**: Qwen3-4B (2560 hidden dim, 36 layers)

All components are implemented in pure C with BLAS acceleration.

## Current Limitations

**Maximum resolution**: 1024x1024 pixels. Higher resolutions require prohibitive memory for the attention mechanisms (VAE attention alone needs ~17GB for the scores matrix at 2048x2048).

**Memory requirements**: The full model requires ~16GB RAM (transformer ~4GB, VAE ~300MB, text encoder ~8GB).

## License

MIT
