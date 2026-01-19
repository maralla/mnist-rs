mnist-rs
========

Handwritten digit recognition neural network implemented in pure Rust trains on MNIST dataset.

## Quick Start

```bash
# Download MNIST Data
./scripts/download_data.sh

# Train
cargo run -- train
cargo run -- predict model.txt image.png
```

## License

[MIT](./LICENSE)
