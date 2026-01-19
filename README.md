mnist-rs
========

Handwritten digit recognition neural network built from scratch in Rust, trained on the MNIST dataset.

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
