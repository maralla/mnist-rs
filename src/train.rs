use crate::layer::Linear;
use crate::data::MnistData;
use crate::network::{cross_entropy_gradient, cross_entropy_loss, Network};
use crate::tensor::Tensor;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 32,
            epochs: 10,
        }
    }
}

pub struct Trainer {
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn train(&mut self, network: &mut Network, data: &MnistData) -> Vec<f32> {
        let mut epoch_losses = Vec::new();
        let num_samples = data.len();
        let num_batches = num_samples / self.config.batch_size;

        for epoch in 0..self.config.epochs {
            let indices = self.shuffle_indices(num_samples);
            let mut total_loss = 0.0;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.config.batch_size;
                let end = start + self.config.batch_size;
                let batch_indices: Vec<usize> = indices[start..end].to_vec();

                let (images, labels) = data.get_batch(&batch_indices);
                let loss = self.train_batch(network, &images, &labels);
                total_loss += loss;
            }

            let avg_loss = total_loss / num_batches as f32;
            epoch_losses.push(avg_loss);

            println!("Epoch {}/{}: Loss = {:.4}", epoch + 1, self.config.epochs, avg_loss);
        }

        epoch_losses
    }

    fn train_batch(&mut self, network: &mut Network, images: &Tensor, labels: &Tensor) -> f32 {
        let predictions = network.forward(images);
        let loss = cross_entropy_loss(&predictions, labels);

        let grad = cross_entropy_gradient(&predictions, labels);
        network.backward(&grad, self.config.learning_rate);

        loss
    }

    fn shuffle_indices(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::rng());
        indices
    }
}

pub fn evaluate(network: &mut Network, data: &MnistData) -> f32 {
    let mut correct = 0;
    let total = data.len();

    for i in 0..total {
        let (image, _) = data.get_batch(&[i]);
        let predictions = network.predict(&image);
        let predicted_digit = predictions[0].0;

        if predicted_digit == data.labels[i] as usize {
            correct += 1;
        }
    }

    correct as f32 / total as f32
}

pub fn save_model(network: &Network, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    for (layer_idx, layer) in network.layers.iter().enumerate() {
        writeln!(file, "# Layer {}", layer_idx)?;

        write!(file, "weights {} {}", layer.weights.shape[0], layer.weights.shape[1])?;
        for val in &layer.weights.data {
            write!(file, " {}", val)?;
        }
        writeln!(file)?;

        write!(file, "bias {} {}", layer.bias.shape[0], layer.bias.shape[1])?;
        for val in &layer.bias.data {
            write!(file, " {}", val)?;
        }
        writeln!(file)?;
    }

    Ok(())
}

pub fn load_model(path: &str) -> std::io::Result<Network> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut layers = Vec::new();
    let mut current_weights: Option<Tensor> = None;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        match parts[0] {
            "weights" => {
                let rows: usize = parts[1].parse().unwrap();
                let cols: usize = parts[2].parse().unwrap();
                let data: Vec<f32> = parts[3..].iter().map(|s| s.parse().unwrap()).collect();

                if data.len() != rows * cols {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Weight data size mismatch",
                    ));
                }

                current_weights = Some(Tensor::new(data, vec![rows, cols]));
            }
            "bias" => {
                let rows: usize = parts[1].parse().unwrap();
                let cols: usize = parts[2].parse().unwrap();
                let data: Vec<f32> = parts[3..].iter().map(|s| s.parse().unwrap()).collect();

                if data.len() != rows * cols {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Bias data size mismatch",
                    ));
                }

                let bias = Tensor::new(data, vec![rows, cols]);
                if let Some(weights) = current_weights.take() {
                    layers.push(Linear::from_weights(weights, bias));
                }
            }
            _ => {}
        }
    }

    Ok(Network::from_layers(layers))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.epochs, 10);
    }

    #[test]
    fn test_trainer_shuffle() {
        let trainer = Trainer::new(TrainingConfig::default());
        let indices = trainer.shuffle_indices(10);

        assert_eq!(indices.len(), 10);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
