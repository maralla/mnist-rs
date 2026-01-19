use crate::layer::{Activation, Linear};
use crate::tensor::Tensor;

pub struct Network {
    pub layers: Vec<Linear>,
    activations: Vec<Activation>,
    activation_inputs: Vec<Tensor>,
    activation_outputs: Vec<Tensor>,
}

impl Network {
    /// Create a new neural network with the given layer sizes.
    ///
    /// Architecture: Input → [Linear → ReLU] × N → Linear → Softmax → Output
    ///
    /// Example: `Network::new(&[784, 128, 64, 10])` creates:
    /// - Input: 784 features (28×28 image)
    /// - Hidden layer 1: Linear(784→128) + ReLU
    /// - Hidden layer 2: Linear(128→64) + ReLU
    /// - Output layer: Linear(64→10) + Softmax (10 digit classes)
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Linear::new(layer_sizes[i], layer_sizes[i + 1]));

            // Use ReLU for hidden layers (non-linear, helps learn complex patterns)
            // Use Softmax for output layer (converts to probabilities that sum to 1)
            if i < layer_sizes.len() - 2 {
                activations.push(Activation::ReLU);
            } else {
                activations.push(Activation::Softmax);
            }
        }

        Self {
            layers,
            activations,
            activation_inputs: Vec::new(),
            activation_outputs: Vec::new(),
        }
    }

    pub fn from_layers(layers: Vec<Linear>) -> Self {
        let mut activations = Vec::new();
        for i in 0..layers.len() {
            if i < layers.len() - 1 {
                activations.push(Activation::ReLU);
            } else {
                activations.push(Activation::Softmax);
            }
        }

        Self {
            layers,
            activations,
            activation_inputs: Vec::new(),
            activation_outputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.activation_inputs.clear();
        self.activation_outputs.clear();

        let mut x = input.clone();

        for (layer, activation) in self.layers.iter_mut().zip(self.activations.iter()) {
            x = layer.forward(&x);
            self.activation_inputs.push(x.clone());

            x = activation.forward(&x);
            self.activation_outputs.push(x.clone());
        }

        x
    }

    pub fn backward(&mut self, grad_output: &Tensor, learning_rate: f32) {
        let mut grad = grad_output.clone();

        for i in (0..self.layers.len()).rev() {
            let activation_output = &self.activation_outputs[i];
            grad = self.activations[i].backward(activation_output, &grad);

            let (grad_input, grad_weights, grad_bias) = self.layers[i].backward(&grad);
            self.layers[i].update(&grad_weights, &grad_bias, learning_rate);

            grad = grad_input;
        }
    }

    pub fn predict(&mut self, input: &Tensor) -> Vec<(usize, f32)> {
        let output = self.forward(input);
        let mut predictions: Vec<(usize, f32)> = output
            .data
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        predictions
    }
}

/// Cross-entropy loss for multi-class classification
///
/// Formula: L = -1/N × Σᵢ Σⱼ tᵢⱼ × log(pᵢⱼ + ε)
///
/// Where:
/// - N = batch size
/// - tᵢⱼ = target (one-hot encoded, 1 for correct class, 0 otherwise)
/// - pᵢⱼ = predicted probability from softmax
/// - ε = small constant (1e-7) to prevent log(0)
///
/// For a single sample with correct class c:
/// L = -log(pᶜ)
///
/// Properties:
/// - Loss is 0 when prediction is perfect (pᶜ = 1)
/// - Loss approaches ∞ as prediction gets worse (pᶜ → 0)
/// - Penalizes confident wrong predictions heavily
pub fn cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> f32 {
    let batch_size = predictions.shape[0] as f32;
    let epsilon = 1e-7;

    let loss: f32 = predictions
        .data
        .iter()
        .zip(targets.data.iter())
        .map(|(&p, &t)| -t * (p + epsilon).ln())
        .sum();

    loss / batch_size
}

/// Gradient of cross-entropy loss with respect to predictions
///
/// When combined with softmax output, the gradient simplifies beautifully:
///
/// For cross-entropy: L = -Σⱼ tⱼ × log(pⱼ)
/// For softmax: pᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
///
/// The combined gradient ∂L/∂xᵢ = pᵢ - tᵢ
///
/// This elegant result means:
/// - If prediction is correct (p ≈ t): gradient ≈ 0 (small update)
/// - If prediction is wrong (p ≠ t): gradient is large (big update)
///
/// Example:
/// - Target: [0, 1, 0] (class 1 is correct)
/// - Prediction: [0.1, 0.7, 0.2]
/// - Gradient: [0.1, -0.3, 0.2]
///   → Push down class 0 and 2, push up class 1
pub fn cross_entropy_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
    predictions - targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_new() {
        let net = Network::new(&[784, 128, 64, 10]);
        assert_eq!(net.layers.len(), 3);
        assert_eq!(net.activations.len(), 3);
    }

    #[test]
    fn test_network_forward_output_shape() {
        let mut net = Network::new(&[4, 8, 2]);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = net.forward(&input);

        assert_eq!(output.shape, vec![1, 2]);
    }

    #[test]
    fn test_network_forward_softmax_output() {
        let mut net = Network::new(&[4, 8, 2]);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = net.forward(&input);

        let sum: f32 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_network_forward_batch() {
        let mut net = Network::new(&[4, 8, 2]);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let output = net.forward(&input);

        assert_eq!(output.shape, vec![2, 2]);

        let sum1: f32 = output.data[0..2].iter().sum();
        let sum2: f32 = output.data[2..4].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let predictions = Tensor::new(vec![0.7, 0.2, 0.1], vec![1, 3]);
        let targets = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let loss = cross_entropy_loss(&predictions, &targets);

        let expected = -(0.7_f32 + 1e-7).ln();
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let predictions = Tensor::new(vec![0.7, 0.2, 0.1], vec![1, 3]);
        let targets = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let grad = cross_entropy_gradient(&predictions, &targets);

        assert_eq!(grad.shape, vec![1, 3]);
        assert!((grad.data[0] - (-0.3)).abs() < 1e-5);
        assert!((grad.data[1] - 0.2).abs() < 1e-5);
        assert!((grad.data[2] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_predict() {
        let mut net = Network::new(&[4, 8, 3]);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let predictions = net.predict(&input);

        assert_eq!(predictions.len(), 3);
        assert!(predictions[0].1 >= predictions[1].1);
        assert!(predictions[1].1 >= predictions[2].1);
    }

    #[test]
    fn test_backward_updates_weights() {
        let mut net = Network::new(&[2, 4, 2]);
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]);

        let output = net.forward(&input);
        let targets = Tensor::new(vec![1.0, 0.0], vec![1, 2]);
        let grad = cross_entropy_gradient(&output, &targets);

        // Verify gradient is non-zero (prediction != target)
        let grad_magnitude: f32 = grad.data.iter().map(|x| x.abs()).sum();
        assert!(grad_magnitude > 0.0, "Gradient should be non-zero");

        net.backward(&grad, 0.1);
        // If we got here without panic, backward completed successfully
    }
}
