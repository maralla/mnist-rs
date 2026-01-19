use crate::tensor::Tensor;
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Softmax,
}

impl Activation {
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => relu_forward(input),
            Activation::Softmax => softmax_forward(input),
        }
    }

    pub fn backward(&self, input: &Tensor, grad_output: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => relu_backward(input, grad_output),
            Activation::Softmax => softmax_backward(input, grad_output),
        }
    }
}

/// ReLU (Rectified Linear Unit) forward pass
/// Formula: f(x) = max(0, x)
/// Output: x if x > 0, else 0
fn relu_forward(input: &Tensor) -> Tensor {
    input.apply(|x| if x > 0.0 { x } else { 0.0 })
}

/// ReLU backward pass (gradient computation)
/// Formula: f'(x) = 1 if x > 0, else 0
/// Gradient flows through only for positive inputs
fn relu_backward(input: &Tensor, grad_output: &Tensor) -> Tensor {
    let grad_input: Vec<f32> = input
        .data
        .iter()
        .zip(grad_output.data.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
        .collect();
    Tensor::new(grad_input, input.shape.clone())
}

/// Softmax forward pass
/// Formula: softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
///
/// Numerically stable version:
/// 1. Subtract max(x) to prevent overflow: x' = x - max(x)
/// 2. Compute: exp(x'ᵢ) / Σⱼ exp(x'ⱼ)
///
/// Properties:
/// - Output values in range (0, 1)
/// - Sum of all outputs = 1 (probability distribution)
/// - Used for multi-class classification
fn softmax_forward(input: &Tensor) -> Tensor {
    let batch_size = input.shape[0];
    let features = input.shape[1];
    let mut result = vec![0.0; input.size()];

    for b in 0..batch_size {
        let row_start = b * features;
        let row_end = row_start + features;
        let row = &input.data[row_start..row_end];

        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        for (i, &e) in exp_vals.iter().enumerate() {
            result[row_start + i] = e / sum;
        }
    }

    Tensor::new(result, input.shape.clone())
}

/// Softmax backward pass (Jacobian computation)
///
/// For softmax output s = softmax(x), the Jacobian is:
/// ∂sᵢ/∂xⱼ = sᵢ × (δᵢⱼ - sⱼ)
/// where δᵢⱼ is the Kronecker delta (1 if i=j, 0 otherwise)
///
/// Expanded:
/// - If i == j: ∂sᵢ/∂xᵢ = sᵢ × (1 - sᵢ)
/// - If i != j: ∂sᵢ/∂xⱼ = -sᵢ × sⱼ
///
/// Chain rule: grad_inputᵢ = Σⱼ (∂sⱼ/∂xᵢ × grad_outputⱼ)
fn softmax_backward(output: &Tensor, grad_output: &Tensor) -> Tensor {
    let batch_size = output.shape[0];
    let features = output.shape[1];
    let mut grad_input = vec![0.0; output.size()];

    for b in 0..batch_size {
        let row_start = b * features;

        for i in 0..features {
            let mut sum = 0.0;
            for j in 0..features {
                let s_i = output.data[row_start + i];
                let s_j = output.data[row_start + j];
                let g_j = grad_output.data[row_start + j];

                if i == j {
                    sum += s_i * (1.0 - s_i) * g_j;
                } else {
                    sum += -s_i * s_j * g_j;
                }
            }
            grad_input[row_start + i] = sum;
        }
    }

    Tensor::new(grad_input, output.shape.clone())
}

pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
    input_cache: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::rng();
        let scale = (2.0 / in_features as f32).sqrt();
        let weights = Tensor::new(
            (0..in_features * out_features)
                .map(|_| rng.random_range(-scale..scale))
                .collect(),
            vec![in_features, out_features],
        );
        let bias = Tensor::zeros(vec![1, out_features]);

        Self {
            weights,
            bias,
            input_cache: None,
        }
    }

    pub fn from_weights(weights: Tensor, bias: Tensor) -> Self {
        Self {
            weights,
            bias,
            input_cache: None,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_cache = Some(input.clone());

        let output = input.matmul(&self.weights);
        broadcast_add(&output, &self.bias)
    }

    pub fn backward(&self, grad_output: &Tensor) -> (Tensor, Tensor, Tensor) {
        let input = self.input_cache.as_ref().expect("Forward must be called first");

        let grad_weights = input.transpose().matmul(grad_output);

        let batch_size = grad_output.shape[0];
        let features = grad_output.shape[1];
        let mut grad_bias_data = vec![0.0; features];
        for b in 0..batch_size {
            for f in 0..features {
                grad_bias_data[f] += grad_output.data[b * features + f];
            }
        }
        let grad_bias = Tensor::new(grad_bias_data, vec![1, features]);

        let grad_input = grad_output.matmul(&self.weights.transpose());

        (grad_input, grad_weights, grad_bias)
    }

    pub fn update(&mut self, grad_weights: &Tensor, grad_bias: &Tensor, learning_rate: f32) {
        let w_update = grad_weights.scale(learning_rate);
        self.weights = &self.weights - &w_update;

        let b_update = grad_bias.scale(learning_rate);
        self.bias = &self.bias - &b_update;
    }
}

fn broadcast_add(tensor: &Tensor, bias: &Tensor) -> Tensor {
    let batch_size = tensor.shape[0];
    let features = tensor.shape[1];
    let mut result = tensor.data.clone();

    for b in 0..batch_size {
        for f in 0..features {
            result[b * features + f] += bias.data[f];
        }
    }

    Tensor::new(result, tensor.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_new() {
        let layer = Linear::new(10, 5);
        assert_eq!(layer.weights.shape, vec![10, 5]);
        assert_eq!(layer.bias.shape, vec![1, 5]);
    }

    #[test]
    fn test_linear_forward() {
        let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let bias = Tensor::new(vec![0.1, 0.2, 0.3], vec![1, 3]);
        let mut layer = Linear::from_weights(weights, bias);

        let input = Tensor::new(vec![1.0, 1.0], vec![1, 2]);
        let output = layer.forward(&input);

        assert_eq!(output.shape, vec![1, 3]);
        assert!((output.data[0] - 5.1).abs() < 1e-5);
        assert!((output.data[1] - 7.2).abs() < 1e-5);
        assert!((output.data[2] - 9.3).abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_batch() {
        let weights = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let bias = Tensor::zeros(vec![1, 2]);
        let mut layer = Linear::from_weights(weights, bias);

        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let output = layer.forward(&input);

        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linear_backward() {
        let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let bias = Tensor::zeros(vec![1, 2]);
        let mut layer = Linear::from_weights(weights, bias);

        let input = Tensor::new(vec![1.0, 1.0], vec![1, 2]);
        let _ = layer.forward(&input);

        let grad_output = Tensor::new(vec![1.0, 1.0], vec![1, 2]);
        let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output);

        assert_eq!(grad_input.shape, vec![1, 2]);
        assert_eq!(grad_weights.shape, vec![2, 2]);
        assert_eq!(grad_bias.shape, vec![1, 2]);
    }

    #[test]
    fn test_broadcast_add() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let bias = Tensor::new(vec![10.0, 20.0], vec![1, 2]);
        let result = broadcast_add(&tensor, &bias);

        assert_eq!(result.data, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_linear_update() {
        let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let bias = Tensor::new(vec![0.5, 0.5], vec![1, 2]);
        let mut layer = Linear::from_weights(weights, bias);

        let grad_weights = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let grad_bias = Tensor::new(vec![1.0, 1.0], vec![1, 2]);

        layer.update(&grad_weights, &grad_bias, 0.1);

        assert!((layer.weights.data[0] - 0.9).abs() < 1e-5);
        assert!((layer.bias.data[0] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_relu_forward_positive() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let output = relu_forward(&input);
        assert_eq!(output.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_forward_negative() {
        let input = Tensor::new(vec![-1.0, -2.0, 3.0, -4.0], vec![2, 2]);
        let output = relu_forward(&input);
        assert_eq!(output.data, vec![0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_relu_backward() {
        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]);
        let grad_output = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let grad_input = relu_backward(&input, &grad_output);
        assert_eq!(grad_input.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_forward_sum_to_one() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let output = softmax_forward(&input);

        let sum: f32 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_forward_batch() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let output = softmax_forward(&input);

        let sum1: f32 = output.data[0..3].iter().sum();
        let sum2: f32 = output.data[3..6].iter().sum();

        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_forward_ordering() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let output = softmax_forward(&input);

        assert!(output.data[0] < output.data[1]);
        assert!(output.data[1] < output.data[2]);
    }

    #[test]
    fn test_activation_enum_relu() {
        let act = Activation::ReLU;
        let input = Tensor::new(vec![-1.0, 2.0], vec![1, 2]);
        let output = act.forward(&input);
        assert_eq!(output.data, vec![0.0, 2.0]);
    }

    #[test]
    fn test_activation_enum_softmax() {
        let act = Activation::Softmax;
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let output = act.forward(&input);

        let sum: f32 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
