use crate::metal::MetalContext;
use std::ops::{Add, Sub};
use std::sync::OnceLock;

static METAL_CONTEXT: OnceLock<Option<MetalContext>> = OnceLock::new();

fn get_metal_context() -> &'static Option<MetalContext> {
    METAL_CONTEXT.get_or_init(MetalContext::new)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn transpose(&self) -> Self {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; self.size()];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Self {
            data,
            shape: vec![cols, rows],
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Try GPU acceleration first
        if let Some(metal) = get_metal_context() {
            if let Some(result) = metal.matmul(self, other) {
                return result;
            }
        }
        self.matmul_cpu(other)
    }

    fn matmul_cpu(&self, other: &Tensor) -> Tensor {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor::new(result, vec![m, n])
    }

    pub fn scale(&self, scalar: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x * scalar).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self {
            data: self.data.iter().map(|&x| f(x)).collect(),
            shape: self.shape.clone(),
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.size(), 4);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![3, 3]);
        assert_eq!(t.size(), 9);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = t.transpose();
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = &a + &b;
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let c = &a - &b;
        assert_eq!(c.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = a.scale(2.0);
        assert_eq!(b.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_apply() {
        let a = Tensor::new(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]);
        let b = a.apply(|x| x.sqrt());
        assert_eq!(b.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
