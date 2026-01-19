#[cfg(target_os = "macos")]
mod metal_impl {
    use crate::tensor::Tensor;
    use metal::{Device, MTLResourceOptions, MTLSize};

    pub struct MetalContext {
        device: Device,
        command_queue: metal::CommandQueue,
    }

    impl MetalContext {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let command_queue = device.new_command_queue();

            Some(Self {
                device,
                command_queue,
            })
        }

        pub fn is_available(&self) -> bool {
            true
        }

        /// Matrix multiplication using MPS (Metal Performance Shaders)
        pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Option<Tensor> {
            if a.ndim() != 2 || b.ndim() != 2 || a.shape[1] != b.shape[0] {
                return None;
            }

            let m = a.shape[0];
            let k = a.shape[1];
            let n = b.shape[1];

            // For small matrices, CPU is faster due to GPU overhead
            if m * k * n < 4096 {
                return None;
            }

            self.execute_matmul(&a.data, &b.data, m, k, n)
        }

        fn execute_matmul(
            &self,
            a: &[f32],
            b: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> Option<Tensor> {
            let a_size = (a.len() * std::mem::size_of::<f32>()) as u64;
            let b_size = (b.len() * std::mem::size_of::<f32>()) as u64;
            let c_size = (m * n * std::mem::size_of::<f32>()) as u64;

            // Create buffers
            let a_buffer = self.device.new_buffer_with_data(
                a.as_ptr() as *const _,
                a_size,
                MTLResourceOptions::StorageModeShared,
            );
            let b_buffer = self.device.new_buffer_with_data(
                b.as_ptr() as *const _,
                b_size,
                MTLResourceOptions::StorageModeShared,
            );
            let c_buffer = self
                .device
                .new_buffer(c_size, MTLResourceOptions::StorageModeShared);

            // Create MPS matrix descriptors
            let a_desc = metal::MPSMatrixDescriptor::new(
                m as u64,
                k as u64,
                (k * std::mem::size_of::<f32>()) as u64,
                metal::MPSDataType::Float32,
            );
            let b_desc = metal::MPSMatrixDescriptor::new(
                k as u64,
                n as u64,
                (n * std::mem::size_of::<f32>()) as u64,
                metal::MPSDataType::Float32,
            );
            let c_desc = metal::MPSMatrixDescriptor::new(
                m as u64,
                n as u64,
                (n * std::mem::size_of::<f32>()) as u64,
                metal::MPSDataType::Float32,
            );

            // Create MPS matrices
            let a_matrix = metal::MPSMatrix::init_with_buffer_descriptor(&a_buffer, 0, &a_desc);
            let b_matrix = metal::MPSMatrix::init_with_buffer_descriptor(&b_buffer, 0, &b_desc);
            let c_matrix = metal::MPSMatrix::init_with_buffer_descriptor(&c_buffer, 0, &c_desc);

            // Create and encode matrix multiplication
            let matmul = metal::MPSMatrixMultiplication::init(
                &self.device,
                false, // transpose left
                false, // transpose right
                m as u64,
                n as u64,
                k as u64,
                1.0, // alpha
                0.0, // beta
            );

            let command_buffer = self.command_queue.new_command_buffer();
            matmul.encode_to_command_buffer(command_buffer, &a_matrix, &b_matrix, &c_matrix);

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read results
            let c_ptr = c_buffer.contents() as *const f32;
            let c_data: Vec<f32> = unsafe { std::slice::from_raw_parts(c_ptr, m * n).to_vec() };

            Some(Tensor::new(c_data, vec![m, n]))
        }
    }
}

#[cfg(target_os = "macos")]
pub use metal_impl::MetalContext;

#[cfg(not(target_os = "macos"))]
pub struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub fn new() -> Option<Self> {
        None
    }

    pub fn is_available(&self) -> bool {
        false
    }

    pub fn matmul(
        &self,
        _a: &crate::tensor::Tensor,
        _b: &crate::tensor::Tensor,
    ) -> Option<crate::tensor::Tensor> {
        None
    }
}

#[cfg(not(target_os = "macos"))]
impl Default for MetalContext {
    fn default() -> Self {
        Self
    }
}

pub fn has_metal_support() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_metal_support() {
        let expected = cfg!(target_os = "macos");
        assert_eq!(has_metal_support(), expected);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_metal_context_unavailable_on_linux() {
        let ctx = MetalContext::new();
        assert!(ctx.is_none());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_matmul_fallback() {
        use crate::tensor::Tensor;

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

        let ctx = MetalContext::default();
        let result = ctx.matmul(&a, &b);
        assert!(result.is_none());
    }
}
