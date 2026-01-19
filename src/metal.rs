#[cfg(target_os = "macos")]
mod metal_impl {
    use crate::tensor::Tensor;
    use metal::foreign_types::{ForeignType, ForeignTypeRef};
    use metal::{Device, MTLResourceOptions};
    use objc2::ffi::NSUInteger;
    use objc2::runtime::{AnyObject, Bool};
    use objc2::{class, msg_send};

    #[link(name = "MetalPerformanceShaders", kind = "framework")]
    unsafe extern "C" {}

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

            self.execute_mps_matmul(&a.data, &b.data, m, k, n)
        }

        fn execute_mps_matmul(
            &self,
            a: &[f32],
            b: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> Option<Tensor> {
            unsafe {
                let a_size = (a.len() * std::mem::size_of::<f32>()) as u64;
                let b_size = (b.len() * std::mem::size_of::<f32>()) as u64;
                let c_size = (m * n * std::mem::size_of::<f32>()) as u64;

                // Create Metal buffers
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

                // MPSDataTypeFloat32 = 0x10000000 | 32
                let float32_type: u32 = 0x10000000 | 32;

                // Create matrix descriptors
                let a_row_bytes = (k * std::mem::size_of::<f32>()) as NSUInteger;
                let b_row_bytes = (n * std::mem::size_of::<f32>()) as NSUInteger;
                let c_row_bytes = (n * std::mem::size_of::<f32>()) as NSUInteger;

                let desc_class = class!(MPSMatrixDescriptor);

                let a_desc: *mut AnyObject = msg_send![
                    desc_class,
                    matrixDescriptorWithRows: m as NSUInteger,
                    columns: k as NSUInteger,
                    rowBytes: a_row_bytes,
                    dataType: float32_type
                ];

                let b_desc: *mut AnyObject = msg_send![
                    desc_class,
                    matrixDescriptorWithRows: k as NSUInteger,
                    columns: n as NSUInteger,
                    rowBytes: b_row_bytes,
                    dataType: float32_type
                ];

                let c_desc: *mut AnyObject = msg_send![
                    desc_class,
                    matrixDescriptorWithRows: m as NSUInteger,
                    columns: n as NSUInteger,
                    rowBytes: c_row_bytes,
                    dataType: float32_type
                ];

                // Create MPS matrices - cast buffer pointers to *mut AnyObject
                let matrix_class = class!(MPSMatrix);

                let a_buffer_ptr = a_buffer.as_ptr() as *mut AnyObject;
                let b_buffer_ptr = b_buffer.as_ptr() as *mut AnyObject;
                let c_buffer_ptr = c_buffer.as_ptr() as *mut AnyObject;

                let a_matrix: *mut AnyObject = msg_send![matrix_class, alloc];
                let a_matrix: *mut AnyObject = msg_send![
                    a_matrix,
                    initWithBuffer: a_buffer_ptr,
                    offset: 0 as NSUInteger,
                    descriptor: a_desc
                ];

                let b_matrix: *mut AnyObject = msg_send![matrix_class, alloc];
                let b_matrix: *mut AnyObject = msg_send![
                    b_matrix,
                    initWithBuffer: b_buffer_ptr,
                    offset: 0 as NSUInteger,
                    descriptor: b_desc
                ];

                let c_matrix: *mut AnyObject = msg_send![matrix_class, alloc];
                let c_matrix: *mut AnyObject = msg_send![
                    c_matrix,
                    initWithBuffer: c_buffer_ptr,
                    offset: 0 as NSUInteger,
                    descriptor: c_desc
                ];

                // Create matrix multiplication kernel - cast device pointer
                let matmul_class = class!(MPSMatrixMultiplication);
                let device_ptr = self.device.as_ptr() as *mut AnyObject;

                let matmul_kernel: *mut AnyObject = msg_send![matmul_class, alloc];
                let matmul_kernel: *mut AnyObject = msg_send![
                    matmul_kernel,
                    initWithDevice: device_ptr,
                    transposeLeft: Bool::NO,
                    transposeRight: Bool::NO,
                    resultRows: m as NSUInteger,
                    resultColumns: n as NSUInteger,
                    interiorColumns: k as NSUInteger,
                    alpha: 1.0f64,
                    beta: 0.0f64
                ];

                // Encode and execute - cast command buffer pointer
                let command_buffer = self.command_queue.new_command_buffer();
                let cmd_buf_ptr = command_buffer.as_ptr() as *mut AnyObject;

                let _: () = msg_send![
                    matmul_kernel,
                    encodeToCommandBuffer: cmd_buf_ptr,
                    leftMatrix: a_matrix,
                    rightMatrix: b_matrix,
                    resultMatrix: c_matrix
                ];

                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Read results
                let c_ptr = c_buffer.contents() as *const f32;
                let c_data: Vec<f32> = std::slice::from_raw_parts(c_ptr, m * n).to_vec();

                // Release MPS objects (they were created with alloc/init)
                let _: () = msg_send![a_matrix, release];
                let _: () = msg_send![b_matrix, release];
                let _: () = msg_send![c_matrix, release];
                let _: () = msg_send![matmul_kernel, release];

                Some(Tensor::new(c_data, vec![m, n]))
            }
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

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_metal_context_unavailable_on_linux() {
        use super::*;
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
