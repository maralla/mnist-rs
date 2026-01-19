use crate::tensor::Tensor;
use image::ImageReader;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const IMAGE_MAGIC: u32 = 2051;
const LABEL_MAGIC: u32 = 2049;

pub struct MnistData {
    pub images: Vec<Tensor>,
    pub labels: Vec<u8>,
}

impl MnistData {
    pub fn load<P: AsRef<Path>>(images_path: P, labels_path: P) -> io::Result<Self> {
        let images = load_images(images_path)?;
        let labels = load_labels(labels_path)?;

        assert_eq!(
            images.len(),
            labels.len(),
            "Number of images and labels must match"
        );

        Ok(Self { images, labels })
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let batch_size = indices.len();
        let image_size = 28 * 28;
        let num_classes = 10;

        let mut image_data = Vec::with_capacity(batch_size * image_size);
        let mut label_data = vec![0.0; batch_size * num_classes];

        for (batch_idx, &idx) in indices.iter().enumerate() {
            image_data.extend_from_slice(&self.images[idx].data);

            let label = self.labels[idx] as usize;
            label_data[batch_idx * num_classes + label] = 1.0;
        }

        let images = Tensor::new(image_data, vec![batch_size, image_size]);
        let labels = Tensor::new(label_data, vec![batch_size, num_classes]);

        (images, labels)
    }
}

fn load_images<P: AsRef<Path>>(path: P) -> io::Result<Vec<Tensor>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let magic = read_u32_be(&buffer[0..4]);
    if magic != IMAGE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid image file magic number: {}", magic),
        ));
    }

    let num_images = read_u32_be(&buffer[4..8]) as usize;
    let rows = read_u32_be(&buffer[8..12]) as usize;
    let cols = read_u32_be(&buffer[12..16]) as usize;

    let image_size = rows * cols;
    let mut images = Vec::with_capacity(num_images);

    for i in 0..num_images {
        let start = 16 + i * image_size;
        let end = start + image_size;
        let pixels: Vec<f32> = buffer[start..end]
            .iter()
            .map(|&b| b as f32 / 255.0)
            .collect();
        images.push(Tensor::new(pixels, vec![1, image_size]));
    }

    Ok(images)
}

fn load_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let magic = read_u32_be(&buffer[0..4]);
    if magic != LABEL_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid label file magic number: {}", magic),
        ));
    }

    let num_labels = read_u32_be(&buffer[4..8]) as usize;
    let labels = buffer[8..8 + num_labels].to_vec();

    Ok(labels)
}

fn read_u32_be(bytes: &[u8]) -> u32 {
    ((bytes[0] as u32) << 24)
        | ((bytes[1] as u32) << 16)
        | ((bytes[2] as u32) << 8)
        | (bytes[3] as u32)
}

pub fn load_image_file<P: AsRef<Path>>(path: P) -> io::Result<Tensor> {
    let img = ImageReader::open(&path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .decode()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let gray = img.to_luma8();
    let resized = image::imageops::resize(&gray, 28, 28, image::imageops::FilterType::Lanczos3);

    // Invert colors: MNIST has white digits on black background,
    // but typical images have black digits on white background
    let normalized: Vec<f32> = resized
        .into_raw()
        .iter()
        .map(|&p| 1.0 - (p as f32 / 255.0))
        .collect();

    Ok(Tensor::new(normalized, vec![1, 28 * 28]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_u32_be() {
        let bytes = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(read_u32_be(&bytes), 0x00010203);
    }
}
