use mnist_rs::data::{load_image_file, MnistData};
use mnist_rs::network::Network;
use mnist_rs::train::{evaluate, load_model, save_model, Trainer, TrainingConfig};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "predict" => cmd_predict(&args[2..]),
        "help" | "--help" | "-h" => print_usage(&args[0]),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage(&args[0]);
            process::exit(1);
        }
    }
}

fn print_usage(program: &str) {
    println!("mnist-rs: Handwritten digit recognition using neural networks\n");
    println!("Usage:");
    println!("  {} train [options]", program);
    println!("  {} predict <model_path> <image_path> [top_k]", program);
    println!();
    println!("Commands:");
    println!("  train     Train a new model on MNIST dataset");
    println!("  predict   Predict digit from an image file");
    println!();
    println!("Train options:");
    println!("  --data <path>       Path to MNIST data directory (default: ./data)");
    println!("  --epochs <n>        Number of training epochs (default: 10)");
    println!("  --batch-size <n>    Batch size for training (default: 32)");
    println!("  --lr <rate>         Learning rate (default: 0.01)");
    println!("  --output <path>     Output model file path (default: model.txt)");
    println!();
    println!("Predict options:");
    println!("  model_path          Path to trained model file");
    println!("  image_path          Path to image file (PNG format, will be resized to 28x28)");
    println!("  top_k               Number of top predictions to show (default: 3)");
}

fn cmd_train(args: &[String]) {
    let mut data_path = String::from("./data");
    let mut epochs = 10;
    let mut batch_size = 32;
    let mut learning_rate = 0.01f32;
    let mut output_path = String::from("model.txt");

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                if i < args.len() {
                    data_path = args[i].clone();
                }
            }
            "--epochs" => {
                i += 1;
                if i < args.len() {
                    epochs = args[i].parse().unwrap_or(10);
                }
            }
            "--batch-size" => {
                i += 1;
                if i < args.len() {
                    batch_size = args[i].parse().unwrap_or(32);
                }
            }
            "--lr" => {
                i += 1;
                if i < args.len() {
                    learning_rate = args[i].parse().unwrap_or(0.01);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = args[i].clone();
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading MNIST training data from {}...", data_path);

    let train_images_path = format!("{}/train-images-idx3-ubyte", data_path);
    let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_path);

    let train_data = match MnistData::load(&train_images_path, &train_labels_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load training data: {}", e);
            eprintln!("Please download MNIST dataset and extract to {}", data_path);
            eprintln!("Run ./scripts/download_data.sh");
            process::exit(1);
        }
    };

    println!("Loaded {} training samples", train_data.len());

    let mut network = Network::new(&[784, 128, 64, 10]);
    println!("Created network: 784 -> 128 -> 64 -> 10");

    let config = TrainingConfig {
        learning_rate,
        batch_size,
        epochs,
    };

    println!("Training with: epochs={}, batch_size={}, lr={}", epochs, batch_size, learning_rate);

    let mut trainer = Trainer::new(config);
    let losses = trainer.train(&mut network, &train_data);

    println!("Training complete. Final loss: {:.4}", losses.last().unwrap_or(&0.0));

    let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_path);
    let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_path);

    if let Ok(test_data) = MnistData::load(&test_images_path, &test_labels_path) {
        println!("Evaluating on {} test samples...", test_data.len());
        let accuracy = evaluate(&mut network, &test_data);
        println!("Test accuracy: {:.2}%", accuracy * 100.0);
    }

    println!("Saving model to {}...", output_path);
    if let Err(e) = save_model(&network, &output_path) {
        eprintln!("Failed to save model: {}", e);
        process::exit(1);
    }

    println!("Model saved successfully!");
}

fn cmd_predict(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Error: Missing required arguments");
        eprintln!("Usage: mnist-rs predict <model_path> <image_path> [top_k]");
        process::exit(1);
    }

    let model_path = &args[0];
    let image_path = &args[1];
    let top_k: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(3)
    } else {
        3
    };

    println!("Loading model from {}...", model_path);
    let mut network = match load_model(model_path) {
        Ok(net) => net,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            process::exit(1);
        }
    };

    println!("Loading image from {}...", image_path);
    let image = match load_image_file(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            process::exit(1);
        }
    };

    let predictions = network.predict(&image);

    println!("\nPrediction results:");
    println!("-------------------");
    for (digit, probability) in predictions.iter().take(top_k) {
        println!("  Digit {}: {:.2}%", digit, probability * 100.0);
    }

    let (top_digit, top_prob) = &predictions[0];
    println!("\nPredicted digit: {} (confidence: {:.2}%)", top_digit, top_prob * 100.0);
}
