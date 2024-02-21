pub mod lib;
use lib::mnist_data::*;
use lib::{network::Network, network_par::NetworkPar, activations::{SIGMOID, RELU}};
use lib::{matrix_par::MatrixPar};
use lib::{matrix::Matrix};
use std::time::{Instant, Duration};

fn main() {
    
    // Full version
    // let test_images: Vec<Vec<f64>> = load_images("./data/t10k-images-idx3-ubyte");
    // let test_labels: Vec<Vec<f64>> = load_labels("./data/t10k-labels-idx1-ubyte");
    // let train_images: Vec<Vec<f64>> = load_images("./data/train-images-idx3-ubyte");
    // let train_labels: Vec<Vec<f64>> = load_labels("./data/train-labels-idx1-ubyte");

    // Concise version
    let test_images: Vec<Vec<f64>> = load_images("./data/t10k-images-idx3-ubyte").into_iter().take(100).collect();
    let test_labels: Vec<Vec<f64>> = load_labels("./data/t10k-labels-idx1-ubyte").into_iter().take(100).collect();
    let train_images: Vec<Vec<f64>> = load_images("./data/train-images-idx3-ubyte").into_iter().take(1000).collect();
    let train_labels: Vec<Vec<f64>> = load_labels("./data/train-labels-idx1-ubyte").into_iter().take(1000).collect();

    // Normal Neural Network
    // let mut network = Network::new(vec![785, 128, 10], 0.001, RELU);
    // network.train(train_images, train_labels, 5);
    // network.test(test_images, test_labels);

    // Parallel Neural Netwok
    // let mut networkPar = NetworkPar::new(vec![785, 128, 10], 0.001, RELU);
    // networkPar.train(train_images, train_labels, 1);
    // networkPar.test(test_images, test_labels);

    fn test_matrix_performance() {
        let matrix_sizes = vec![10000]; // Add more sizes as needed
        let num_iterations = 10; // Adjust as needed
    
        for &size in &matrix_sizes {
            let mut matrix_a = MatrixPar::randoms(size, size);
            let mut matrix_b = MatrixPar::randoms(size, size);
    
            let mut total_duration = Duration::new(0, 0);
    
            for _ in 0..num_iterations {
                let start_time = Instant::now();
                
                let _result = matrix_a.dot_multiply(&matrix_b); // Change the method here  
    
                let duration = start_time.elapsed();
                total_duration += duration;
            }
    
            let average_duration = total_duration / num_iterations as u32;
    
            println!(
                "MatrixPar Size: {}, Average Duration: {:?}",
                size, average_duration
            );
        }

        for &size in &matrix_sizes {
            let mut matrix_a = Matrix::randoms(size, size); // Assuming you have a method to generate a random matrix
            let mut matrix_b = Matrix::randoms(size, size);
    
            let mut total_duration = Duration::new(0, 0);
    
            for _ in 0..num_iterations {
                let start_time = Instant::now();
    
                let _result = matrix_a.dot_multiply(&matrix_b);
    
                let duration = start_time.elapsed();
                total_duration += duration;
            }
    
            let average_duration = total_duration / num_iterations as u32;
    
            println!(
                "Matrix Size: {}, Average Duration: {:?}",
                size, average_duration
            );
        }
    }

    // test_matrix_performance();
    
    // Test Neural Network
    // let inputs = vec![
    //     vec![0.0, 0.0],
    //     vec![0.0, 1.0],
    //     vec![1.0, 0.0],
    //     vec![1.0, 1.0]
    // ];
    // let targets = vec![
    //     vec![0.0],
    //     vec![1.0],
    //     vec![1.0],
    //     vec![0.0]
    // ];
    // let mut network = Network::new(vec![2, 3, 1], 0.5, SIGMOID);
    // network.train(inputs, targets, 1000);

    // // // Test after training
    // println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
    // println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
    // println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
    // println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}
