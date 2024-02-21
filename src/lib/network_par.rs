use super::{matrix_par::MatrixPar, activations::Activation};
use std::time::Instant;
use rayon::prelude::*;

pub struct NetworkPar<'a> {
    layers: Vec<usize>,
    weights: Vec<MatrixPar>,
    biases: Vec<MatrixPar>,
    data: Vec<MatrixPar>,
    learning_rate: f64,
    activation: Activation<'a>
}

impl NetworkPar<'_> {
    // The will be a list of the size of each layer
    // example, 100-200-200-200-10 = [100,200,200,200,10]
    pub fn new<'a>(layers: Vec<usize>, learning_rate: f64, activation: Activation<'a>) -> NetworkPar {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in  0..layers.len() - 1 { //first layer doesn't have weight
            weights.push(MatrixPar::randoms(layers[i+1], layers[i]));
            biases.push(MatrixPar::randoms(layers[i+1], 1));
        }
        NetworkPar {
            layers: layers,
            weights: weights,
            biases: biases,
            data: vec![],
            learning_rate,
            activation
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Invalid number of inputs")
        }
        let mut current = MatrixPar::from(vec![inputs]).transpose();
        self.data = vec![current.clone()];

        for i in 0..self.layers.len() -1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        current.transpose().data[0].to_owned()
    }

    pub fn back_prop(&mut self, outputs: Vec<f64>,targets: Vec<f64>) {
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("Invald number of target")
        }

        let mut parsed = MatrixPar::from(vec![outputs]).transpose();
        let mut errors = MatrixPar::from(vec![targets]).transpose().subtract(&parsed);
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
		let start_time = Instant::now();
        
        for i in 1..=epochs {
            let epoch_start_time = Instant::now();
			if epochs < 100 || i % (epochs / 10) == 0 {
                println!("Epoch {} of {} - Time: {:?}", i, epochs, epoch_start_time.duration_since(start_time));
            }
			for j in 0..inputs.len() {
				let outputs = self.feed_forward(inputs[j].clone());
				self.back_prop(outputs, targets[j].clone());
			}
		}
        let end_time = start_time.elapsed();
        println!("Training complete. Total time: {:?}", end_time);
	}

    pub fn test(&mut self, test_inputs: Vec<Vec<f64>>, test_targets: Vec<Vec<f64>>) {
        let mut correct_predictions = 0;
        let total_tests = test_inputs.len();

        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let predicted_outputs = self.feed_forward(input.clone());
            let predicted_label = predicted_outputs
                .par_iter()
                .position(|&x| x == predicted_outputs.iter().cloned().fold(0. / 0., f64::max))
                .unwrap();
            
            let true_label = target
                .par_iter()
                .position(|&x| x == target.iter().cloned().fold(0. / 0., f64::max))
                .unwrap();

            if predicted_label == true_label {
                correct_predictions += 1;
            }
        }

        let accuracy = correct_predictions as f64 / total_tests as f64;
        println!("Test Accuracy: {:.2}%", accuracy * 100.0);
    }
}