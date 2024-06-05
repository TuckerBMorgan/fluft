// This could maybe be its own lib, along with model.rs
use poro::{central::Tensor, Shape};

pub trait Module {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn get_parameters(&self) -> Vec<Tensor>;
}

pub struct LinearLayer {
    weights: Tensor,
    bias: Tensor,
}

impl LinearLayer {
    #[allow(unused)]
    pub fn new(number_of_inputs: usize, number_of_weights: usize) -> LinearLayer {
        // Initialize the weights and bias tensors
        let weights = Tensor::randn(Shape::new(vec![number_of_inputs, number_of_weights]));
        let bias = Tensor::ones(Shape::new(vec![number_of_weights]));
        LinearLayer { weights, bias }
    }
}

impl Module for LinearLayer {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * weights + bias
        (*x << self.weights) + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

pub struct BatchNorm {
    gain: Tensor,
    bias: Tensor,
}

impl Module for BatchNorm {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        // Perform the forward pass: x * gain + bias
        let batch_mean = x.mean(0);
        let batch_variance = x.variance(0);
        let normalized = (*x - batch_mean) / batch_variance;
        normalized * self.gain + self.bias
    }

    fn get_parameters(&self) -> Vec<Tensor> {
        vec![self.gain.clone(), self.bias.clone()]
    }
}
