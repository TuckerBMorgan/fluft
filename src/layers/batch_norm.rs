use poro::{central::Tensor, Shape};
use super::Module;

pub struct BatchNorm1d {
    gain: Tensor,
    bias: Tensor,
}

impl BatchNorm1d {
    #[allow(unused)]
    pub fn new(number_of_weights: usize) -> BatchNorm1d {
        // Initialize the gain and bias tensors
        let gain = Tensor::ones(Shape::new(vec![number_of_weights]));
        let bias = Tensor::zeroes(Shape::new(vec![number_of_weights]));
        BatchNorm1d { gain, bias }
    }
}

impl Module for BatchNorm1d {
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

impl From<BatchNorm1d> for Box<dyn Module> {
    fn from(layer: BatchNorm1d) -> Box<dyn Module> {
        Box::new(layer)
    }
}