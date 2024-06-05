use poro::central::Tensor;

pub trait Optimizer {
    fn step(&mut self, parameters: &mut Vec<Tensor>);
}
