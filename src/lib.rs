mod layers;
mod model;
mod optimizer;

#[cfg(test)]
mod tests {
    use crate::layers::*;
    use poro::central::*;

    #[test]
    fn linear_module() {
        let mut linear = LinearLayer::new(3, 1);

        let inputs = vec![
            vec![2.0f32, 3.0, -1.0],
            vec![3.0, -1.0, 0.5],
            vec![0.5, 1.0, 1.0],
            vec![1.0, 1.0, -1.0],
        ];

        let inputs_as_tensor = Tensor::from_vec(
            inputs.iter().flatten().map(|x| *x).collect(),
            vec![4, 3].into(),
        );

        let outputs = vec![1.0f32, -1.0, -1.0, 1.0];
        let outputs_as_tensor =
            Tensor::from_vec(outputs.iter().map(|x| *x).collect(), vec![4, 1].into());

        for _ in 0..50 {
            zero_all_grads();
            let prediction = linear.forward(&inputs_as_tensor);
            let loss = (prediction - outputs_as_tensor).pow(2.0);
            loss.backward();
            update_parameters(-0.01);
        }
    }

    #[test]
    fn batch_norm_test () {

        let batch_size = 32;
        let block_size = 3;
        let vocab_size = 100;
        let n_embd = 10;
        let n_hidden = 100;

        let mut C = Tensor::randn(Shape::new(vec![vocab_size, n_embd]));

        let layers: Vec<Box<dyn Module>> = vec![
            LinearLayer::new(n_embd, n_hidden).into(),    BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, vocab_size).into(),BatchNorm1d::new(vocab_size).into(),
        ];


    }
}
