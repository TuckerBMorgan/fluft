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

    use std::collections::HashMap;

    fn build_dataset_from_subset(
        words: &[String],
        stoi: &HashMap<char, usize>,
    ) -> (Vec<[usize; 3]>, Vec<usize>) {
        let mut xs = vec![];
        let mut ys = vec![];
        for word in words {
            let fixed = String::from("...") + word + ".";
            let chars: Vec<char> = fixed.chars().collect();
            for i in 0..chars.len() - 3 {
                let pair = (chars[i], chars[i + 1], chars[i + 2], chars[i + 3]);
                xs.push([stoi[&pair.0], stoi[&pair.1], stoi[&pair.2]]);
                ys.push(stoi[&pair.3]);
            }
        }
        (xs, ys)
    }

    use std::fs::read_to_string;

    fn read_lines(filename: &str) -> Vec<String> {
        let mut result = Vec::new();

        for line in read_to_string(filename).unwrap().lines() {
            result.push(line.to_string())
        }

        result
    }

    #[test]
    fn batch_norm_simple_test() {
        let n_embd = 10;
        let n_hidden = 200;
        let block_size = 3;

        const BATCH_SIZE: usize = 32;
        let names = read_lines("./data/bigram/names.txt");

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();
        let mut i = 0;
        for c in ".abcdefghijklmnopqrstuvwxyz".chars() {
            stoi.insert(c, i);
            itos.insert(i, c);
            i += 1;
        }
        let n1 = (names.len() as f32 * 0.8f32) as usize;
        let n2 = (names.len() as f32 * 0.9f32) as usize;
        let (xtr, ytr) = build_dataset_from_subset(&names[..n1], &stoi);
        let (_xdev, _ydev) = build_dataset_from_subset(&names[n1..n2], &stoi);
        let (_cte, _yte) = build_dataset_from_subset(&names[n2..], &stoi);

        let vocab_size = itos.keys().len();
        let mut c = Tensor::load_from_weight_file("./data/batchnorm/C.json");
        c.set_requires_grad(true);
        let mut w1 = Tensor::load_from_weight_file("./data/batchnorm/W1.json");
        w1.set_requires_grad(true);
        let mut w2 = Tensor::load_from_weight_file("./data/batchnorm/W2.json");
        w2.set_requires_grad(true);
        let mut b2 = Tensor::load_from_weight_file("./data/batchnorm/b2.json");
        b2.set_requires_grad(true);

        let mut bngain = Tensor::load_from_weight_file("./data/batchnorm/bngain.json");
        bngain.set_requires_grad(true);
        let mut bnbiases = Tensor::load_from_weight_file("./data/batchnorm/bnbias.json");
        bnbiases.set_requires_grad(true);

        let mut bnmean_running = Tensor::zeroes(Shape::new(vec![1, n_hidden]));
        bnmean_running.set_requires_grad(true);
        let mut bnvar_running = Tensor::ones(Shape::new(vec![1, n_hidden]));
        bnvar_running.set_requires_grad(true);


        let max_steps = 2;
        let batch_size = 32;

        for i in 0..max_steps {

            zero_all_grads();
            let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![BATCH_SIZE, 3]));
            for b in 0..BATCH_SIZE {
                test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
                test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
                test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
            }
            let test = c.view(Indexable::FromTensor(test_index_tensor.tensor_id));
            let reshape = test.reshape(Shape::new(vec![BATCH_SIZE, 30]));
            let hpreact = reshape << w1;

            let bnmeani = hpreact.mean(0);
            let bnvari = hpreact.std(0);
            let offset = hpreact - bnmeani;
            let numer =  offset * bngain;
            let hpreact = numer / bnvari + bnbiases;

            let h = hpreact.tanh();
            let logits = (h << w2) + b2;

            
            let mut test_ytrue_onehot = Tensor::element(Shape::new(vec![BATCH_SIZE, 27]), 0.0);
            for b in 0..BATCH_SIZE {
                test_ytrue_onehot.set_index([b, ytr[b]].into(), vec![1.0].into());
            }


            let loss = logits.cross_entropy_loss(test_ytrue_onehot);
            println!("Loss: {}", loss.item());

            loss.backward();
            update_parameters(-0.01);
        }
        println!("w1 grad {:?}", w1.grad());
        
        
    }

    use crate::model::{Sequential, Model};
    #[test]
    fn batch_norm_test () {

        let batch_size = 32;
        let block_size = 3;
        let vocab_size = 100;
        let n_embd = 10;
        let n_hidden = 100;
        let names = read_lines("./data/bigram/names.txt");

        let mut stoi = HashMap::new();
        let mut itos = HashMap::new();
        let mut i = 0;
        for c in ".abcdefghijklmnopqrstuvwxyz".chars() {
            stoi.insert(c, i);
            itos.insert(i, c);
            i += 1;
        }
        let n1 = (names.len() as f32 * 0.8f32) as usize;
        let n2 = (names.len() as f32 * 0.9f32) as usize;
        let (xtr, ytr) = build_dataset_from_subset(&names[..n1], &stoi);

        let mut test_index_tensor = Tensor::zeroes(Shape::new(vec![batch_size, 3]));
        for b in 0..batch_size {
            test_index_tensor.set_index([b, 0].into(), vec![xtr[b][0] as f32].into());
            test_index_tensor.set_index([b, 1].into(), vec![xtr[b][1] as f32].into());
            test_index_tensor.set_index([b, 2].into(), vec![xtr[b][2] as f32].into());
        }

        let mut C = Tensor::randn(Shape::new(vec![vocab_size, n_embd]));

        let mut linear_model: Sequential = vec![
            LinearLayer::new(n_embd * block_size, n_hidden).into(),    BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, n_hidden).into(),  BatchNorm1d::new(n_hidden).into(), Tanh::new().into(),
            LinearLayer::new(n_hidden, vocab_size).into(),BatchNorm1d::new(vocab_size).into(),
        ].into();

        let test = C.view(Indexable::FromTensor(test_index_tensor.tensor_id));
        let reshape = test.reshape(Shape::new(vec![32, 30]));

        let output = linear_model.forward(&reshape);
        output.backward();
        update_parameters(-0.01);

    }

    struct EmbeddingLayer {
        weight: Tensor,
    }

    impl EmbeddingLayer {
        pub fn new(number_of_embeddings: usize, embedding_dims: usize) -> Self {
            EmbeddingLayer {
                weight: Tensor::randn(Shape::new(vec![number_of_embeddings, embedding_dims])),
            }
        }
    }

    impl Module for EmbeddingLayer {
        fn forward(&mut self, input: &Tensor) -> Tensor {
            self.weight.view(Indexable::FromTensor(input.tensor_id))
        }

        fn get_parameters(&self) -> Vec<Tensor> {
            vec![self.weight.clone()]
        }
    }

    impl From<EmbeddingLayer> for Box<dyn Module> {
        fn from(layer: EmbeddingLayer) -> Box<dyn Module> {
            Box::new(layer)
        }
    }


    struct FlattenConsecutive {
        block_size: usize,
    }

    impl FlattenConsecutive {
        pub fn new(block_size: usize) -> Self {
            FlattenConsecutive { block_size }
        }
    }

    impl Module for FlattenConsecutive {
        fn forward(&mut self, input: &Tensor) -> Tensor {
            input.reshape(Shape::new(vec![input.shape.number_of_indices / self.block_size, self.block_size]))
        }

        fn get_parameters(&self) -> Vec<Tensor> {
            vec![]
        }
    }

    impl From<FlattenConsecutive> for Box<dyn Module> {
        fn from(layer: FlattenConsecutive) -> Box<dyn Module> {
            Box::new(layer)
        }
    }

    #[test]
    fn wavenet_test() {
        let n_embd = 24;
        let n_hidden = 128;

        let mut model : Sequential = vec![
            
            EmbeddingLayer::new(27, n_embd).into(),
            FlattenConsecutive::new(2).into(),
            LinearLayer::new(n_embd * 2, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),

            FlattenConsecutive::new(2).into(),
            LinearLayer::new(n_hidden * 2, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),

            FlattenConsecutive::new(2).into(),
            LinearLayer::new(n_hidden * 2, n_hidden).into(),
            BatchNorm1d::new(n_hidden).into(),
            Tanh::new().into(),
            LinearLayer::new(n_hidden, 27).into(),
        ].into();
        


    }
}
