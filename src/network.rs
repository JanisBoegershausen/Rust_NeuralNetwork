use rand::Rng;

pub struct NeuralNetwork {
    pub nodes : Vec<Vec<f64>>,
    pub weights : Vec<Vec<Vec<f64>>>
}

impl NeuralNetwork {
    pub fn new(structure : Vec<usize>) -> NeuralNetwork{
        let mut net = NeuralNetwork{
            nodes: vec![],
            weights: vec![]
        };

        net.initialize(structure);

        return net;
    }

    pub fn set_weights(&mut self, new_weights : Vec<Vec<Vec<f64>>>) {
        let mut structure = vec![];
        for l in 0..new_weights.len() {
            structure.push(new_weights[l].len());
        }
        structure.push(new_weights[new_weights.len()-1][0].len());
        //println!("Set Weights: {}", new_weights.len());
        self.initialize(structure);
        self.weights = new_weights;
    }

    pub fn set_inputs(&mut self, input : Vec<f64>) {
        if input.len() != self.nodes[0].len() {
            eprintln!("{}", "Input vector does not match the length of the input layer.");
        }
        self.nodes[0] = input.clone();
    }

    pub fn get_outputs(&mut self) -> Vec<f64> {
        //println!("Len: {}", self.nodes.len());
        return self.nodes[self.nodes.len()-1].clone();
    }

    fn initialize(&mut self, structure : Vec<usize>) {
        //println!("Initializing with structure: {0} -> {1}, ...", structure.len(), structure[0]);

        self.nodes = vec![];
        for l in 0..structure.len() {
            self.nodes.push(vec![0.0; structure[l]]);
        }

        self.weights = vec![];
        for l in 0..structure.len()-1 {
            self.weights.push(vec![vec![]; structure[l]]);
            for n in 0..structure[l] {
                self.weights[l][n] = vec![1.0; structure[l+1]];
            }
        }
    }

    pub fn calculate_network(&mut self) {
        for i in 1..self.nodes.len() {
            self.calculate_layer_values(i);
        }
    }

    pub fn calculate_layer_values(&mut self, layer_index : usize) {
        for i in 0..self.nodes[layer_index].len() {
            self.calculate_node_value(layer_index, i);
        }
    }

    pub fn calculate_node_value(&mut self, layer_index : usize, node_index : usize) {
        let mut sum = 0.0;
        for i in 0..self.nodes[layer_index-1].len() {
            sum += self.nodes[layer_index-1][i] * self.weights[layer_index-1][i][node_index];
        }
        //let average = sum / self.nodes[layer_index-1].len() as f64;
        let value = NeuralNetwork::sigmoid(sum);
        self.nodes[layer_index][node_index] = value;
    }

    pub fn sigmoid(v : f64) -> f64 {
        return 1.0 / (1.0 + std::f64::consts::E.powf(-v));
    }

    pub fn linear(v : f64) -> f64 {
        return v;
    }

    pub fn mutate(&mut self, mutation : f64) {
        let mut rng = rand::thread_rng();

        for l in 0..self.weights.len() {
            for n in 0..self.weights[l].len() {
                for t in 0..self.weights[l][n].len() {
                    self.weights[l][n][t] += rng.gen_range(-mutation, mutation);
                }
            }
        }
    }
}