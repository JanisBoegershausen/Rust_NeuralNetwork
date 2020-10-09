use rand::Rng;

pub struct NeuralNetwork {
    pub nodes : Vec<Vec<f64>>,
    pub weights : Vec<Vec<Vec<f64>>>,
    pub biases : Vec<Vec<f64>>
}

impl NeuralNetwork {
    /// Create a new Neural-Network with a given structure
    pub fn new(structure : Vec<usize>) -> NeuralNetwork{
        let mut net = NeuralNetwork{
            nodes: vec![],
            weights: vec![],
            biases: vec![]
        };

        net.initialize(structure);

        return net;
    }

    /// Set the weigts of this network
    pub fn set_weights(&mut self, new_weights : Vec<Vec<Vec<f64>>>) {
        self.weights = new_weights;
    }

    /// Set the biases of this network
    pub fn set_biases(&mut self, new_biases : Vec<Vec<f64>>) {
        self.biases = vec![];
        for l in 0..new_biases.len() {
            self.biases.push(vec![]);
            for n in 0..new_biases[l].len() {
                self.biases[l].push(new_biases[l][n].clone());
            }
        }
    }

    // Set the input layer of this network
    pub fn set_inputs(&mut self, input : Vec<f64>) {
        if input.len() != self.nodes[0].len() {
            eprintln!("{0}: {1} -> {2}", "Input vector does not match the length of the input layer.", input.len(), self.nodes[0].len());
        }
        self.nodes[0] = input.clone();
    }

    /// Get the outputs of the network as a vector of the node values in the output layers
    /// This does not run the calculate_network method. It must be run before this. 
    pub fn get_outputs(&mut self) -> Vec<f64> {
        //println!("Len: {}", self.nodes.len());
        return self.nodes[self.nodes.len()-1].clone();
    }

    /// Initialize this network with a given structure
    pub fn initialize(&mut self, structure : Vec<usize>) {
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

        self.biases = vec![];
        for l in 0..structure.len() {
            self.biases.push(vec![]);
            for _n in 0..structure[l] {
                self.biases[l].push(0.0);
            }
        }
    }

    pub fn calculate_network(&mut self) {
        for i in 1..self.nodes.len() {
            self.calculate_layer_values(i);
        }
    }

    fn calculate_layer_values(&mut self, layer_index : usize) {
        for i in 0..self.nodes[layer_index].len() {
            self.calculate_node_value(layer_index, i);
        }
    }

    /// Calculate the value of the node at node_index in the layer layer_index
    fn calculate_node_value(&mut self, layer_index : usize, node_index : usize) {
        let mut sum = 0.0;
        for i in 0..self.nodes[layer_index-1].len() {
            sum += self.nodes[layer_index-1][i] * self.weights[layer_index-1][i][node_index];
        }
        
        let value = NeuralNetwork::sigmoid(sum + self.biases[layer_index][node_index]);
        self.nodes[layer_index][node_index] = value;
    }

    pub fn sigmoid(v : f64) -> f64 {
        return 1.0 / (1.0 + std::f64::consts::E.powf(-v));
    }

    pub fn linear(v : f64) -> f64 {
        return v;
    }

    /// Mutate the weights of this network by a given mutation ammount in both positive and negative directions
    pub fn mutate_weights(&mut self, mutation : f64) {
        let mut rng = rand::thread_rng();

        for l in 0..self.weights.len() {
            for n in 0..self.weights[l].len() {
                for t in 0..self.weights[l][n].len() {
                    self.weights[l][n][t] += rng.gen_range(-mutation, mutation);
                }
            }
        }
    }

    /// Mutate the biases of this network by a given mutation ammount in both positive and negative directions
    pub fn mutate_biases(&mut self, mutation : f64) {
        let mut rng = rand::thread_rng();

        for l in 0..self.weights.len() {
            for n in 0..self.weights[l].len() {
                self.biases[l][n] += rng.gen_range(-mutation, mutation);
            }
        }
    }

    /// Get the structure of the network as a vector of usize, each usize representing a layer as its size
    pub fn get_structure(&self) -> Vec<usize> {
        let mut structure = vec![];
        for l in 0..self.nodes.len() {
            structure.push(self.nodes[l].len());
        }
        return structure;
    }

    pub fn get_structure_from_weights(&mut self, weights : Vec<Vec<Vec<f64>>>) -> Vec<usize> {
        let mut structure = vec![];
        for l in 0..weights.len() {
            structure.push(weights[l].len());
        }
        structure.push(weights[weights.len()-1][0].len());
        return structure;
    }
}