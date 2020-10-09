pub struct Trainer {
    
}

impl Trainer {
    pub fn new() -> Trainer {
        let t = Trainer {
            
        };
        return t;
    }

    /// Evaluate a give network with all datasets in the training set and return the score.
    pub fn evaluate_with_training_data(training_set : &Vec<TrainingData>, network : &mut super::network::NeuralNetwork) -> f64 {
        let mut score = 0.0;
        for ts in training_set {
            network.set_inputs(ts.input.clone());
            network.calculate_network();
            let output = network.get_outputs();
            for n in 0..output.len() {
                score -= (ts.output[n] - output[n]).abs();
            }
        }

        return score.clone();
    }

    /// Train the network using a genetic algorithm. Evaluation using the training set.
    pub fn train_genetic_algorithm_dataset (&mut self, network : &super::network::NeuralNetwork, training_set : &Vec<TrainingData>, generations : usize, population : usize, mutation_start : f64, mutation_change_mult : f64) -> TrainingResult {
        return self.train_genetic_algorithm_custom(network, training_set, generations, population, mutation_start, mutation_change_mult, &mut Trainer::evaluate_with_training_data);
    }

    /// Train the network using a genetic algorithm. Evaluation using the given evaluation function.
    pub fn train_genetic_algorithm_custom (&mut self, network : &super::network::NeuralNetwork, training_set : &Vec<TrainingData>,  generations : usize, population : usize, mutation_start : f64, mutation_change_mult : f64, evaluation_function : &mut dyn FnMut(&Vec<TrainingData>, &mut super::network::NeuralNetwork) -> f64) -> TrainingResult {
        let mut parent_network = super::network::NeuralNetwork::new(vec![1,1]);
        parent_network.initialize(network.get_structure());
        parent_network.set_weights(network.weights.clone());
        parent_network.set_biases(network.biases.clone());

        let mut current_net = super::network::NeuralNetwork::new(vec![1,1]);
        let mut current_score = -1000000.0;

        let mut mutation_ammount = mutation_start; 

        let mut generation_scores = vec![];

        println!("Training Network using custom function...");
        for _i in 0..generations {
            for _p in 0..population {
                current_net.initialize(parent_network.get_structure());
                current_net.set_weights(parent_network.weights.clone());
                current_net.set_biases(parent_network.biases.clone());

                current_net.mutate_weights(mutation_ammount);
                current_net.mutate_biases(mutation_ammount);

                let score = evaluation_function(training_set, &mut current_net).clone();
                if score > current_score {
                    parent_network.set_weights(current_net.weights.clone());
                    current_score = score;
                }
            }
            mutation_ammount *= mutation_change_mult;
            generation_scores.push(current_score);

            if _i % (generations / 10) == 0 && _i > 0 {
                println!("{}%", _i*100/generations);
            }
        }
        println!("Done! Final score: {0}", generation_scores[generation_scores.len()-1]);

        let mut output_network = super::network::NeuralNetwork::new(vec![1,1]);
        output_network.initialize(network.get_structure());
        output_network.set_weights(current_net.weights.clone());
        output_network.set_biases(current_net.biases.clone());

        return TrainingResult {
            network:output_network,
            generation_score_curve: generation_scores
        };
    }

    pub fn train_backpropagation (&mut self, network : &super::network::NeuralNetwork, training_set : &Vec<TrainingData>,  generations : usize) -> TrainingResult {
        self.calculate_errors(network, vec![0.0; network.nodes[network.nodes.len()-1].len()]);
        return TrainingResult{
            network: super::network::NeuralNetwork::new(vec![2,2]),
            generation_score_curve: vec![0.0]
        }
    }

    pub fn calculate_errors(&mut self, network : &super::network::NeuralNetwork, expected_outputs : Vec<f64>) -> Vec<Vec<f64>> {
        let mut errors : Vec<Vec<f64>> = network.nodes.clone(); // Clone the network nodes to get a matrix in the correct size

        let last_layer_index = errors.len()-1;
        // Calculate output errors
        for n in 0..errors[last_layer_index].len() {
            errors[last_layer_index][n] = expected_outputs[n] - network.nodes[network.nodes.len()-1][n];
        }
        
        for layer in 0..errors.len()-1 {
            for node in 0..errors[layer].len() {
                for w in 0..network.weights[layer][node].len() {
                    let mut weightSum = 0.0;
                    for n in 0..network.weights[layer].len() {
                        weightSum += network.weights[layer][n][w]
                    }
                    errors[layer][node] = (network.weights[layer][node][w] / weightSum) * errors[layer+1][w];
                }
            }
        }
        

        return errors;
    }
}

/// Data for training a network. Contains the inputs and the expected outputs for this input.
pub struct TrainingData {
    input: Vec<f64>,
    output: Vec<f64>
}

/// Result of training a network. Contains the trained network and a curve of the scores over all training generations.
pub struct TrainingResult {
    pub network : super::network::NeuralNetwork,
    pub generation_score_curve : Vec<f64>
}

/// Data containing one set of inputs and the expected outputs for them. 
impl TrainingData {
    pub fn new(inputs : Vec<f64>, output : Vec<f64>) -> TrainingData {
        return TrainingData {
            input: inputs,
            output: output
        }
    }
}