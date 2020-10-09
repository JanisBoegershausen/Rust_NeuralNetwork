pub struct Trainer {
    pub net : super::network::NeuralNetwork,
    training_set : Vec<TrainingData>
}

impl Trainer {
    pub fn new() -> Trainer {
        let t = Trainer {
            net : super::network::NeuralNetwork::new(vec![1,1]),
            training_set : vec![]
        };

        return t;
    }

    /// Add training data to the current training set. 
    pub fn add_training_data(&mut self, input : Vec<f64>, output : Vec<f64>) {
        self.training_set.push( TrainingData {
            input:input,
            output:output
        } )
    }
    
    /// Delete all training data from the current set.
    pub fn clear_training_data(&mut self) {
        self.training_set = vec![];
    }

    /// Evaluate a give network with all datasets in the current training set and return the score
    pub fn evaluate_with_training_data(trainer : &Trainer, network : &mut super::network::NeuralNetwork) -> f64 {
        let mut score = 0.0;
        for ts in &trainer.training_set {
            network.set_inputs(ts.input.clone());
            network.calculate_network();
            let output = network.get_outputs();
            for n in 0..output.len() {
                score -= (ts.output[n] - output[n]).abs();
            }
        }

        return score.clone();
    }

    /// Train the current network using a genetic algorithm. Evaluation using the current data set
    pub fn train_genetic_algorithm_dataset (&mut self, generations : usize, population : usize, mutation_start : f64, mutation_change_mult : f64) -> Vec<f64> {
        return self.train_genetic_algorithm_custom(generations, population, mutation_start, mutation_change_mult, &mut Trainer::evaluate_with_training_data);
    }

    /// Train the current network using a genetic algorithm. Evaluation using the given evaluation function
    pub fn train_genetic_algorithm_custom (&mut self, generations : usize, population : usize, mutation_start : f64, mutation_change_mult : f64, evaluation_function : &mut dyn FnMut(&Trainer, &mut super::network::NeuralNetwork) -> f64) -> Vec<f64> {
        let mut parent_network = super::network::NeuralNetwork::new(vec![1,1]);
        parent_network.initialize(self.net.get_structure());
        parent_network.set_weights(self.net.weights.clone());
        parent_network.set_biases(self.net.biases.clone());

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

                let score = evaluation_function(&self, &mut current_net).clone();
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

        self.net.set_weights(current_net.weights.clone());
        self.net.set_biases(current_net.biases.clone());

        return generation_scores;
    }
}

/// Data for training a network. Contains the inputs and the expected outputs for this input
pub struct TrainingData {
    input: Vec<f64>,
    output: Vec<f64>
}