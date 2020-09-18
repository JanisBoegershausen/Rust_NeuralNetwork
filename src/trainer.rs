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

    pub fn add_training_data(&mut self, input : Vec<f64>, output : Vec<f64>) {
        self.training_set.push( TrainingData {
            input:input,
            output:output
        } )
    }
    
    pub fn clear_training_data(&mut self) {
        self.training_set = vec![];
    }

    pub fn evaluate(&mut self, network : &mut super::network::NeuralNetwork) -> f64 {
        let mut score = 0.0;

        for ts in &self.training_set {
            network.set_inputs(ts.input.clone());
            network.calculate_network();
            let output = network.get_outputs();
            //println!("Net Out: {}", output.len());
            //println!("Dat Out: {}", ts.output.len());
            for n in 0..output.len() {
                //println!("{}", n);
                score -= (ts.output[n] - output[n]).abs();
            }
        }

        return score;
    }

    pub fn train_genetic_algorithm (&mut self, generations : usize, population : usize, mutation_start : f64, mutation_change_mult : f64) -> Vec<f64> {
        let mut parent_network = super::network::NeuralNetwork::new(vec![1,1]);
        parent_network.set_weights(self.net.weights.clone());

        let mut current_net = super::network::NeuralNetwork::new(vec![1,1]);
        let mut current_score = -1000000.0;

        let mut mutation_ammount = mutation_start; 

        let mut generation_scores = vec![];

        println!("Training Network...");
        for _i in 0..generations {
            for _p in 0..population {
                current_net.set_weights(parent_network.weights.clone());
                current_net.mutate(mutation_ammount);
                let score = self.evaluate(&mut current_net);
                if score > current_score {
                    parent_network.set_weights(current_net.weights.clone());
                    current_score = score;
                }
            }
            mutation_ammount *= mutation_change_mult;
            generation_scores.push(current_score);
        }
        println!("Done!");

        self.net.set_weights(current_net.weights.clone());

        return generation_scores;
    }
}

struct TrainingData {
    input: Vec<f64>,
    output: Vec<f64>
}