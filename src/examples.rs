use rand::Rng;

/// Run example 1
/// Network learns to invert two inputs using a set of training data
fn run_example_one(trainer : &mut super::trainer::Trainer) -> Vec<f64> {
    trainer.net = super::network::NeuralNetwork::new(vec![2,10,10,2]); // Create a network with a given size for the trainer
    trainer.clear_training_data(); // Clear the trainings data (not needet here)
    // Add training data
    trainer.add_training_data(vec![0.0, 0.0], vec![1.0, 1.0]);
    trainer.add_training_data(vec![1.0, 1.0], vec![0.0, 0.0]);
    trainer.add_training_data(vec![0.0, 1.0], vec![1.0, 0.0]);
    trainer.add_training_data(vec![1.0, 0.0], vec![0.0, 1.0]);
    // Run network and save the learning curve in a variable 
    return trainer.train_genetic_algorithm_dataset(200, 100, 1.0, 0.98); 
}

/// Run example 2
/// Network learns to _____, using a custom evaluation function that returns a score by which the networks are compared. 
pub fn run_example_two(trainer : &mut super::trainer::Trainer) -> Vec<f64> {
    trainer.net = super::network::NeuralNetwork::new(vec![2, 2,2]); // Create a network with a given size for the trainer
    // Run network and save the learning curve in a variable 
    return trainer.train_genetic_algorithm_custom(500, 200, 1.0, 0.98, &mut evaluate_copy_input); 
}

/// Evaluate the given network based on how simmilar the outputs are to the inputs
pub fn evaluate_copy_input(trainer : &super::trainer::Trainer, network : &mut super::network::NeuralNetwork) -> f64{
    if network.nodes[0].len() != network.nodes[network.nodes.len()-1].len() {
        println!("{}", "Error: Input layer does not match output layer!");
        return 0.0;
    }

    let mut score :f64 = 0.0;

    for g in 0..20 {
        let mut rng = rand::thread_rng(); // Create random generator instance
        let mut input = vec![];
        for i in 0..network.nodes[0].len() {
            input.push(rng.gen_range(0.0, 1.0));
        }
        network.set_inputs(input.clone());
        network.calculate_network();
        let output = network.get_outputs();

        let mut difference : f64 = 0.0;
        for i in 0..output.len() {
            difference += (input[i] - output[i]).abs();
        }
        score -= difference;
    }
    
    return score;
}