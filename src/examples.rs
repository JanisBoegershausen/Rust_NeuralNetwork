use rand::Rng;

/// Run example 1
/// Network learns to invert two inputs using a set of training data
fn run_example_one(trainer : &mut super::trainer::Trainer) -> super::trainer::TrainingResult {
    let network = super::network::NeuralNetwork::new(vec![2,10,10,2]); // Create a network with a given size for the trainer
    // Create training data
    let mut training_set : Vec<super::trainer::TrainingData> = vec![];
    training_set.push(super::trainer::TrainingData::new(vec![0.0, 0.0], vec![1.0, 1.0]));
    training_set.push(super::trainer::TrainingData::new(vec![1.0, 1.0], vec![0.0, 0.0]));
    training_set.push(super::trainer::TrainingData::new(vec![0.0, 1.0], vec![1.0, 0.0]));
    training_set.push(super::trainer::TrainingData::new(vec![1.0, 0.0], vec![0.0, 1.0]));
    // Run network and save the learning curve in a variable 
    return trainer.train_genetic_algorithm_dataset(&network, &training_set, 200, 100, 1.0, 0.98); 
}

/// Run example 2
/// Network learns to replicate the inputs in its outputs, using a custom evaluation function that returns a score by which the networks are compared. 
pub fn run_example_two(trainer : &mut super::trainer::Trainer) -> super::trainer::TrainingResult {
    let network = super::network::NeuralNetwork::new(vec![2,2]); // Create a network with a given size for the trainer
    let training_set : Vec<super::trainer::TrainingData> = vec![]; // We dont need a training set, because we can calculate the expected output from the input
    // Run network and save the learning curve in a variable 
    return trainer.train_genetic_algorithm_custom(&network, &training_set, 200, 300, 1.0, 0.98, &mut evaluate_copy_input); 
}

/// Evaluate the given network based on how simmilar the outputs are to the inputs and return the score
pub fn evaluate_copy_input(training_set : &Vec<super::trainer::TrainingData>, network : &mut super::network::NeuralNetwork) -> f64{
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

/// Run example 2
/// Shows how to train two networks using a GAN (Generative adversarial network)
pub fn run_example_three(trainer : &mut super::trainer::Trainer) {
    // Define set of training data for the second network to train on
    // Call trainer method that repeats this for x iterations: 
    // Give the second net real data, evaluate the result and modify it using something like back propagation
    // Run the first network
    // Pass the result into the second network, evaulate it and modify it using back prop again
    // Use the outputs of the second network as a score for the first network and modify the first using back prop
}