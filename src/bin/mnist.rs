use neural_networks::{*, utils::Console};

use::neural_networks::utils::csv_loader;


fn load_mnist(filename: &str) -> Vec<(String, Vec<f32>)> {

    let data = csv_loader(filename).unwrap();

    data.into_iter().skip(1).map(|row| {

        let mut row_iter = row.into_iter();

        let label = row_iter.next().unwrap();

        (
            label,
            row_iter.map(|pixel| pixel.parse::<f32>().unwrap() / 255.0).collect::<Vec<_>>()
        )
    }).collect()
}

fn display_mnist(data: Vec<f32>, label: String, predicted_label: String) {

    println!("predicted: {predicted_label}, correct: {label}");

    for x in 0..28 {
        for y in 0..28 {
            print!("{}, ", if data[y + x * 28] > 0.5 { "#" } else { " " });
        }
        println!();
    }

    println!();
}


fn main() {

    let labels = vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"].into_iter().map(|str| str.to_owned()).collect::<Vec<_>>();

    let mut mnist_train = load_mnist("mnist/mnist_train.csv");
    let mnist_test = load_mnist("mnist/mnist_test.csv");


    let config = NeuronNetworkConfig { 
        weights_init: (-0.3, 0.3),
        loss: CrossEntropy::new().into(),
        optimizer: OptimizerAdam::new(32, 0.0005, OptmizerAdamConfig::default()).into(),
    };

    let mut neuron_network = NeuronNetworkBuilder::new(config)
        .add(Inputs::new(28 * 28))

        .add(Dense::new(255))
        .add(ReLU::new())

        .add(Dense::new(255))
        .add(ReLU::new())

        .add(Dense::new(labels.len()))
        .add(Softmax::new())
        .build().unwrap();


    
    let mut epoch = 0.0_f32;
    let mut train_i = 0;

    let mut test_i = 0;

    let console = Console::new();

    console.run(move |command, params| {

        match command.as_str() {
            "train" => {

                let train_n = (params.get(0).cloned().unwrap_or("1".to_owned()).parse::<f32>().unwrap_or(1.0) * mnist_train.len() as f32) as usize;

                for _ in 0..train_n {

                    train_i += 1;

                    if train_i >= mnist_train.len() { 
                        train_i = 0;

                        use rand::thread_rng;
                        use rand::seq::SliceRandom;

                        mnist_train.shuffle(&mut thread_rng());
                    }

                    let (label, data) = mnist_train[train_i].clone();
        
                    let index = labels.binary_search(&label).unwrap();
                    let mut one_hot = vec![0.0; labels.len()];
                    one_hot[index] = 1.0;
            
                    neuron_network.train(data, one_hot);
                }


                epoch += train_n as f32 / mnist_train.len() as f32;

                println!("trained {train_n}, current epoch: {epoch}");
            }

            "test" => {

                let test_n = params.get(0).cloned().unwrap_or(mnist_test.len().to_string()).parse::<usize>().unwrap_or(mnist_test.len());
                let is_display = params.get(1).cloned().unwrap_or("false".to_owned()).parse::<bool>().unwrap_or(false);

  
                let mut correct_n = 0;

                for _ in 0..test_n {

                    test_i += 1;
                    if test_i >= mnist_test.len() { test_i = 0 }
                    
                    let (label, data) = mnist_test[test_i].clone();

                    let outputs = neuron_network.forward(data.clone());

   
                    let (predicted_index, _) = outputs.into_iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

                    let index = labels.binary_search(&label).unwrap();

                    if index == predicted_index { correct_n += 1 }


                    let precicted_label = labels[predicted_index].clone();

                    if is_display { display_mnist(data, label, precicted_label) }
                }

                println!("tested {test_n}, accurracy: {}", correct_n as f32 / test_n as f32);
            }

            _ => println!("not a valid command")
        }
    });
}