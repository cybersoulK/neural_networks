use neural_networks::{*, utils::Console};

use::neural_networks::utils::csv_loader;


fn load_alphabet(filename: &str) -> Vec<(String, Vec<f32>)> {

    let data = csv_loader(filename).unwrap();

    let mut alphabet = data.into_iter().skip(1).map(|row| {

        let mut row_iter = row.into_iter();

        let i = row_iter.next().unwrap().parse::<u32>().unwrap();
        let label = char::from_u32('a' as u32 + i).unwrap().to_string();

        (
            label,
            row_iter.map(|pixel| pixel.parse::<f32>().unwrap() / 255.0).collect::<Vec<_>>()
        )
    }).collect::<Vec<_>>();


    use rand::thread_rng;
    use rand::seq::SliceRandom;

    alphabet.shuffle(&mut thread_rng());

    alphabet
}

fn display_alphabet(data: Vec<f32>, label: String, predicted_label: String) {

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

    let labels = (0..26).map(|i| {
        char::from_u32('a' as u32 + i).unwrap().to_string()
    }).collect::<Vec<_>>();

    let alphabet = load_alphabet("alphabet.csv");

    let train_test_division = (alphabet.len() as f32 * 0.9) as usize;
    let mut alphabet_train = alphabet[0..train_test_division].to_vec();
    let alphabet_test = alphabet[train_test_division..alphabet.len()].to_vec();


    let config = NeuronNetworkConfig { 
        weights_init: (-0.3, 0.3),
        loss: CrossEntropy::new().into(),
        optimizer: OptimizerSGD::new(16, 0.0001).into(),
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

    let mut train_correct_n = 0;
    let mut train_loss = 0.0;


    let mut test_i = 0;


    let console = Console::new();

    console.run(move |command, params| {

        match command.as_str() {
            "train" => {

                let train_n = (params.get(0).cloned().unwrap_or("1".to_owned()).parse::<f32>().unwrap_or(1.0) * alphabet_train.len() as f32) as usize;

                for _ in 0..train_n {

                    train_i += 1;

                    if train_i >= alphabet_train.len() { 

                        epoch += 1.0;

                        println!("epoch: {epoch}, training accuracy: {}, avg loss: {}", train_correct_n as f32 / train_i as f32, train_loss / train_i as f32);

                        train_i = 0;
                        train_correct_n = 0;
                        train_loss = 0.0;


                        use rand::thread_rng;
                        use rand::seq::SliceRandom;

                        alphabet_train.shuffle(&mut thread_rng());
                    }

                    let (label, data) = alphabet_train[train_i].clone();
        
                    let index = labels.binary_search(&label).unwrap();
                    let mut one_hot = vec![0.0; labels.len()];
                    one_hot[index] = 1.0;
            

                    let (outputs, loss) = neuron_network.train(data, one_hot);


                    let (predicted_index, _) = outputs.into_iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

                    let index = labels.binary_search(&label).unwrap();
                    if index == predicted_index { train_correct_n += 1 }

                    train_loss += loss;
                }
            }

            "test" => {

                let test_n = params.get(0).cloned().unwrap_or(alphabet_test.len().to_string()).parse::<usize>().unwrap_or(alphabet_test.len());
                let is_display = params.get(1).cloned().unwrap_or("false".to_owned()).parse::<bool>().unwrap_or(false);

  
                let mut test_correct_n = 0;

                for _ in 0..test_n {

                    test_i += 1;
                    if test_i >= alphabet_test.len() { test_i = 0 }
                    
                    let (label, data) = alphabet_test[test_i].clone();

                    let outputs = neuron_network.forward(data.clone());

   
                    let (predicted_index, _) = outputs.into_iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

                    let index = labels.binary_search(&label).unwrap();

                    if index == predicted_index { test_correct_n += 1 }


                    let precicted_label = labels[predicted_index].clone();

                    if is_display { display_alphabet(data, label, precicted_label) }
                }

                println!("tested {test_n}, accurracy: {}", test_correct_n as f32 / test_n as f32);
            }

            _ => println!("not a valid command")
        }
    });
}