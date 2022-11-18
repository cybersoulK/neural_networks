use neural_networks::{*, utils::Console};

use::neural_networks::utils::csv_loader;


fn load_chinese_mnist(filename: &str) -> Vec<(usize, Vec<f32>)> {

    let data = csv_loader(filename).unwrap();

    let mut chinese_mnist = data.into_iter().skip(1).map(|row| {

        let mut row_iter = row.into_iter();

        row_iter.next_back();
        let label = row_iter.next_back().unwrap().parse::<usize>().unwrap();

        (
            label,
            row_iter.map(|pixel| pixel.parse::<f32>().unwrap() / 255.0).collect::<Vec<_>>()
        )
    }).collect::<Vec<_>>();


    use rand::thread_rng;
    use rand::seq::SliceRandom;

    chinese_mnist.shuffle(&mut thread_rng());

    chinese_mnist
}

fn display_chinese_mnist(data: Vec<f32>, predicted_label: String, correct_label: String) {

    println!("predicted: {predicted_label}, correct: {correct_label}");

    for x in 0..64 {
        for y in 0..64 {
            print!("{}, ", if data[y + x * 64] > 0.3 { "#" } else { " " });
        }
        println!();
    }

    println!();
}


fn main() {

    let labels = vec!["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿"].into_iter().map(|str| str.to_owned()).collect::<Vec<_>>();

    let chinese_mnist = load_chinese_mnist("chinese_mnist.csv");

    let train_test_division = (chinese_mnist.len() as f32 * 0.9) as usize;
    let mut chinese_mnist_train = chinese_mnist[0..train_test_division].to_vec();
    let chinese_mnist_test = chinese_mnist[train_test_division..chinese_mnist.len()].to_vec();


    let config = NeuronNetworkConfig { 
        weights_init: (-0.3, 0.3),
        loss: CrossEntropy::new().into(),
        optimizer: OptimizerAdam::new(32, 0.0005, OptmizerAdamConfig::default()).into(),
    };

    let mut neuron_network = NeuronNetworkBuilder::new(config)
        .add(Inputs::new(64 * 64))

        .add(Dense::new(255))
        .add(ReLU::new())

        .add(Dropout::new(0.5))

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

                let train_n = (params.get(0).cloned().unwrap_or("1".to_owned()).parse::<f32>().unwrap_or(1.0) * chinese_mnist_train.len() as f32) as usize;

                for _ in 0..train_n {

                    train_i += 1;

                    if train_i >= chinese_mnist_train.len() { 

                        epoch += 1.0;

                        println!("epoch: {epoch}, training accuracy: {}, avg loss: {}", train_correct_n as f32 / train_i as f32, train_loss / train_i as f32);

                        train_i = 0;
                        train_correct_n = 0;
                        train_loss = 0.0;


                        use rand::thread_rng;
                        use rand::seq::SliceRandom;

                        chinese_mnist_train.shuffle(&mut thread_rng());
                    }

                    let (index, data) = chinese_mnist_train[train_i].clone();
        
                    let mut one_hot = vec![0.0; labels.len()];
                    one_hot[index] = 1.0;
            

                    let (outputs, loss) = neuron_network.train(data, one_hot);


                    let (predicted_index, _) = outputs.into_iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

                    if index == predicted_index { train_correct_n += 1 }

                    train_loss += loss;
                }
            }

            "test" => {

                let test_n = params.get(0).cloned().unwrap_or(chinese_mnist_test.len().to_string()).parse::<usize>().unwrap_or(chinese_mnist_test.len());
                let is_display = params.get(1).cloned().unwrap_or("false".to_owned()).parse::<bool>().unwrap_or(false);

  
                let mut test_correct_n = 0;

                for _ in 0..test_n {

                    test_i += 1;
                    if test_i >= chinese_mnist_test.len() { test_i = 0 }
                    
                    let (index, data) = chinese_mnist_test[test_i].clone();

                    let outputs = neuron_network.forward(data.clone());

   
                    let (predicted_index, _) = outputs.into_iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

                    if index == predicted_index { test_correct_n += 1 }


                    let precicted_label = labels[predicted_index].clone();
                    let correct_label = labels[index].clone();

                    if is_display { display_chinese_mnist(data, precicted_label, correct_label) }
                }

                println!("tested {test_n}, accurracy: {}", test_correct_n as f32 / test_n as f32);
            }

            _ => println!("not a valid command")
        }
    });
}