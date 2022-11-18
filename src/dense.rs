use crate::{NeuronLayer, Gradient};


#[derive(Clone)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}



pub struct Dense {
    neurons: Vec<Neuron>,
}

impl Dense {
    pub fn new(n: usize) -> Self {
        Self {
            neurons: vec![Neuron { weights: Vec::new(), bias: 0.0 }; n],
        }
    }
}

impl NeuronLayer for Dense {
    fn init(&mut self, inputs_n: usize, min: f32, max: f32) {

        for Neuron { weights, .. } in &mut self.neurons {
            for _ in 0..inputs_n {
                weights.push(min + (max - min) * rand::random::<f32>())
            }
        }
    }

    fn output_size(&self) -> usize {
        self.neurons.len()   
    }


    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {

        self.neurons.iter().map(|Neuron { weights, bias }| {

            inputs.iter().zip(weights).map(|(input, weight)| {
                input * weight
            }).sum::<f32>() + bias

        }).collect()
    }

    fn backward(&self, inputs: Vec<f32>, d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>) {

        let mut gradient = Gradient::new();

        let mut d_weights = Vec::new();

        for (neuron_i, _) in self.neurons.iter().enumerate() {

            d_weights.extend(inputs.iter().map(|input| {
                input * d_inputs[neuron_i]
            }));

            let d_bias = d_inputs[neuron_i];


            gradient.weights.append(&mut d_weights);
            gradient.biases.push(d_bias);
        }


        let mut d_outputs = vec![0.0; inputs.len()];
        
        for (i, d_output) in d_outputs.iter_mut().enumerate() {
            
            *d_output = self.neurons.iter().enumerate().map(|(neuron_i, Neuron { weights, .. })| {
                weights[i] * d_inputs[neuron_i]
            }).sum::<f32>();
        }


        (d_outputs, Some(gradient))
    }

    fn train(&mut self, gradients: &mut Vec<Gradient>) { 

        let mut gradient = gradients.pop().unwrap();
        gradient.weights.reverse();
        gradient.biases.reverse();

        for Neuron { weights, bias } in self.neurons.iter_mut() {
            
            for weight in weights { *weight += gradient.weights.pop().unwrap(); }
            *bias += gradient.biases.pop().unwrap();
        }
    }
}