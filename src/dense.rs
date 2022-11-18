use ndarray::{Array1, array, Array2, Axis};

use crate::{Layer, Gradient};


#[derive(Clone)]
struct Neuron {
    weights: Array1<f32>,
    bias: f32,
}



pub struct Dense {
    neurons: Array1<Neuron>,
    gradient: Gradient,

    input_n: usize,
    weights_grouped: Array2<f32>,
}

impl Dense {
    pub fn new(n: usize) -> Self {
        Self {
            neurons: vec![Neuron { weights: array![], bias: 0.0 }; n].into(),
            gradient: Gradient::new(),

            input_n: 0,
            weights_grouped: vec![[]].into(),
        }
    }
}

impl Layer for Dense {
    fn init(&mut self, inputs_n: usize, min: f32, max: f32) {

        for Neuron { weights, .. } in &mut self.neurons {

            let mut new_weights = Vec::new();

            for _ in 0..inputs_n {
                new_weights.push(min + (max - min) * rand::random::<f32>())
            }

            *weights = new_weights.into();
        }


        self.gradient.init(self.neurons.len() * inputs_n, self.neurons.len());

        
        self.input_n = inputs_n;
        self.calc_weights_grouped();
    }

    fn output_size(&self) -> usize {
        self.neurons.len()   
    }


    fn forward(&self, inputs: Array1<f32>) -> Array1<f32> {

        self.neurons.mapv(|Neuron { weights, bias }| {
            inputs.dot(&weights) + bias
        })
    }

    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32> {

        for neuron_i in 0..self.neurons.len() {

            let weight_i = neuron_i * self.input_n;

            for (i, weight) in (&inputs * d_inputs[neuron_i]).into_iter().enumerate() {
                self.gradient.weights[weight_i + i] += weight;
            }

            self.gradient.biases[neuron_i] += d_inputs[neuron_i];
        }


        (self.weights_grouped.clone() * d_inputs).sum_axis(Axis(1))
    }


    fn take_gradient(&mut self) -> Option<Gradient> {
        let gradient = self.gradient.clone();
        self.gradient.clear();

        Some(gradient)
    }

    fn train(&mut self, mut gradient: Gradient) { 

        gradient.weights.reverse();
        gradient.biases.reverse();

        for Neuron { weights, bias } in self.neurons.iter_mut() {
            
            for weight in weights { *weight += gradient.weights.pop().unwrap(); }
            *bias += gradient.biases.pop().unwrap();
        }

        self.calc_weights_grouped();
    }
}



impl Dense {
    fn calc_weights_grouped(&mut self) {

        let mut weights_grouped = Array2::<f32>::zeros((self.input_n, self.neurons.len()));

        for i in 0..self.input_n {
            for (neuron_i, Neuron { weights, .. }) in self.neurons.iter().enumerate() {
                weights_grouped[[i, neuron_i]] = weights[i];
            }
        }

        self.weights_grouped = weights_grouped;
    }
}


/*

fn backward(&self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> (Array1<f32>, Option<Gradient>) {

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


        let mut d_outputs = Array1::from_vec(v)vec![0.0; inputs.len()];
        
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

*/