
use crate::{OptimizerSGD, Optimizer, Gradient};


pub trait NeuronLayer {
    fn init(&mut self, inputs_n: usize, min: f32, max: f32);
    fn output_size(&self) -> usize;

    fn is_inputs_layer(&self) -> bool { false }

    fn forward(&self, inputs: Vec<f32>) -> Vec<f32>;

    fn backward(&self, inputs: Vec<f32>, d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>);
    fn train(&mut self, _gradients: &mut Vec<Gradient>) {}
}



pub struct NeuronNetworkConfig {
    ///(min, max)
    pub weights_init: (f32, f32), //better weight initialization enum for different techniques
}


pub struct NeuronNetworkBuilder {
    layers: Vec<Box<dyn NeuronLayer>>,
    config: NeuronNetworkConfig,
}

impl NeuronNetworkBuilder {
    pub fn new(config: NeuronNetworkConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
        }
    }

    pub fn add<NL: NeuronLayer + 'static>(mut self, layer: NL) -> Self {

        self.layers.push(Box::new(layer));
        self
    }

    pub fn build(mut self) -> Result<NeuronNetwork, String> {

        if self.layers.len() < 2 { return Err("expected at least two layers".to_owned()) }

        
        let first_layer = self.layers.get(0).unwrap();
        if first_layer.is_inputs_layer() == false { return Err("first layer must be Inputs".to_owned()) }

        
        let mut input_size = first_layer.output_size();
        let (weights_min, weights_max) = self.config.weights_init;

        for layer in self.layers.iter_mut().skip(1) {
            layer.init(input_size, weights_min, weights_max);
            input_size = layer.output_size();
        }


        Ok(NeuronNetwork { 
            layers: self.layers,
            optimizer: Box::new(OptimizerSGD::new(16, 0.0001)), //TODO: temp, pick from enum in config
        })
    }
}



pub struct NeuronNetwork {
    layers: Vec<Box<dyn NeuronLayer>>,
    optimizer: Box<dyn Optimizer>,
    //TODO metrics  
}

impl NeuronNetwork {

    pub fn forward(&mut self, mut inputs: Vec<f32>) -> Vec<f32> {
        
        let inputs_size = self.layers.get(0).unwrap().output_size();
        if inputs.len() != inputs_size { panic!("inputs must be length {}, but {} were sent", inputs_size, inputs.len()) }


        for layer in self.layers.iter_mut() {
            let outputs = layer.forward(std::mem::take(&mut inputs));
            inputs = outputs;
        }

        inputs
    }


    pub fn train(&mut self, mut inputs: Vec<f32>, one_hot: Vec<f32>) {
        
        let inputs_size = self.layers.get(0).unwrap().output_size();
        if inputs.len() != inputs_size { panic!("inputs must be length {}, but {} were sent", inputs_size, inputs.len()) }


        let mut inputs_vec = Vec::new();

        for layer in self.layers.iter_mut() {
            inputs = layer.forward(std::mem::take(&mut inputs));
            inputs_vec.push(inputs.clone());
        }


        //let loss = crate::cross_entropy::forward(inputs.clone(), one_hot.clone());
        let mut d_inputs = crate::cross_entropy::backward(inputs_vec.pop().unwrap(), one_hot);


        let mut gradients = Vec::new();

        for layer in self.layers.iter_mut().skip(1).rev() {
            let (d_outputs, gradient) = layer.backward(inputs_vec.pop().unwrap(), d_inputs);
            d_inputs = d_outputs;
  
            gradients.push(gradient);
        }


        let gradients = gradients.into_iter().flatten().collect();

        self.optimizer.update(gradients, &mut self.layers);
    }
}
