
use crate::{Optimizer, Layer, Loss};



pub struct NeuronNetworkConfig {
    ///(min, max)
    pub weights_init: (f32, f32), //better weight initialization enum for different techniques
    pub loss: Loss,
    pub optimizer: Optimizer, 
}


pub struct NeuronNetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    config: NeuronNetworkConfig,
}

impl NeuronNetworkBuilder {
    pub fn new(config: NeuronNetworkConfig) -> Self {
        Self {
            layers: Vec::new(),
            config,
        }
    }

    pub fn add<NL: Layer + 'static>(mut self, layer: NL) -> Self {

        self.layers.push(Box::new(layer));
        self
    }

    pub fn build(mut self) -> Result<NeuronNetwork, String> {

        if self.layers.len() < 2 { return Err("expected at least two layers".to_owned()) }

        let first_layer = self.layers.get(0).unwrap();
        if first_layer.is_inputs_layer() == false { return Err("first layer must be Inputs".to_owned()) }


        let NeuronNetworkConfig { weights_init, loss, optimizer } = self.config;
        

        let mut input_size = first_layer.output_size();
        let (weights_min, weights_max) = weights_init;

        for layer in self.layers.iter_mut().skip(1) {
            layer.init(input_size, weights_min, weights_max);
            input_size = layer.output_size();
        }


        Ok(NeuronNetwork { 
            layers: self.layers,
            loss,
            optimizer,
        })
    }
}



pub struct NeuronNetwork {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Optimizer,
    loss: Loss,
}

impl NeuronNetwork {

    pub fn forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        
        let inputs_size = self.layers.get(0).unwrap().output_size();
        if inputs.len() != inputs_size { panic!("inputs must be length {}, but {} were sent", inputs_size, inputs.len()) }


        let mut inputs = inputs.into();

        for layer in self.layers.iter_mut() {
            let outputs = layer.forward(inputs);
            inputs = outputs;
        }

        inputs.to_vec()
    }


    pub fn train(&mut self, inputs: Vec<f32>, one_hot: Vec<f32>) -> (Vec<f32>, f32) {
        
        let inputs_size = self.layers.get(0).unwrap().output_size();
        if inputs.len() != inputs_size { panic!("inputs must be length {}, but {} were sent", inputs_size, inputs.len()) }


        let mut inputs = inputs.into();
        let mut inputs_vec = Vec::new();

        for layer in self.layers.iter_mut() {
            let outputs = layer.forward(inputs);
            inputs = outputs;
            inputs_vec.push(inputs.clone());
        }

        let outputs = inputs.to_vec();

        
        let (loss, mut d_inputs, ) = match &self.loss {

            Loss::CrossEntropy(loss) => (
                loss.forward(inputs.clone(), one_hot.clone()).sum(), 
                loss.backward(inputs_vec.pop().unwrap(), one_hot)
            ),
        };


        for layer in self.layers.iter_mut().skip(1).rev() {
            let d_outputs = layer.backward(inputs_vec.pop().unwrap(), d_inputs);
            d_inputs = d_outputs;
        }


        match &mut self.optimizer {
            Optimizer::OptimizerSGD(optimizer) => optimizer.update(&mut self.layers),
        }


        (outputs, loss)
    }
}
