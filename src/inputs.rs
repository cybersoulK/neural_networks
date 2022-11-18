use crate::{NeuronLayer, Gradient};


pub struct Inputs {
    n: usize
}

impl Inputs {
    pub fn new(n: usize) -> Self {
        Self {
            n
        }
    }
}

impl NeuronLayer for Inputs {
    fn init(&mut self, _weights_n: usize, _min: f32, _max: f32) {}

    fn output_size(&self) -> usize {
        self.n
    }

    fn is_inputs_layer(&self) -> bool {
        true
    }


    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        inputs
    }

    fn backward(&self, _inputs: Vec<f32>, _d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>) { (vec![], None) }
}