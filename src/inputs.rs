use ndarray::{Array1, array};

use crate::Layer;


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

impl Layer for Inputs {
    fn init(&mut self, _weights_n: usize, _min: f32, _max: f32) {}

    fn output_size(&self) -> usize {
        self.n
    }

    fn is_inputs_layer(&self) -> bool {
        true
    }


    fn forward(&mut self, inputs: Array1<f32>) -> Array1<f32> {
        inputs.clone()
    }

    fn backward(&mut self, _inputs: Array1<f32>, _d_inputs: Array1<f32>) -> Array1<f32> { array![] }
}