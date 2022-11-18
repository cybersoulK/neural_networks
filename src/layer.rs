use ndarray::Array1;

use crate::Gradient;


pub trait Layer {
    fn init(&mut self, inputs_n: usize, min: f32, max: f32);
    fn output_size(&self) -> usize;

    fn is_inputs_layer(&self) -> bool { false }

    fn forward(&self, inputs: Array1<f32>) -> Array1<f32>;
    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32>;

    fn take_gradient(&mut self) -> Option<Gradient> { None }
    fn train(&mut self, _gradients: Gradient) {}
}