use std::f32::consts::E;

use ndarray::Array1;



pub enum Loss {
    CrossEntropy(CrossEntropy),
}

impl From<CrossEntropy> for Loss {
    fn from(value: CrossEntropy) -> Self {
        Self::CrossEntropy(value)
    }
}



pub struct CrossEntropy;

impl CrossEntropy {

    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, inputs: Array1<f32>, one_hot: Vec<f32>) -> Array1<f32> {

        inputs.into_iter().zip(one_hot).map(|(input, one_hot)| {

            let input = input.max(10.0_f32.powi(-8));

            -1.0 * input.log(E) * one_hot
        }).collect()
    }

    pub fn backward(&self, inputs: Array1<f32>, one_hot: Vec<f32>) -> Array1<f32> {

        inputs.into_iter().zip(one_hot).map(|(input, one_hot)| {

            let input = input.max(10.0_f32.powi(-8));

            -1.0 * (1.0 / input) * one_hot
        }).collect()
    }
}