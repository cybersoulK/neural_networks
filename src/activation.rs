use std::f32::consts::E;

use ndarray::Array1;

use crate::Layer;


pub struct ReLU { size: usize }
impl ReLU { pub fn new() -> Self { Self { size: 0 } } }

impl Layer for ReLU {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Array1<f32>) -> Array1<f32> {
        
        inputs.into_iter().map(|input| {
            if input > 0.0 { input } else { 0.0 }
        }).collect()        
    }
    

    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32> {

        inputs.into_iter().enumerate().map(|(i, input)| {
            (if input > 0.0 { 1.0 } else { 0.0 }) * d_inputs[i]
        }).collect()
    }
}



pub struct Sigmoid { size: usize }
impl Sigmoid { pub fn new() -> Self { Self { size: 0 } } }

impl Layer for Sigmoid {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Array1<f32>) -> Array1<f32> {

        todo!()     
    }

    
    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32> {

        todo!()
    }
}



pub struct Softmax { size: usize }
impl Softmax { pub fn new() -> Self { Self { size: 0 } } }

impl Layer for Softmax {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Array1<f32>) -> Array1<f32> {
        
        let max = inputs.iter().copied().map(|float| float).max_by(|a, b| a.total_cmp(b)).unwrap();

        let values = inputs.into_iter().map(|input| {
            E.powf(input - max)
        }).collect::<Array1<f32>>();

        let sum = values.sum();

        values.into_iter().map(|value| {
            value / sum
        }).collect()
    }


    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32> {
        
        let values = self.forward(inputs);

        values.iter().enumerate().map(|(i, value)| {

            values.iter().enumerate().map(|(i_2, value_2)| {
                (if i == i_2 {
                    value_2 * (1.0 - value)
                }
                else {
                    -value_2 * value
                }) * d_inputs[i_2]
            }).sum::<f32>()

        }).collect()
    }
}