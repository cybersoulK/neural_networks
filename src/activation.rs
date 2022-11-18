use std::f32::consts::E;

use crate::{NeuronLayer, Gradient};


pub struct ReLU { size: usize }
impl ReLU { pub fn new() -> Self { Self { size: 0 } } }

impl NeuronLayer for ReLU {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        
        inputs.into_iter().map(|input| {
            if input > 0.0 { input } else { 0.0 }
        }).collect()        
    }
    

    fn backward(&self, inputs: Vec<f32>, d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>) {

        let d_outputs = inputs.into_iter().enumerate().map(|(i, input)| {
            (if input > 0.0 { 1.0 } else { 0.0 }) * d_inputs[i]
        }).collect();

        (d_outputs, None)
    }
}



pub struct Sigmoid { size: usize }
impl Sigmoid { pub fn new() -> Self { Self { size: 0 } } }

impl NeuronLayer for Sigmoid {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {

        todo!()     
    }

    
    fn backward(&self, inputs: Vec<f32>, d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>) {

        todo!()
    }
}



pub struct Softmax { size: usize }
impl Softmax { pub fn new() -> Self { Self { size: 0 } } }

impl NeuronLayer for Softmax {

    fn init(&mut self, inputs_n: usize, _min: f32, _max: f32) { self.size = inputs_n }
    fn output_size(&self) -> usize { self.size }


    fn forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        
        let max = inputs.iter().copied().map(|float| float).max_by(|a, b| a.total_cmp(b)).unwrap();

        let values = inputs.into_iter().map(|input| {
            E.powf(input - max)
        });

        let sum = values.clone().sum::<f32>();

        values.map(|value| {
            value / sum
        }).collect()
    }


    fn backward(&self, inputs: Vec<f32>, d_inputs: Vec<f32>) -> (Vec<f32>, Option<Gradient>) {
        
        let values = self.forward(inputs);


        let d_outputs = values.iter().enumerate().map(|(i, value)| {

            values.iter().enumerate().map(|(i_2, value_2)| {
                (if i == i_2 {
                    value_2 * (1.0 - value)
                }
                else {
                    -value_2 * value
                }) * d_inputs[i_2]
            }).sum::<f32>()

        }).collect();
       
        (d_outputs, None)
    }
}