
use crate::{NeuronLayer, Gradient};



pub trait Optimizer {
    fn update(&mut self, gradients: Vec<Gradient>, layers: &mut Vec<Box<dyn NeuronLayer>>);
}



pub struct OptimizerSGD {
    grandients_buffer: Vec<Vec<Gradient>>,
    batch_size: usize,
    learning_rate: f32,
}

impl OptimizerSGD {
    pub fn new(batch_size: usize, learning_rate: f32) -> Self {
        Self {
            grandients_buffer: Vec::new(),
            batch_size, 
            learning_rate,
        }
    }
}



impl Optimizer for OptimizerSGD {
    
    fn update(&mut self, gradients: Vec<Gradient>, layers: &mut Vec<Box<dyn NeuronLayer>>) {

        self.grandients_buffer.push(gradients);


        if self.grandients_buffer.len() == self.batch_size {

            let mut gradients_buffer = std::mem::take(&mut self.grandients_buffer).into_iter();
            let mut gradients = gradients_buffer.next().unwrap();
            
            for join_gradients in gradients_buffer {
                for (gradient, join_gradient) in gradients.iter_mut().zip(join_gradients) {
                    *gradient += join_gradient;
                }
            }

            for gradient in gradients.iter_mut() {
                *gradient *= -self.learning_rate / self.batch_size as f32
            }


            for layer in layers {
                layer.train(&mut gradients);
            }
        }
    }
}



pub struct OptimizerAdam {
    grandients_buffer: Vec<Vec<Gradient>>,
    batch_size: usize,
    learning_rate: f32,
}

impl OptimizerAdam {
    pub fn new(batch_size: usize, learning_rate: f32) -> Self {
        Self {
            grandients_buffer: Vec::new(),
            batch_size, 
            learning_rate,
        }
    }
}


impl Optimizer for OptimizerAdam {
    
    fn update(&mut self, gradients: Vec<Gradient>, layers: &mut Vec<Box<dyn NeuronLayer>>) {

        self.grandients_buffer.push(gradients);


        if self.grandients_buffer.len() == self.batch_size {

            let mut gradients_buffer = std::mem::take(&mut self.grandients_buffer).into_iter();
            let mut gradients = gradients_buffer.next().unwrap();
            
            for join_gradients in gradients_buffer {
                for (gradient, join_gradient) in gradients.iter_mut().zip(join_gradients) {
                    *gradient += join_gradient;
                }
            }

            for gradient in gradients.iter_mut() {
                *gradient *= -self.learning_rate / self.batch_size as f32
            }


            for layer in layers {
                layer.train(&mut gradients);
            }
        }
    }
}