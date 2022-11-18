
use crate::Layer;



pub enum Optimizer {
    OptimizerSGD(OptimizerSGD),
}

impl From<OptimizerSGD> for Optimizer {
    fn from(value: OptimizerSGD) -> Self {
        Self::OptimizerSGD(value)
    }
}




pub struct OptimizerSGD {
    batch_size: usize,
    learning_rate: f32,

    steps: usize,
}

impl OptimizerSGD {
    pub fn new(batch_size: usize, learning_rate: f32) -> Self {
        Self {
            batch_size, 
            learning_rate,

            steps: 0,
        }
    }
}



impl OptimizerSGD {
    
    pub fn update(&mut self, layers: &mut Vec<Box<dyn Layer>>) {

        self.steps += 1;

        if self.steps == self.batch_size {
            self.steps = 0;

            for layer in layers {

                if let Some(mut gradient) = layer.take_gradient() {

                    gradient *= -self.learning_rate / self.batch_size as f32;
                    
                    layer.train(gradient);
                }                
            }
        }
    }
}