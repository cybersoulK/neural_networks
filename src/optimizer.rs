
use crate::{Layer, Gradient};



pub enum Optimizer {
    OptimizerSGD(OptimizerSGD),
    OptimizerAdam(OptimizerAdam),
}

impl From<OptimizerSGD> for Optimizer {
    fn from(value: OptimizerSGD) -> Self {
        Self::OptimizerSGD(value)
    }
}

impl From<OptimizerAdam> for Optimizer {
    fn from(value: OptimizerAdam) -> Self {
        Self::OptimizerAdam(value)
    }
}



pub struct OptimizerSGD {
    batch_size: usize,
    batch_step: usize,

    learning_rate: f32,
}

impl OptimizerSGD {
    pub fn new(batch_size: usize, learning_rate: f32) -> Self {
        Self {
            batch_size, 
            batch_step: 0,

            learning_rate,
        }
    }
    
    pub fn update(&mut self, layers: &mut Vec<Box<dyn Layer>>) {

        self.batch_step += 1;

        if self.batch_step == self.batch_size {
            self.batch_step = 0;

            for layer in layers {

                if let Some(gradient) = layer.gradient_mut() {

                    *gradient /= self.batch_size as f32;
                    *gradient *= -self.learning_rate;
                    
                    layer.train();
                }                
            }
        }
    }
}





pub struct OptimizerAdam {
    batch_size: usize,
    batch_step: usize,

    learning_rate: f32,
    decay: f32,
    epsilon: f32,
    beta_1: f32,
    beta_2: f32,

    steps: usize,

    cache: Vec<(Gradient, Gradient)>,
}

pub struct OptmizerAdamConfig {
    decay: f32,
    epsilon: f32,
    beta_1: f32,
    beta_2: f32,
}

impl Default for OptmizerAdamConfig {
    fn default() -> Self {
        Self {
            decay: 0.0,
            epsilon: 10.0_f32.powi(-8),
            beta_1: 0.9,
            beta_2: 0.999,
        }
    }
}


impl OptimizerAdam {
    pub fn new(batch_size: usize, learning_rate: f32, config: OptmizerAdamConfig) -> Self {

        let OptmizerAdamConfig { decay, epsilon, beta_1, beta_2 } = config;

        Self {
            batch_size, 
            batch_step: 0,

            learning_rate,
            decay,
            epsilon,
            beta_1,
            beta_2,

            steps: 0,

            cache: Vec::new(),
        }
    }
    
    pub fn update(&mut self, layers: &mut Vec<Box<dyn Layer>>) {

        self.batch_step += 1;

        if self.batch_step == self.batch_size {
            self.batch_step = 0;

            let mut layer_gradient_i = 0;
            for layer in layers {

                if let Some(gradient) = layer.gradient_mut() {
                
                    let (cache, momentum) = if let Some(cache) = self.cache.get_mut(layer_gradient_i) {
                        cache
                    }
                    else {
                        let (mut cache, mut momentum) = (gradient.clone(), gradient.clone());
                        cache.clear();
                        momentum.clear();

                        self.cache.push((cache, momentum));
                        self.cache.last_mut().unwrap()
                    };

                    layer_gradient_i += 1;


                    *gradient /= self.batch_size as f32;


                    momentum.weights = self.beta_1 * &momentum.weights + (1.0 - self.beta_1) * &gradient.weights;
                    momentum.biases = self.beta_1 * &momentum.biases + (1.0 - self.beta_1) * &gradient.biases;

                    let mut momentum_corrected = Gradient::new();
                    momentum_corrected.weights = &momentum.weights / (1.0 - self.beta_1.powi(self.steps as i32 + 1));
                    momentum_corrected.biases = &momentum.biases / (1.0 - self.beta_1.powi(self.steps as i32 + 1));
                    

                    cache.weights = self.beta_2 * &cache.weights + (1.0 - self.beta_2) * (&gradient.weights * &gradient.weights);
                    cache.biases = self.beta_2 * &cache.biases + (1.0 - self.beta_2) * (&gradient.biases * &gradient.biases);

                    let mut cache_corrected = Gradient::new();
                    cache_corrected.weights = &cache.weights / (1.0 - self.beta_2.powi(self.steps as i32 + 1));
                    cache_corrected.biases = &cache.biases / (1.0 - self.beta_2.powi(self.steps as i32 + 1));

        
                    gradient.weights += &(&momentum_corrected.weights / (cache_corrected.weights.mapv(|el| el.sqrt()) + self.epsilon));
                    gradient.biases += &(&momentum_corrected.biases / (cache_corrected.biases.mapv(|el| el.sqrt()) + self.epsilon));


                    let learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.steps as f32));
                    *gradient *= -learning_rate;


                    layer.train();
                }                
            }

            self.steps += 1;
        }
    }

    
}