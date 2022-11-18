

#[derive(Clone, Debug)]
pub struct Gradient {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl Gradient {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
        }
    }

    pub fn init(&mut self, weights_n: usize, biases_n: usize) {
        self.weights = vec![0.0; weights_n];
        self.biases = vec![0.0; biases_n];
    }

    pub fn clear(&mut self) {
        self.weights.fill(0.0);
        self.biases.fill(0.0);
    }
}


use std::ops::MulAssign;

impl MulAssign<f32> for Gradient {
    fn mul_assign(&mut self, rhs: f32) {
        for weight in &mut self.weights { *weight *= rhs }
        for bias in &mut self.biases { *bias *= rhs }
    }
}