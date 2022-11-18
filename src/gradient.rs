use std::ops::{MulAssign, AddAssign};
use std::iter::Sum;



#[derive(Debug)]
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
}

impl MulAssign<f32> for Gradient {
    fn mul_assign(&mut self, rhs: f32) {
        for weight in &mut self.weights { *weight *= rhs }
        for bias in &mut self.biases { *bias *= rhs }
    }
}

impl AddAssign<Gradient> for Gradient {
    fn add_assign(&mut self, rhs: Gradient) {
        for (weight, weight_rhs) in self.weights.iter_mut().zip(rhs.weights) { *weight += weight_rhs }
        for (bias, bias_rhs) in self.biases.iter_mut().zip(rhs.biases) { *bias += bias_rhs }
    }
}



impl Sum for Gradient {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut gradient = iter.next().unwrap();

        for join_gradient in iter {
            gradient += join_gradient;
        }

        gradient
    }
}