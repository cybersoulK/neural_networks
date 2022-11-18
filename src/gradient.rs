use ndarray::Array1;


#[derive(Clone, Debug)]
pub struct Gradient {
    pub weights: Array1<f32>,
    pub biases: Array1<f32>,
}

impl Gradient {
    pub fn new() -> Self {
        Self {
            weights: vec![].into(),
            biases: vec![].into(),
        }
    }

    pub fn init(&mut self, weights_n: usize, biases_n: usize) {
        self.weights = vec![0.0; weights_n].into();
        self.biases = vec![0.0; biases_n].into();
    }

    pub fn clear(&mut self) {
        self.weights.fill(0.0);
        self.biases.fill(0.0);
    }
}


use std::ops::MulAssign;

impl MulAssign<f32> for Gradient {
    fn mul_assign(&mut self, rhs: f32) {
        self.weights *= rhs;
        self.biases *= rhs;
    }
}


use std::ops::DivAssign;

impl DivAssign<f32> for Gradient {
    fn div_assign(&mut self, rhs: f32) {
        self.weights /= rhs;
        self.biases /= rhs;
    }
}