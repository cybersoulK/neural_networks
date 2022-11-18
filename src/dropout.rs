use ndarray::Array1;

use crate::Layer;


pub struct Dropout {
    ratio: f32,
    size: usize,

    dropout_array: Array1<f32>,
}

impl Dropout {
    pub fn new(ratio: f32) -> Self {
        Self {
            ratio,
            size: 0,

            dropout_array: vec![].into()
        }
    }
}

impl Layer for Dropout {
    fn init(&mut self, inputs_n: usize, min: f32, max: f32) {
        self.size = inputs_n;

        self.dropout_array = vec![0.0; self.size].into();

        for (i, el) in self.dropout_array.iter_mut().enumerate() {

            let perc = i as f32 / self.size as f32;
            if perc > self.ratio { *el = 1.0 }
        }
    }

    fn output_size(&self) -> usize {
        self.size
    }


    fn forward(&mut self, inputs: Array1<f32>) -> Array1<f32> {

        use rand::thread_rng;
        use rand::seq::SliceRandom;

        let mut dropout_array = self.dropout_array.to_vec();
        dropout_array.shuffle(&mut thread_rng());
        self.dropout_array = dropout_array.into();


        inputs * &self.dropout_array
    }

    fn backward(&mut self, inputs: Array1<f32>, d_inputs: Array1<f32>) -> Array1<f32> {

        &self.dropout_array * d_inputs
    }
}