
pub enum Loss { //TODO loss can be a layer added at the end by neuron network
    CrossEntropy,
}


pub mod cross_entropy {
    use std::f32::consts::E;

    pub fn forward(inputs: Vec<f32>, one_hot: Vec<f32>) -> Vec<f32> {

        inputs.into_iter().zip(one_hot).map(|(input, one_hot)| {

            let input = input.max(10.0_f32.powi(-8));

            -1.0 * input.log(E) * one_hot
        }).collect()
    }

    pub fn backward(inputs: Vec<f32>, one_hot: Vec<f32>) -> Vec<f32> {

        inputs.into_iter().zip(one_hot).map(|(input, one_hot)| {

            let input = input.max(10.0_f32.powi(-8));

            -1.0 * (1.0 / input) * one_hot
        }).collect()
    }
}
