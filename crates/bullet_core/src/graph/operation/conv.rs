use crate::shape::Shape;

#[derive(Clone, Copy, Debug)]
pub struct ConvolutionDescription {
    pub input_shape: Shape,
    pub input_channels: usize,
    pub output_shape: Shape,
    pub output_channels: usize,
    pub filter_shape: Shape,
    /// Can be (0, 0), which is not a valid shape
    pub padding_shape: (usize, usize),
    pub stride_shape: Shape,
}

impl ConvolutionDescription {
    pub fn new(
        input_shape: Shape,
        input_channels: usize,
        output_channels: usize,
        filter_shape: Shape,
        padding_shape: (usize, usize),
        stride_shape: Shape,
    ) -> Self {
        let hout = (input_shape.rows() + 2 * padding_shape.0 - filter_shape.rows()) / stride_shape.rows() + 1;
        let wout = (input_shape.cols() + 2 * padding_shape.1 - filter_shape.cols()) / stride_shape.cols() + 1;

        Self {
            input_shape,
            input_channels,
            output_shape: Shape::new(hout, wout),
            output_channels,
            filter_shape,
            padding_shape,
            stride_shape,
        }
    }
}
