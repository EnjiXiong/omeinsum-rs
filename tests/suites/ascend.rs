use num_complex::Complex32;
use omeinsum::{realify_einsum, split_re_im, Ascend, RealifiedOutput, RealifyInput};

#[test]
fn realify_c32_matmul_runs_as_f32_on_ascend() {
    let a = [
        Complex32::new(1.0, 2.0),
        Complex32::new(0.0, -1.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, -1.0),
    ];
    let b = [
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, -1.0),
        Complex32::new(0.0, 1.0),
        Complex32::new(4.0, 0.0),
    ];
    let inputs = [
        RealifyInput::Complex {
            data: &a,
            shape: &[2, 2],
            conjugate: false,
        },
        RealifyInput::Complex {
            data: &b,
            shape: &[2, 2],
            conjugate: false,
        },
    ];
    let backend = Ascend::new().expect("initialize Ascend device 0");

    let (result, output) = realify_einsum(&inputs, &[vec![0, 1], vec![1, 2]], &[0, 2], backend);

    assert_eq!(output, RealifiedOutput::ReImAxis);
    assert_eq!(result.shape(), &[2, 2, 2]);
    let (re, im) = split_re_im(&result);
    assert_eq!(re, vec![5.0, 1.0, 10.0, 9.0]);
    assert_eq!(im, vec![1.0, -5.0, 1.0, -4.0]);
}
