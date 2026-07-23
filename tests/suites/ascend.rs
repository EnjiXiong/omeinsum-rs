use num_complex::Complex32;
use omeinsum::{
    realify_einsum, split_re_im, Ascend, Cpu, Einsum, RealifiedOutput, RealifyInput, Standard,
    Tensor,
};

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

#[test]
fn rank_nine_operand_permutation_falls_back_with_cpu_parity() {
    let input_modes: Vec<usize> = (0..9).collect();
    let output_modes: Vec<usize> = (1..9).collect();
    let sizes = (0..9).map(|mode| (mode, 2)).collect();
    let mut einsum = Einsum::new(vec![input_modes, vec![0]], output_modes, sizes);
    einsum.optimize_greedy();
    let input = vec![1.0f32; 512];
    let vector = vec![1.0f32; 2];

    let cpu_tensors = [
        Tensor::from_data_with_backend(&input, &[2; 9], Cpu),
        Tensor::from_data_with_backend(&vector, &[2], Cpu),
    ];
    let cpu_refs = cpu_tensors.iter().collect::<Vec<_>>();
    let expected = einsum
        .execute::<Standard<f32>, f32, Cpu>(&cpu_refs)
        .to_vec();

    let ascend = Ascend::new().expect("initialize Ascend device 0");
    let ascend_tensors = [
        Tensor::from_data_with_backend(&input, &[2; 9], ascend.clone()),
        Tensor::from_data_with_backend(&vector, &[2], ascend.clone()),
    ];
    let ascend_refs = ascend_tensors.iter().collect::<Vec<_>>();
    let actual = einsum
        .execute::<Standard<f32>, f32, Ascend>(&ascend_refs)
        .to_vec();

    assert_eq!(actual, expected);
    assert_eq!(actual, vec![2.0; 256]);
}
