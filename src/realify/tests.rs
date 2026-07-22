use std::collections::HashMap;

use num_complex::Complex;

use super::*;
use crate::backend::Cpu;

fn sizes(entries: &[(usize, usize)]) -> HashMap<usize, usize> {
    entries.iter().copied().collect()
}

fn c_index(a: usize, b: usize, c: usize) -> usize {
    a + 2 * b + 4 * c
}

#[test]
fn realify_code_handles_zero_complex_inputs() {
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let iy = vec![0, 2];
    let plan = realify_code(
        &ixs,
        &iy,
        &sizes(&[(0, 2), (1, 3), (2, 4)]),
        &[false, false],
    );

    assert_eq!(plan.einsum.ixs, ixs);
    assert_eq!(plan.einsum.iy, iy);
    assert_eq!(plan.num_mul_vertices, 0);
    assert_eq!(plan.mul_vertex_positions, Vec::<usize>::new());
    assert_eq!(plan.output, RealifiedOutput::Real);
}

#[test]
fn realify_code_identity_does_not_allocate_fresh_labels() {
    let ixs = vec![vec![usize::MAX]];
    let iy = vec![usize::MAX];
    let plan = realify_code(&ixs, &iy, &sizes(&[(usize::MAX, 2)]), &[false]);

    assert_eq!(plan.einsum.ixs, ixs);
    assert_eq!(plan.einsum.iy, iy);
    assert_eq!(plan.output, RealifiedOutput::Real);
}

#[test]
fn realify_code_handles_one_complex_input() {
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let plan = realify_code(
        &ixs,
        &[0, 2],
        &sizes(&[(0, 2), (1, 3), (2, 4)]),
        &[true, false],
    );

    assert_eq!(plan.einsum.ixs, vec![vec![0, 1, 3], vec![1, 2]]);
    assert_eq!(plan.einsum.iy, vec![0, 2, 3]);
    assert_eq!(plan.einsum.size_dict[&3], 2);
    assert_eq!(plan.num_mul_vertices, 0);
    assert_eq!(plan.output, RealifiedOutput::ReImAxis);
}

#[test]
fn realify_code_can_use_usize_max_as_final_fresh_label() {
    let existing = usize::MAX - 1;
    let ixs = vec![vec![existing]];
    let plan = realify_code(&ixs, &[], &sizes(&[(existing, 2)]), &[true]);

    assert_eq!(plan.einsum.ixs, vec![vec![existing, usize::MAX]]);
    assert_eq!(plan.einsum.iy, vec![usize::MAX]);
    assert_eq!(plan.einsum.size_dict[&usize::MAX], 2);
    assert_eq!(plan.output, RealifiedOutput::ReImAxis);
}

#[test]
fn realify_code_uses_free_hole_when_usize_max_is_occupied() {
    let ixs = vec![vec![usize::MAX]];
    let plan = realify_code(&ixs, &[], &sizes(&[(usize::MAX, 2)]), &[true]);

    assert_eq!(plan.einsum.ixs, vec![vec![usize::MAX, 0]]);
    assert_eq!(plan.einsum.iy, vec![0]);
    assert_eq!(plan.einsum.size_dict[&0], 2);
}

#[test]
fn realify_code_wraps_multi_label_allocation_into_free_holes() {
    let existing = usize::MAX - 1;
    let ixs = vec![vec![existing], vec![existing]];
    let plan = realify_code(&ixs, &[], &sizes(&[(existing, 2)]), &[true, true]);

    assert_eq!(
        plan.einsum.ixs,
        vec![
            vec![existing, usize::MAX],
            vec![existing, 0],
            vec![usize::MAX, 0, 1],
        ]
    );
    assert_eq!(plan.einsum.iy, vec![1]);
}

#[test]
fn realify_code_handles_two_complex_inputs() {
    let ixs = vec![vec![0, 1], vec![1, 2]];
    let plan = realify_code(
        &ixs,
        &[0, 2],
        &sizes(&[(0, 2), (1, 3), (2, 4)]),
        &[true, true],
    );

    assert_eq!(
        plan.einsum.ixs,
        vec![vec![0, 1, 3], vec![1, 2, 4], vec![3, 4, 5]]
    );
    assert_eq!(plan.einsum.iy, vec![0, 2, 5]);
    assert_eq!(plan.mul_vertex_positions, vec![2]);
    assert_eq!(plan.num_mul_vertices, 1);
    assert_eq!(plan.einsum.size_dict[&3], 2);
    assert_eq!(plan.einsum.size_dict[&4], 2);
    assert_eq!(plan.einsum.size_dict[&5], 2);
    assert_eq!(plan.output, RealifiedOutput::ReImAxis);
}

#[test]
fn realify_code_handles_three_complex_inputs_left_deep() {
    let ixs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
    let plan = realify_code(
        &ixs,
        &[0, 3],
        &sizes(&[(0, 2), (1, 3), (2, 4), (3, 5)]),
        &[true, true, true],
    );

    assert_eq!(
        plan.einsum.ixs,
        vec![
            vec![0, 1, 4],
            vec![1, 2, 5],
            vec![2, 3, 6],
            vec![4, 5, 7],
            vec![7, 6, 8],
        ]
    );
    assert_eq!(plan.einsum.iy, vec![0, 3, 8]);
    assert_eq!(plan.mul_vertex_positions, vec![3, 4]);
    assert_eq!(plan.num_mul_vertices, 2);
    assert_eq!(plan.output, RealifiedOutput::ReImAxis);
}

#[test]
fn realify_code_fresh_labels_include_sparse_size_dict_keys() {
    let ixs = vec![vec![0, 1]];
    let plan = realify_code(&ixs, &[0], &sizes(&[(0, 2), (1, 2), (100, 7)]), &[true]);

    assert_eq!(plan.einsum.ixs, vec![vec![0, 1, 101]]);
    assert_eq!(plan.einsum.iy, vec![0, 101]);
    assert_eq!(plan.einsum.size_dict[&101], 2);
}

#[test]
fn realify_code_preserves_repeated_labels() {
    let ixs = vec![vec![0, 0]];
    let plan = realify_code(&ixs, &[], &sizes(&[(0, 2)]), &[true]);

    assert_eq!(plan.einsum.ixs, vec![vec![0, 0, 1]]);
    assert_eq!(plan.einsum.iy, vec![1]);
    assert_eq!(plan.num_mul_vertices, 0);
}

#[test]
fn c_data_is_permutation_invariant() {
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let value = constants::C_DATA[c_index(a, b, c)];
                assert_eq!(value, constants::C_DATA[c_index(a, c, b)]);
                assert_eq!(value, constants::C_DATA[c_index(b, a, c)]);
                assert_eq!(value, constants::C_DATA[c_index(b, c, a)]);
                assert_eq!(value, constants::C_DATA[c_index(c, a, b)]);
                assert_eq!(value, constants::C_DATA[c_index(c, b, a)]);
            }
        }
    }
}

#[test]
fn z_tensor_cube_leaves_c_data_invariant() {
    let z = [1.0, -1.0];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let index = c_index(a, b, c);
                assert_eq!(
                    constants::C_DATA[index] * z[a] * z[b] * z[c],
                    constants::C_DATA[index]
                );
            }
        }
    }
}

#[test]
fn m_times_e0_is_identity_on_either_input_leg() {
    for x in 0..2 {
        for y in 0..2 {
            let expected = if x == y { 1.0 } else { 0.0 };
            assert_eq!(constants::M_DATA[c_index(x, 0, y)], expected);
            assert_eq!(constants::M_DATA[c_index(0, x, y)], expected);
        }
    }
}

#[test]
fn realify_data_deinterleaves_column_major_complex_data() {
    let a = vec![
        Complex::new(1.0, 2.0),
        Complex::new(0.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(2.0, -1.0),
    ];

    assert_eq!(
        realify_data(&a, false),
        vec![1.0, 0.0, 3.0, 2.0, 2.0, -1.0, 0.0, -1.0]
    );
    assert_eq!(
        realify_data(&a, true),
        vec![1.0, 0.0, 3.0, 2.0, -2.0, 1.0, -0.0, 1.0]
    );
}

#[test]
fn split_re_im_round_trips_realified_data() {
    let data = vec![
        Complex::new(1.0, 2.0),
        Complex::new(0.0, -1.0),
        Complex::new(3.0, 0.0),
        Complex::new(2.0, -1.0),
    ];
    let realified = realify_data(&data, false);
    let tensor = Tensor::<f64, Cpu>::from_data(&realified, &[2, 2, 2]);

    let (re, im) = split_re_im(&tensor);

    assert_eq!(re, vec![1.0, 0.0, 3.0, 2.0]);
    assert_eq!(im, vec![2.0, -1.0, 0.0, -1.0]);
}

#[test]
fn mul_vertex_tensor_uses_runtime_m_data() {
    let tensor = mul_vertex_tensor::<f32, Cpu>(Cpu);

    assert_eq!(tensor.shape(), &[2, 2, 2]);
    assert_eq!(
        tensor.to_vec(),
        constants::M_DATA.map(|value| value as f32).to_vec()
    );
}
