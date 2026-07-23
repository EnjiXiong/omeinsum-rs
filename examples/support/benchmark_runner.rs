use super::format::{slice_column_major, BenchmarkNetwork, TreeNode};
use num_complex::Complex32;
use num_traits::{One, Zero};
use omeinsum::algebra::Scalar;
use omeinsum::{Backend, BackendScalar, Standard, Tensor};
use std::collections::HashMap;

type RealValue<B> = (Tensor<f32, B>, Option<Tensor<f32, B>>);

pub fn assignments(network: &BenchmarkNetwork) -> Vec<HashMap<usize, usize>> {
    (0..network.assignment_count)
        .map(|mut n| {
            let mut a = HashMap::new();
            for &cut in network.cuts.iter().rev() {
                let extent = network.size_dict[&cut];
                a.insert(cut, n % extent);
                n /= extent;
            }
            a
        })
        .collect()
}

pub fn real_cache<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    backend: B,
) -> Vec<Vec<RealValue<B>>>
where
    f32: BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(t, ix)| {
            local_assignments(network, ix)
                .iter()
                .map(|a| {
                    let (re, shape) = slice_column_major(&t.data_re, &t.shape, ix, a);
                    let r = Tensor::from_data_with_backend(&re, &shape, backend.clone());
                    let im = t.structurally_complex.then(|| {
                        let (v, _) = slice_column_major(&t.data_im, &t.shape, ix, a);
                        Tensor::from_data_with_backend(&v, &shape, backend.clone())
                    });
                    (r, im)
                })
                .collect()
        })
        .collect()
}

fn real_walk<B: Backend + Clone>(node: &TreeNode, leaves: &[RealValue<B>]) -> RealValue<B>
where
    f32: BackendScalar<B>,
{
    match node {
        TreeNode::Leaf { tensor_index } => leaves[*tensor_index].clone(),
        TreeNode::Node {
            args,
            input_indices,
            output_indices,
        } => {
            let (ar, ai) = real_walk(&args[0], leaves);
            let (br, bi) = real_walk(&args[1], leaves);
            let c = |x: &Tensor<f32, B>, y: &Tensor<f32, B>| {
                x.contract_binary::<Standard<f32>>(
                    y,
                    &input_indices[0],
                    &input_indices[1],
                    output_indices,
                )
            };
            match (ai, bi) {
                (None, None) => (c(&ar, &br), None),
                (Some(ai), None) => {
                    let re = c(&ar, &br);
                    let im = c(&ai, &br);
                    (re, Some(im))
                }
                (None, Some(bi)) => {
                    let re = c(&ar, &br);
                    let im = c(&ar, &bi);
                    (re, Some(im))
                }
                (Some(ai), Some(bi)) => {
                    let asum = ar.linear_combination(&ai, 1.0);
                    let bsum = br.linear_combination(&bi, 1.0);
                    let p1 = c(&asum, &bsum);
                    let p2 = c(&ar, &br);
                    let p3 = c(&ai, &bi);
                    let re = p2.linear_combination(&p3, -1.0);
                    let im = p1
                        .linear_combination(&p2, -1.0)
                        .linear_combination(&p3, -1.0);
                    (re, Some(im))
                }
            }
        }
    }
}

pub fn solve_real<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<RealValue<B>>],
    backend: &B,
) -> Complex32
where
    f32: BackendScalar<B>,
{
    let outputs = assignments(network)
        .iter()
        .map(|assignment| {
            let leaves = select_real_leaves(network, cache, assignment);
            real_walk(&network.contraction_order, &leaves)
        })
        .collect::<Vec<_>>();
    backend.synchronize();
    let mut re = Kahan::default();
    let mut im = Kahan::default();
    for (r, i) in outputs {
        assert_eq!(
            r.numel(),
            1,
            "benchmark reduction currently requires scalar output"
        );
        re.add(r.to_vec()[0] as f64);
        if let Some(i) = i {
            im.add(i.to_vec()[0] as f64)
        }
    }
    Complex32::new(re.sum as f32, im.sum as f32)
}

pub fn native_cache<T, B: Backend + Clone, F: Fn(f32, f32) -> T>(
    network: &BenchmarkNetwork,
    backend: B,
    make: F,
) -> Vec<Vec<Tensor<T, B>>>
where
    T: Scalar + BackendScalar<B>,
{
    network
        .tensors
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(t, ix)| {
            let data = t
                .data_re
                .iter()
                .zip(&t.data_im)
                .map(|(&r, &i)| make(r, i))
                .collect::<Vec<_>>();
            local_assignments(network, ix)
                .iter()
                .map(|a| {
                    let (v, shape) = slice_column_major(&data, &t.shape, ix, a);
                    Tensor::from_data_with_backend(&v, &shape, backend.clone())
                })
                .collect()
        })
        .collect()
}
fn native_walk<T, B: Backend + Clone>(node: &TreeNode, leaves: &[Tensor<T, B>]) -> Tensor<T, B>
where
    T: Scalar + Zero + One + PartialEq + BackendScalar<B>,
{
    match node {
        TreeNode::Leaf { tensor_index } => leaves[*tensor_index].clone(),
        TreeNode::Node {
            args,
            input_indices,
            output_indices,
        } => native_walk(&args[0], leaves).contract_binary::<Standard<T>>(
            &native_walk(&args[1], leaves),
            &input_indices[0],
            &input_indices[1],
            output_indices,
        ),
    }
}
pub fn solve_native<T, B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<T, B>>],
    backend: &B,
) -> Complex32
where
    T: Scalar + Zero + One + PartialEq + Into<Complex32> + BackendScalar<B>,
{
    let out = assignments(network)
        .iter()
        .map(|assignment| {
            let leaves = select_native_leaves(network, cache, assignment);
            native_walk(&network.contraction_order, &leaves)
        })
        .collect::<Vec<_>>();
    backend.synchronize();
    let (mut re, mut im) = (Kahan::default(), Kahan::default());
    for x in out {
        assert_eq!(x.numel(), 1);
        let c: Complex32 = x.to_vec()[0].into();
        re.add(c.re as f64);
        im.add(c.im as f64)
    }
    Complex32::new(re.sum as f32, im.sum as f32)
}

fn local_assignments(network: &BenchmarkNetwork, labels: &[usize]) -> Vec<HashMap<usize, usize>> {
    let cuts = network
        .cuts
        .iter()
        .copied()
        .filter(|cut| labels.contains(cut))
        .collect::<Vec<_>>();
    let count = cuts.iter().map(|cut| network.size_dict[cut]).product();
    (0..count)
        .map(|mut n| {
            let mut assignment = HashMap::new();
            for cut in cuts.iter().rev() {
                assignment.insert(*cut, n % network.size_dict[cut]);
                n /= network.size_dict[cut];
            }
            assignment
        })
        .collect()
}

fn local_index(
    network: &BenchmarkNetwork,
    labels: &[usize],
    assignment: &HashMap<usize, usize>,
) -> usize {
    network
        .cuts
        .iter()
        .filter(|cut| labels.contains(cut))
        .fold(0, |index, cut| {
            index * network.size_dict[cut] + assignment[cut]
        })
}

fn select_real_leaves<B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<RealValue<B>>],
    assignment: &HashMap<usize, usize>,
) -> Vec<RealValue<B>> {
    cache
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(leaf, labels)| leaf[local_index(network, labels, assignment)].clone())
        .collect()
}

fn select_native_leaves<T: Scalar, B: Backend + Clone>(
    network: &BenchmarkNetwork,
    cache: &[Vec<Tensor<T, B>>],
    assignment: &HashMap<usize, usize>,
) -> Vec<Tensor<T, B>> {
    cache
        .iter()
        .zip(&network.eincode.input_indices)
        .map(|(leaf, labels)| leaf[local_index(network, labels, assignment)].clone())
        .collect()
}
#[derive(Default)]
struct Kahan {
    sum: f64,
    c: f64,
}
impl Kahan {
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omeinsum::Cpu;
    #[test]
    fn sliced_native_equals_tree_real() {
        let n = BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "x".into(),
            source_mode: "x".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: super::super::format::BenchmarkEinCode {
                input_indices: vec![vec![0], vec![0]],
                output_indices: vec![],
            },
            tensors: vec![
                super::super::format::BenchmarkTensor {
                    shape: vec![2],
                    data_re: vec![1., 2.],
                    data_im: vec![0., 0.],
                    structurally_complex: false,
                },
                super::super::format::BenchmarkTensor {
                    shape: vec![2],
                    data_re: vec![3., 4.],
                    data_im: vec![1., -1.],
                    structurally_complex: true,
                },
            ],
            size_dict: HashMap::from([(0, 2)]),
            contraction_order: TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: vec![vec![0], vec![0]],
                output_indices: vec![],
            },
            cuts: vec![0],
            assignment_count: 2,
            complexity: Default::default(),
            tree_real_audit: Default::default(),
        };
        let rc = real_cache(&n, Cpu);
        let nc = native_cache(&n, Cpu, Complex32::new);
        assert_eq!(solve_real(&n, &rc, &Cpu), Complex32::new(11., -1.));
        assert_eq!(solve_native(&n, &nc, &Cpu), solve_real(&n, &rc, &Cpu));
    }

    #[test]
    fn cache_counts_are_products_of_only_each_leafs_cuts() {
        let mut n = test_network(
            vec![vec![0], vec![1], vec![2]],
            vec![vec![2], vec![3], vec![5]],
            vec![0, 1],
            HashMap::from([(0, 2), (1, 3), (2, 5)]),
        );
        n.assignment_count = 6;
        let native = native_cache(&n, Cpu, Complex32::new);
        let real = real_cache(&n, Cpu);
        assert_eq!(
            native.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 3, 1]
        );
        assert_eq!(real.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 3, 1]);
    }

    #[test]
    fn sliced_multidimensional_complex_gauss_matches_native_values() {
        let shape = vec![2, 3, 4];
        let re_a = (0..24).map(|i| i as f32 * 0.25 + 1.0).collect::<Vec<_>>();
        let im_a = (0..24).map(|i| i as f32 * -0.1 + 0.5).collect::<Vec<_>>();
        let re_b = (0..24).map(|i| i as f32 * 0.15 - 0.75).collect::<Vec<_>>();
        let im_b = (0..24).map(|i| i as f32 * 0.2 + 0.25).collect::<Vec<_>>();
        let mut n = test_network(
            vec![vec![0, 1, 2], vec![0, 1, 2]],
            vec![shape.clone(), shape],
            vec![1],
            HashMap::from([(0, 2), (1, 3), (2, 4)]),
        );
        n.assignment_count = 3;
        n.tensors[0].data_re = re_a;
        n.tensors[0].data_im = im_a;
        n.tensors[1].data_re = re_b;
        n.tensors[1].data_im = im_b;
        n.tensors
            .iter_mut()
            .for_each(|t| t.structurally_complex = true);
        let real = solve_real(&n, &real_cache(&n, Cpu), &Cpu);
        let native = solve_native(&n, &native_cache(&n, Cpu, Complex32::new), &Cpu);
        assert!(real.re != 0.0 && real.im != 0.0);
        assert!(
            (real - native).norm() <= 2e-4 * native.norm().max(1.0),
            "real={real:?}, native={native:?}"
        );
    }

    fn test_network(
        labels: Vec<Vec<usize>>,
        shapes: Vec<Vec<usize>>,
        cuts: Vec<usize>,
        size_dict: HashMap<usize, usize>,
    ) -> BenchmarkNetwork {
        let tensors = shapes
            .iter()
            .map(|shape| {
                let len = shape.iter().product();
                super::super::format::BenchmarkTensor {
                    shape: shape.clone(),
                    data_re: vec![1.0; len],
                    data_im: vec![0.0; len],
                    structurally_complex: false,
                }
            })
            .collect::<Vec<_>>();
        let contraction_order = if tensors.len() == 2 {
            TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: labels.clone(),
                output_indices: vec![],
            }
        } else {
            TreeNode::Leaf { tensor_index: 0 }
        };
        BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "x".into(),
            source_mode: "x".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: super::super::format::BenchmarkEinCode {
                input_indices: labels,
                output_indices: vec![],
            },
            tensors,
            size_dict,
            contraction_order,
            cuts,
            assignment_count: 1,
            complexity: Default::default(),
            tree_real_audit: Default::default(),
        }
    }
}
