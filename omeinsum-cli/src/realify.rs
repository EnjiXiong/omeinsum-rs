use num_complex::Complex;
use num_traits::Float;
use omeinsum::algebra::Scalar;
use omeinsum::{realify_code, realify_data, BackendScalar, Cpu, Einsum, Tensor};

use crate::common::{build_explicit_einsum, load_complex_tensors};
use crate::format::TensorsFile;

pub(crate) struct PreparedRealify<T: Scalar> {
    pub(crate) einsum: Einsum<usize>,
    pub(crate) tensors: Vec<Tensor<T, Cpu>>,
    pub(crate) source_input_count: usize,
    pub(crate) source_output_shape: Vec<usize>,
}

pub(crate) fn prepare<T>(
    tensors_file: &TensorsFile,
    topology_path: Option<&str>,
    expr: Option<&str>,
    make_complex: fn(f64, f64) -> Complex<T>,
) -> Result<PreparedRealify<T>, String>
where
    T: Scalar + Float + BackendScalar<Cpu>,
    Complex<T>: Scalar + BackendScalar<Cpu>,
{
    let complex_tensors = load_complex_tensors(tensors_file, make_complex)?;
    let complex_refs: Vec<&Tensor<Complex<T>, Cpu>> = complex_tensors.iter().collect();
    let source = build_explicit_einsum(&complex_refs, topology_path, expr)?;
    let source_output_shape = source
        .iy
        .iter()
        .map(|label| source.size_dict[label])
        .collect();
    let source_input_count = complex_tensors.len();
    let is_complex = vec![true; source_input_count];
    let mut plan = realify_code(&source.ixs, &source.iy, &source.size_dict, &is_complex);

    let mut tensors = Vec::with_capacity(source_input_count + plan.num_mul_vertices);
    for tensor in &complex_tensors {
        tensors.push(realify_tensor(tensor));
    }
    for _ in 0..plan.num_mul_vertices {
        tensors.push(omeinsum::realify::mul_vertex_tensor::<T, Cpu>(Cpu));
    }

    // Source trees have no leaves for inserted multiplication vertices, so the
    // transformed topology must be planned independently.
    plan.einsum.optimize_greedy();
    Ok(PreparedRealify {
        einsum: plan.einsum,
        tensors,
        source_input_count,
        source_output_shape,
    })
}

pub(crate) fn realify_tensor<T>(tensor: &Tensor<Complex<T>, Cpu>) -> Tensor<T, Cpu>
where
    T: Scalar + Float + BackendScalar<Cpu>,
    Complex<T>: Scalar + BackendScalar<Cpu>,
{
    let data = realify_data(&tensor.to_vec(), false);
    let mut shape = tensor.shape().to_vec();
    shape.push(2);
    Tensor::from_data(&data, &shape)
}
