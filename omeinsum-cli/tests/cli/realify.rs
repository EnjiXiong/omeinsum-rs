use predicates::str::contains;
use tempfile::NamedTempFile;

use super::{cmd, write_temp_json};

#[test]
fn test_contract_realify_c64_matmul() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 2.0, -1.0]},
            {"shape": [2, 2], "data": [2.0, 0.0, 1.0, -1.0, 0.0, 1.0, 4.0, 0.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--realify",
            "--expr",
            "ij,jk->ik",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c64");
    assert_eq!(result["order"], "col_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![5.0, 1.0, 1.0, -5.0, 10.0, 1.0, 9.0, -4.0]);
}

#[test]
fn test_contract_realify_c32_row_major_with_topology() {
    let topology = NamedTempFile::new().unwrap();
    cmd()
        .args([
            "optimize",
            "ij->ji",
            "--sizes",
            "i=2,j=2",
            "-o",
            topology.path().to_str().unwrap(),
        ])
        .assert()
        .success();
    let tensors = write_temp_json(
        r#"{
        "dtype": "c32",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--realify",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c32");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![1.0, 1.0, 3.0, 0.0, 2.0, -1.0, 4.0, 2.0]);
}

#[test]
fn test_contract_realify_rejects_real_dtype() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "tensors": [{"shape": [1], "data": [2.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "--realify",
            "--expr",
            "i->i",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("--realify requires dtype c32 or c64"));
}

#[test]
fn test_contract_realify_rejects_topology_tensor_size_mismatch() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i->i",
        "label_map": {"i": 0},
        "size_dict": {"0": 3},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 0}}
    }"#,
    );
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "tensors": [{"shape": [2], "data": [1.0, 0.0, 2.0, 0.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "--realify",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("axis 0 has size 2"));
}

#[test]
fn test_contract_realify_rejects_duplicate_topology_leaf() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i,j->ij",
        "label_map": {"i": 0, "j": 1},
        "size_dict": {"0": 2, "1": 2},
        "method": "greedy",
        "tree": {
            "Node": {
                "args": [
                    {"Leaf": {"tensor_index": 0}},
                    {"Leaf": {"tensor_index": 0}}
                ],
                "eins": {"ixs": [[0], [0]], "iy": [0]}
            }
        }
    }"#,
    );
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "tensors": [
            {"shape": [2], "data": [1.0, 0.0, 2.0, 0.0]},
            {"shape": [2], "data": [3.0, 0.0, 4.0, 0.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "--realify",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("exactly once"));
}

#[test]
fn test_autodiff_realify_complex_scalar_uses_real_part_seed() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "col_major",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0, 3.0, -1.0]},
            {"shape": [2], "data": [2.0, -1.0, -1.0, 4.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--realify",
            "--expr",
            "i,i->",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c64");
    assert_eq!(result["result"]["shape"], serde_json::json!([]));
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![5.0, 16.0]);

    let gradients = result["gradients"].as_array().unwrap();
    assert_eq!(gradients.len(), 2);
    let grad_a: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    let grad_b: Vec<f64> = serde_json::from_value(gradients[1]["data"].clone()).unwrap();
    assert_eq!(grad_a, vec![2.0, 1.0, -1.0, -4.0]);
    assert_eq!(grad_b, vec![1.0, -2.0, 3.0, 1.0]);
}

#[test]
fn test_autodiff_realify_complex_product_with_complex_seed() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0, 3.0, -1.0]},
            {"shape": [2], "data": [2.0, -1.0, -1.0, 4.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "col_major",
        "shape": [],
        "data": [0.5, -1.0]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--realify",
            "--expr",
            "i,i->",
            "--grad-output",
            grad_output.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let gradients = result["gradients"].as_array().unwrap();
    let grad_a: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    let grad_b: Vec<f64> = serde_json::from_value(gradients[1]["data"].clone()).unwrap();
    assert_eq!(grad_a, vec![2.0, -1.5, -4.5, -1.0]);
    assert_eq!(grad_b, vec![-1.5, -2.0, 2.5, -2.5]);
}

#[test]
fn test_autodiff_realify_complex_seeded_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c32",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "c32",
        "order": "row_major",
        "shape": [2, 2],
        "data": [0.5, -1.0, 1.5, 0.0, -2.0, 0.25, 3.0, -4.0]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--realify",
            "--expr",
            "ij->ij",
            "--grad-output",
            grad_output.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c32");
    assert_eq!(result["order"], "row_major");
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]);
    let grad_data: Vec<f64> =
        serde_json::from_value(result["gradients"][0]["data"].clone()).unwrap();
    assert_eq!(grad_data, vec![0.5, -1.0, 1.5, 0.0, -2.0, 0.25, 3.0, -4.0]);
}

#[test]
fn test_autodiff_realify_rejects_real_dtype() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f32",
        "tensors": [{"shape": [1], "data": [2.0]}]
    }"#,
    );

    cmd()
        .args([
            "autodiff",
            "--realify",
            "--expr",
            "i->",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("--realify requires dtype c32 or c64"));
}
