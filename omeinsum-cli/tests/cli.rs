use std::io::Write;

use assert_cmd::Command;
use predicates::str::contains;
use tempfile::NamedTempFile;

fn cmd() -> Command {
    Command::cargo_bin("omeinsum").unwrap()
}

fn write_temp_json(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();
    file
}

fn two_leaf_yao_tn() -> serde_json::Value {
    serde_json::json!({
        "format": "yao-tn-v1",
        "mode": "overlap",
        "eincode": {
            "input_indices": [["0"], ["0"]],
            "output_indices": [],
        },
        "tensors": [
            {
                "shape": [2],
                "data_re": [1.0, 2.0],
                "data_im": [0.0, 0.0],
            },
            {
                "shape": [2],
                "data_re": [3.0, 4.0],
                "data_im": [0.0, 0.0],
            },
        ],
        "size_dict": {"0": 2},
        "contraction_order": {
            "isleaf": false,
            "args": [
                {"isleaf": true, "tensorindex": 0},
                {"isleaf": true, "tensorindex": 1},
            ],
            "eins": {
                "ixs": [[0], [0]],
                "iy": [],
            },
        },
    })
}

fn assert_static_plan_rejected(json: &serde_json::Value, case: &str) {
    let input = write_temp_json(&json.to_string());
    let output = cmd()
        .args([
            "static-plan",
            input.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "{case} unexpectedly succeeded: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn static_plan_rejects_invalid_yao_tn() {
    let base = two_leaf_yao_tn();
    let mut cases = Vec::new();

    let mut value = base.clone();
    value["format"] = "other".into();
    cases.push(("format", value));

    let mut value = base.clone();
    value["mode"] = "pure".into();
    cases.push(("mode", value));

    let mut value = base.clone();
    value["eincode"]["output_indices"] = serde_json::json!(["0"]);
    cases.push(("non-scalar output", value));

    let mut value = base.clone();
    value["contraction_order"] = serde_json::Value::Null;
    cases.push(("missing order", value));

    let mut value = base.clone();
    value["tensors"].as_array_mut().unwrap().pop();
    cases.push(("tensor count", value));

    let mut value = base.clone();
    value["tensors"][0]["data_re"] = serde_json::json!([1.0]);
    cases.push(("shape product", value));

    let mut value = base.clone();
    value["size_dict"] = serde_json::json!({});
    cases.push(("missing size label", value));

    let mut value = base.clone();
    value["size_dict"]["0"] = 3.into();
    cases.push(("inconsistent size label", value));

    let mut value = base.clone();
    value["eincode"]["input_indices"][0] = serde_json::json!(["0", "0"]);
    value["tensors"][0]["shape"] = serde_json::json!([2, 2]);
    value["tensors"][0]["data_re"] = serde_json::json!([1.0, 2.0, 3.0, 4.0]);
    value["tensors"][0]["data_im"] = serde_json::json!([0.0, 0.0, 0.0, 0.0]);
    cases.push(("repeated tensor label", value));

    let mut value = base.clone();
    value["contraction_order"]["args"][1]["tensorindex"] = 9.into();
    cases.push(("leaf index", value));

    let mut value = base;
    value["contraction_order"]["args"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"isleaf": true, "tensorindex": 0}));
    cases.push(("non-binary node", value));

    for (case, value) in cases {
        assert_static_plan_rejected(&value, case);
    }
}

#[test]
fn static_plan_normalizes_optimized_yao_tn_without_changing_tree_order() {
    let input = write_temp_json(&two_leaf_yao_tn().to_string());
    let output = cmd()
        .args([
            "static-plan",
            input.path().to_str().unwrap(),
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
    let normalized: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let root = &normalized["tree"]["Node"];
    assert_eq!(root["output_modes"], serde_json::json!([]));
    assert_eq!(root["left"]["Leaf"]["tensor_index"], 0);
    assert_eq!(root["right"]["Leaf"]["tensor_index"], 1);
}

#[test]
fn test_optimize_matmul() {
    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "--pretty",
            "true",
        ])
        .assert()
        .success()
        .stdout(contains("schema_version"))
        .stdout(contains("\"expression\": \"ij,jk->ik\""));
}

#[test]
fn test_optimize_to_file() {
    let out = NamedTempFile::new().unwrap();
    let out_path = out.path().to_str().unwrap().to_string();

    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "-o",
            &out_path,
        ])
        .assert()
        .success();

    let content = std::fs::read_to_string(&out_path).unwrap();
    let topology: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(topology["schema_version"], 1);
}

#[test]
fn test_contract_with_expr_f64() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            {"shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
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
    assert_eq!(result["dtype"], "f64");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_contract_with_expr_f32_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f32",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            {"shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
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
    assert_eq!(result["dtype"], "f32");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_contract_trace_c32() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c32",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ii->",
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
    assert_eq!(result["shape"], serde_json::json!([]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![3.0, 0.0]);
}

#[test]
fn test_contract_transpose_c64_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ij->ji",
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
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![1.0, 1.0, 3.0, 0.0, 2.0, -1.0, 4.0, 2.0]);
}

#[test]
fn test_optimize_then_contract_pipeline() {
    let topo_file = NamedTempFile::new().unwrap();
    let topo_path = topo_file.path().to_str().unwrap().to_string();

    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=2,k=2",
            "-o",
            &topo_path,
        ])
        .assert()
        .success();

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]},
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "-t",
            &topo_path,
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
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_contract_scalar_output() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ii->",
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
    assert_eq!(result["shape"], serde_json::json!([]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![2.0]);
}

#[test]
fn test_contract_both_flags_error() {
    let tensors = write_temp_json(r#"{"dtype":"f64","tensors":[]}"#);
    cmd()
        .args([
            "contract",
            "-t",
            "topo.json",
            "--expr",
            "ij->ij",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("Cannot specify both"));
}

#[test]
fn test_contract_no_flags_error() {
    let tensors = write_temp_json(r#"{"dtype":"f64","tensors":[]}"#);
    cmd()
        .args(["contract", tensors.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(contains("Must specify either"));
}

#[test]
fn test_optimize_invalid_method_error() {
    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "--method",
            "badmethod",
        ])
        .assert()
        .failure()
        .stderr(contains("Unknown method"));
}

#[test]
fn test_contract_invalid_schema_version() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 99,
        "expression": "ij,jk->ik",
        "label_map": {"i": 0, "j": 1, "k": 2},
        "size_dict": {"0": 2, "1": 2, "2": 2},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 0}}
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("schema_version"));
}

#[test]
fn test_contract_shape_mismatch() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2, 3], "data": [1.0, 2.0, 3.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "--expr",
            "ij->ij",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("doesn't match shape"));
}

#[test]
fn test_contract_invalid_leaf_index_in_topology() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i->i",
        "label_map": {"i": 0},
        "size_dict": {"0": 2},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 9}}
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2], "data": [1.0, 2.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("out of range"));
}

#[test]
fn test_contract_non_binary_topology_error() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i,j,k->i",
        "label_map": {"i": 0, "j": 1, "k": 2},
        "size_dict": {"0": 2, "1": 2, "2": 2},
        "method": "greedy",
        "tree": {
            "Node": {
                "args": [
                    {"Leaf": {"tensor_index": 0}},
                    {"Leaf": {"tensor_index": 1}},
                    {"Leaf": {"tensor_index": 2}}
                ],
                "eins": {"ixs": [[0], [1], [2]], "iy": [0]}
            }
        }
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0]},
            {"shape": [2], "data": [3.0, 4.0]},
            {"shape": [2], "data": [5.0, 6.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("binary"));
}

#[test]
fn test_contract_unknown_label_in_topology_error() {
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
                    {"Leaf": {"tensor_index": 1}}
                ],
                "eins": {"ixs": [[0], [9]], "iy": [0, 9]}
            }
        }
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0]},
            {"shape": [2], "data": [3.0, 4.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("unknown label index"));
}

#[test]
fn test_contract_missing_size_for_topology_expression_error() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i->i",
        "label_map": {"i": 0},
        "size_dict": {},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 0}}
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2], "data": [1.0, 2.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("Missing size for label index"));
}

#[test]
fn test_autodiff_scalar_output_default_seed() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--expr",
            "ii->",
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
    assert_eq!(result["dtype"], "f64");
    assert_eq!(result["order"], "col_major");
    assert_eq!(result["result"]["shape"], serde_json::json!([]));
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![5.0]);

    let gradients = result["gradients"].as_array().unwrap();
    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0]["input_index"], 0);
    assert_eq!(gradients[0]["shape"], serde_json::json!([2, 2]));
    let grad_data: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    assert_eq!(grad_data, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_autodiff_complex_scalar_output_default_seed() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--expr",
            "ii->",
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
    assert_eq!(result["result"]["shape"], serde_json::json!([]));
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![3.0, 0.0]);

    let gradients = result["gradients"].as_array().unwrap();
    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0]["input_index"], 0);
    assert_eq!(gradients[0]["shape"], serde_json::json!([2, 2]));
    let grad_data: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    assert_eq!(grad_data, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_autodiff_seeded_nonscalar_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]},
            {"shape": [2, 2], "data": [5.0, 6.0, 7.0, 8.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "row_major",
        "shape": [2, 2],
        "data": [1.0, 1.0, 1.0, 1.0]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "--expr",
            "ij,jk->ik",
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
    assert_eq!(result["dtype"], "f64");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["result"]["shape"], serde_json::json!([2, 2]));
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![19.0, 22.0, 43.0, 50.0]);

    let gradients = result["gradients"].as_array().unwrap();
    assert_eq!(gradients.len(), 2);
    assert_eq!(gradients[0]["input_index"], 0);
    assert_eq!(gradients[1]["input_index"], 1);
    let grad_a: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    let grad_b: Vec<f64> = serde_json::from_value(gradients[1]["data"].clone()).unwrap();
    assert_eq!(grad_a, vec![11.0, 15.0, 11.0, 15.0]);
    assert_eq!(grad_b, vec![4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_autodiff_complex_seeded_nonscalar_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "row_major",
        "shape": [2, 2],
        "data": [0.5, -1.0, 1.5, 0.0, -2.0, 0.25, 3.0, -4.0]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
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
    assert_eq!(result["dtype"], "c64");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["result"]["shape"], serde_json::json!([2, 2]));
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]);

    let gradients = result["gradients"].as_array().unwrap();
    assert_eq!(gradients.len(), 1);
    assert_eq!(gradients[0]["input_index"], 0);
    assert_eq!(gradients[0]["shape"], serde_json::json!([2, 2]));
    let grad_data: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    assert_eq!(grad_data, vec![0.5, -1.0, 1.5, 0.0, -2.0, 0.25, 3.0, -4.0]);
}

#[test]
fn test_autodiff_nonscalar_requires_grad_output() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]},
            {"shape": [2, 2], "data": [5.0, 6.0, 7.0, 8.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "autodiff",
            "--expr",
            "ij,jk->ik",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("Non-scalar output requires --grad-output"));
}

#[test]
fn test_autodiff_grad_output_shape_mismatch() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]},
            {"shape": [2, 2], "data": [5.0, 6.0, 7.0, 8.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "shape": [4],
        "data": [1.0, 1.0, 1.0, 1.0]
    }"#,
    );

    cmd()
        .args([
            "autodiff",
            "--expr",
            "ij,jk->ik",
            "--grad-output",
            grad_output.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("grad_output shape"));
}

#[test]
fn test_autodiff_with_topology_scalar_output() {
    let topo_file = NamedTempFile::new().unwrap();
    let topo_path = topo_file.path().to_str().unwrap().to_string();

    cmd()
        .args(["optimize", "ii->", "--sizes", "i=2", "-o", &topo_path])
        .assert()
        .success();

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "autodiff",
            "-t",
            &topo_path,
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
    let result_data: Vec<f64> = serde_json::from_value(result["result"]["data"].clone()).unwrap();
    assert_eq!(result_data, vec![5.0]);
    let gradients = result["gradients"].as_array().unwrap();
    let grad_data: Vec<f64> = serde_json::from_value(gradients[0]["data"].clone()).unwrap();
    assert_eq!(grad_data, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_autodiff_grad_output_order_mismatch() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]},
            {"shape": [2, 2], "data": [5.0, 6.0, 7.0, 8.0]}
        ]
    }"#,
    );
    let grad_output = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "row_major",
        "shape": [2, 2],
        "data": [1.0, 1.0, 1.0, 1.0]
    }"#,
    );

    cmd()
        .args([
            "autodiff",
            "--expr",
            "ij,jk->ik",
            "--grad-output",
            grad_output.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("grad_output order"));
}
