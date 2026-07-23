use omeinsum::static_plan::build_plan_bundle;

pub(crate) fn run(
    input: &str,
    realness_tol: f64,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let json = std::fs::read_to_string(input)
        .map_err(|error| format!("Failed to read '{input}': {error}"))?;
    let network = crate::yao_tn::parse_yao_tn(&json).map_err(|error| error.to_string())?;
    let bundle = build_plan_bundle(&network, realness_tol).map_err(|error| error.to_string())?;
    bundle.validate().map_err(|error| error.to_string())?;
    crate::common::write_json_output(&bundle, output, pretty)
}
