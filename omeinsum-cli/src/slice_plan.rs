use omeinsum::static_plan::build_sliced_plan_bundle;

pub(crate) fn run(
    source: &str,
    slice_modes: &str,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let bundle = crate::execute_plan::read_bundle(source)?;
    let modes = parse_slice_modes(slice_modes)?;
    let sliced = build_sliced_plan_bundle(&bundle, &modes).map_err(|error| error.to_string())?;
    sliced.validate().map_err(|error| error.to_string())?;
    crate::common::write_json_output(&sliced, output, pretty)
}

fn parse_slice_modes(value: &str) -> Result<Vec<i32>, String> {
    let modes = value
        .split(',')
        .map(str::trim)
        .enumerate()
        .map(|(index, token)| {
            if token.is_empty() {
                return Err(format!("slice mode at position {index} is empty"));
            }
            token.parse::<i32>().map_err(|error| {
                format!("invalid slice mode {token:?} at position {index}: {error}")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if modes.is_empty() {
        return Err("at least one slice mode is required".to_string());
    }
    Ok(modes)
}
