use omeinsum::backend::ascend::{
    run_matmul_smoke, AscendPrecisionMode, AscendSession, AscendSessionConfig,
};
use serde::Serialize;

#[derive(Serialize)]
struct Report {
    status: &'static str,
    device_id: i32,
    soc_name: String,
    #[serde(flatten)]
    smoke: omeinsum::backend::ascend::AscendMatmulSmokeReport,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let mut device_id = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--device-id" => {
                let value = arguments.next().ok_or("--device-id requires a value")?;
                device_id = Some(value.parse()?);
            }
            _ => return Err(format!("unrecognized argument {argument:?}").into()),
        }
    }
    let device_id = device_id.ok_or("--device-id is required")?;
    let session = AscendSession::new(&AscendSessionConfig {
        device_id,
        precision_mode: AscendPrecisionMode::KeepDtype,
    })?;
    let smoke = run_matmul_smoke(&session)?;
    if smoke.descriptor_creations_during_replay != 0
        || smoke.workspace_queries_during_replay != 0
        || smoke.op_runs_during_replay != 2
    {
        return Err("replay unexpectedly prepared native resources".into());
    }
    println!(
        "{}",
        serde_json::to_string(&Report {
            status: "passed",
            device_id,
            soc_name: session.device_info().soc_name.clone(),
            smoke,
        })?
    );
    Ok(())
}
