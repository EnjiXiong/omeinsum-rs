use omeinsum::backend::ascend::{AscendPrecisionMode, AscendSession, AscendSessionConfig};
use serde::Serialize;

#[derive(Serialize)]
struct SmokeReport {
    status: &'static str,
    device_id: i32,
    soc_name: String,
    bytes: usize,
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
    let input = [1.0f32, -2.0, 3.5, f32::MIN_POSITIVE];
    let output = session.round_trip_f32(&input)?;
    if input
        .iter()
        .zip(&output)
        .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("Ascend H2D/D2H result differs bitwise".into());
    }
    println!(
        "{}",
        serde_json::to_string(&SmokeReport {
            status: "passed",
            device_id,
            soc_name: session.device_info().soc_name.clone(),
            bytes: std::mem::size_of_val(&input),
        })?
    );
    Ok(())
}
