use anyhow::{Context as _, Result};
use clap::Parser as _;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        let failure = serde_json::json!({
            "ok": false,
            "error": format!("{error:#}"),
        });
        println!(
            "{}",
            serde_json::to_string(&failure).expect("failure JSON must serialize")
        );
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let arguments = koharu_agent_e2e::Arguments::try_parse()
        .context("invalid koharu-agent-e2e command line")?;
    let summary = koharu_agent_e2e::execute(arguments).await?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
