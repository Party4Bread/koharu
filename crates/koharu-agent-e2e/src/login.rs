use anyhow::{Context as _, Result, bail};
use clap::Parser as _;
use koharu_agent::{Account, Codex, Control, LoginEvent};
use serde::Serialize;

#[derive(Debug, clap::Parser)]
#[command(
    version,
    about = "Manage Koharu Codex authentication without a desktop or webview"
)]
struct Arguments {
    /// Report whether Koharu has stored Codex credentials without starting login.
    #[arg(long, conflicts_with = "logout")]
    status: bool,

    /// Remove Koharu's stored Codex credentials without starting login.
    #[arg(long, conflicts_with = "status")]
    logout: bool,
}

#[derive(Debug, Eq, PartialEq)]
enum Action {
    Login,
    Status,
    Logout,
}

#[derive(Debug, Serialize)]
struct Success {
    ok: bool,
    signed_in: bool,
    account: Option<Account>,
}

impl Arguments {
    fn action(&self) -> Action {
        if self.status {
            Action::Status
        } else if self.logout {
            Action::Logout
        } else {
            Action::Login
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        print_json(&serde_json::json!({
            "ok": false,
            "error": format!("{error:#}"),
        }));
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let arguments = Arguments::try_parse().context("invalid koharu-agent-login command line")?;
    let codex = Codex::new()?;
    let account = match arguments.action() {
        Action::Login => Some(login(&codex).await?),
        Action::Status => codex.account()?,
        Action::Logout => {
            codex.logout()?;
            None
        }
    };
    print_json(&Success {
        ok: true,
        signed_in: account.is_some(),
        account,
    });
    Ok(())
}

async fn login(codex: &Codex) -> Result<Account> {
    let control = Control::default();
    let login = codex.login_device(&control, |event| {
        eprintln!("{}", render_event(&event));
    });
    tokio::pin!(login);

    tokio::select! {
        result = &mut login => result,
        signal = tokio::signal::ctrl_c() => {
            signal.context("failed to listen for login cancellation")?;
            control.cancel();
            bail!("Codex device login cancelled")
        }
    }
}

fn render_event(event: &LoginEvent) -> String {
    match event {
        LoginEvent::Progress { message } => message.clone(),
        LoginEvent::DeviceCode {
            verification_url,
            user_code,
        } => format!("Verification URL: {verification_url}\nUser code: {user_code}"),
    }
}

fn print_json(value: &impl Serialize) {
    println!(
        "{}",
        serde_json::to_string(value).expect("CLI result JSON must serialize")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_selects_device_login() {
        let arguments = Arguments::try_parse_from(["koharu-agent-login"]).unwrap();

        assert_eq!(arguments.action(), Action::Login);
    }

    #[test]
    fn status_and_logout_are_explicit_and_mutually_exclusive() {
        let status = Arguments::try_parse_from(["koharu-agent-login", "--status"]).unwrap();
        let logout = Arguments::try_parse_from(["koharu-agent-login", "--logout"]).unwrap();

        assert_eq!(status.action(), Action::Status);
        assert_eq!(logout.action(), Action::Logout);
        assert!(Arguments::try_parse_from(["koharu-agent-login", "--status", "--logout"]).is_err());
    }

    #[test]
    fn device_code_event_renders_url_and_one_time_code() {
        let rendered = render_event(&LoginEvent::DeviceCode {
            verification_url: "https://auth.example/device".to_owned(),
            user_code: "ABCD-EFGH".to_owned(),
        });

        assert_eq!(
            rendered,
            "Verification URL: https://auth.example/device\nUser code: ABCD-EFGH"
        );
    }

    #[test]
    fn progress_event_renders_only_its_safe_message() {
        let rendered = render_event(&LoginEvent::Progress {
            message: "Waiting for approval".to_owned(),
        });

        assert_eq!(rendered, "Waiting for approval");
    }

    #[test]
    fn success_json_contains_only_safe_account_metadata() {
        let value = serde_json::to_value(Success {
            ok: true,
            signed_in: true,
            account: Some(Account {
                id: "account-id".to_owned(),
                email: Some("person@example.com".to_owned()),
                plan: Some("plus".to_owned()),
            }),
        })
        .unwrap();

        assert_eq!(
            value,
            serde_json::json!({
                "ok": true,
                "signed_in": true,
                "account": {
                    "id": "account-id",
                    "email": "person@example.com",
                    "plan": "plus",
                },
            })
        );
    }
}
