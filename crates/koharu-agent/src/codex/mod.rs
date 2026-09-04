mod auth;
mod catalog;
mod protocol;
mod stream;
mod token_store;

use anyhow::{Result, anyhow};
use reqwest::{Client, StatusCode};
use serde::Serialize;
use specta::Type;

#[cfg(test)]
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{Control, Reasoning, provider::ProviderRequestError};

pub use auth::Account;
use auth::Auth;
pub(crate) use protocol::{
    Request, function_output, message, project_context, retained_function_output,
};
pub(crate) use stream::{Delta, Turn};

const RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";

#[derive(Clone, Debug, Serialize, Type)]
pub struct CodexModel {
    pub id: String,
    pub name: String,
    pub reasoning: Vec<Reasoning>,
}

#[derive(Clone, Debug, Serialize, Type)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LoginEvent {
    Progress {
        message: String,
    },
    DeviceCode {
        verification_url: String,
        user_code: String,
    },
}

#[derive(Clone, Debug)]
pub struct Codex {
    client: Client,
    auth: Auth,
    #[cfg(test)]
    scripted: Option<ScriptedCodex>,
}

impl Codex {
    pub fn new() -> Result<Self> {
        let client = koharu_runtime::http_client()?;
        Ok(Self {
            auth: Auth::new(client.clone()),
            client,
            #[cfg(test)]
            scripted: None,
        })
    }

    pub fn account(&self) -> Result<Option<Account>> {
        self.auth.account()
    }

    #[tracing::instrument(skip_all)]
    pub async fn login_device<F>(&self, control: &Control, publish: F) -> Result<Account>
    where
        F: FnMut(LoginEvent),
    {
        self.auth.login_device(control, publish).await
    }

    pub fn logout(&self) -> Result<()> {
        self.auth.logout()
    }

    #[tracing::instrument(skip_all)]
    pub async fn models(&self) -> Result<Vec<CodexModel>> {
        #[cfg(test)]
        if self.scripted.is_some() {
            return Ok(vec![CodexModel {
                id: "test-codex".to_owned(),
                name: "Test Codex".to_owned(),
                reasoning: Vec::new(),
            }]);
        }
        catalog::models(&self.client, &self.auth).await
    }

    pub(crate) async fn respond<F>(
        &self,
        request: &Request,
        control: &Control,
        publish: F,
    ) -> Result<Turn>
    where
        F: FnMut(Delta),
    {
        #[cfg(test)]
        if let Some(scripted) = &self.scripted {
            control.ensure_running()?;
            scripted.calls.fetch_add(1, Ordering::SeqCst);
            scripted
                .requests
                .lock()
                .expect("scripted request lock must not be poisoned")
                .push(serde_json::to_value(request)?);
            return match scripted
                .responses
                .lock()
                .expect("scripted response lock must not be poisoned")
                .pop_front()
                .expect("scripted Codex response queue was exhausted")
            {
                ScriptedResponse::Turn(turn) => Ok(turn),
                ScriptedResponse::Error(class) => Err(anyhow!(ProviderRequestError::new(class))),
            };
        }
        control.ensure_running()?;
        let session = self.auth.session().await?;
        let mut response = self.send(request, &session, control).await?;
        if response.status() == StatusCode::UNAUTHORIZED {
            let session = self.auth.force_refresh().await?;
            response = self.send(request, &session, control).await?;
        }
        if !response.status().is_success() {
            let status = response.status();
            let mut body = response.text().await.unwrap_or_default();
            body.truncate(16 * 1024);
            return Err(anyhow!(ProviderRequestError::from_http(status, &body)));
        }
        stream::read(response, control, publish).await
    }

    async fn send(
        &self,
        request: &Request,
        session: &auth::Session,
        control: &Control,
    ) -> Result<reqwest::Response> {
        let request_id = request.prompt_cache_key.clone();
        let send = self
            .client
            .post(RESPONSES_URL)
            .bearer_auth(&session.access)
            .header("chatgpt-account-id", &session.account.id)
            .header("originator", "koharu")
            .header("OpenAI-Beta", "responses=experimental")
            .header("accept", "text/event-stream")
            .header("session_id", &request_id)
            .header("x-client-request-id", request_id)
            .json(request)
            .send();
        tokio::select! {
            response = send => response.map_err(|error| {
                ProviderRequestError::from_transport(&error)
                    .map_or_else(|| anyhow!(error), anyhow::Error::new)
            }),
            () = control.cancelled() => {
                control.ensure_running()?;
                unreachable!("cancelled control must fail ensure_running")
            }
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct ScriptedCodexHandle {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[cfg(test)]
impl ScriptedCodexHandle {
    pub(crate) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub(crate) fn requests(&self) -> Vec<serde_json::Value> {
        self.requests
            .lock()
            .expect("scripted request lock must not be poisoned")
            .clone()
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct ScriptedCodex {
    responses: Arc<Mutex<VecDeque<ScriptedResponse>>>,
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum ScriptedResponse {
    Turn(Turn),
    Error(crate::provider::ProviderErrorClass),
}

#[cfg(test)]
impl Codex {
    pub(crate) fn scripted(
        responses: impl IntoIterator<Item = ScriptedResponse>,
    ) -> (Self, ScriptedCodexHandle) {
        let calls = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let scripted = ScriptedCodex {
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            calls: Arc::clone(&calls),
            requests: Arc::clone(&requests),
        };
        (
            Self {
                client: Client::new(),
                auth: Auth::new(Client::new()),
                scripted: Some(scripted),
            },
            ScriptedCodexHandle { calls, requests },
        )
    }
}
