//! API client library.
//!
//! Provides a Rust client for the miner's HTTP API, shared by the CLI
//! and TUI binaries.
//!
//! Uses a minimal HTTP/1.1 client built on `hyper-util` instead of a
//! full-featured library like reqwest.  The API client only makes plain
//! HTTP GET requests to localhost, so TLS, unicode normalisation, and
//! content-encoding support are unnecessary.

pub mod types;

use anyhow::{Context, Result};
use http_body_util::{BodyExt, Empty};
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;

use types::MinerState;

/// Default API base URL.
///
/// Port 7785 = ASCII 'M' (77) + 'U' (85).
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:7785";

/// HTTP client for the miner API.
pub struct Client {
    http: HyperClient<hyper_util::client::legacy::connect::HttpConnector, Empty<bytes::Bytes>>,
    base_url: String,
}

impl Client {
    /// Create a client connecting to the default local address.
    pub fn new() -> Self {
        let http = HyperClient::builder(TokioExecutor::new()).build_http();
        Self {
            http,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create a client connecting to a specific base URL.
    pub fn with_base_url(base_url: String) -> Self {
        let http = HyperClient::builder(TokioExecutor::new()).build_http();
        Self { http, base_url }
    }

    /// Fetch the current miner state snapshot.
    pub async fn get_miner(&self) -> Result<MinerState> {
        self.get_json("miner").await
    }

    /// Perform a GET request and return the response body bytes.
    async fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let uri: hyper::Uri = format!("{}/api/v0/{}", self.base_url, path)
            .parse()
            .context("invalid API URL")?;
        let response = self
            .http
            .get(uri)
            .await
            .context("failed to connect to miner API")?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("API request failed: {}", status);
        }
        let body = response
            .into_body()
            .collect()
            .await
            .context("failed to read API response")?
            .to_bytes();
        Ok(body.to_vec())
    }

    /// GET a v0 API endpoint and deserialize the JSON response.
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let bytes = self.get_bytes(path).await?;
        serde_json::from_slice(&bytes).context("failed to parse API response")
    }

    /// GET a v0 API endpoint and return the raw response body.
    pub async fn get_raw(&self, path: &str) -> Result<String> {
        let bytes = self.get_bytes(path).await?;
        String::from_utf8(bytes).context("failed to read API response")
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tokio::net::TcpListener;
    use tokio::sync::{mpsc, watch};

    use super::*;
    use crate::api::commands::SchedulerCommand;
    use crate::api::registry::BoardRegistry;
    use crate::api::server::build_router;
    use crate::api_client::types::{BoardState, SourceState};
    use crate::board::BoardRegistration;

    /// Spin up a real HTTP server on a random port and return the API client
    /// pointed at it. The returned handles must be kept alive for the server
    /// to stay up.
    struct TestServer {
        client: Client,
        _miner_tx: watch::Sender<MinerState>,
        _board_senders: Vec<watch::Sender<BoardState>>,
        _cmd_rx: mpsc::Receiver<SchedulerCommand>,
    }

    async fn start_test_server(
        miner_state: MinerState,
        board_states: Vec<BoardState>,
    ) -> TestServer {
        let (miner_tx, miner_rx) = watch::channel(miner_state);
        let (cmd_tx, cmd_rx) = mpsc::channel::<SchedulerCommand>(16);

        let mut registry = BoardRegistry::new();
        let mut board_senders = Vec::new();
        for state in board_states {
            let (tx, rx) = watch::channel(state);
            registry.push(BoardRegistration { state_rx: rx });
            board_senders.push(tx);
        }

        let router = build_router(miner_rx, Arc::new(Mutex::new(registry)), cmd_tx);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let base_url = format!("http://{}", addr);
        let client = Client::with_base_url(base_url);

        TestServer {
            client,
            _miner_tx: miner_tx,
            _board_senders: board_senders,
            _cmd_rx: cmd_rx,
        }
    }

    #[tokio::test]
    async fn test_get_miner_state() {
        let miner_state = MinerState {
            uptime_secs: 120,
            hashrate: 5_000_000,
            shares_submitted: 42,
            paused: false,
            sources: vec![SourceState {
                name: "test-pool".into(),
                url: Some("stratum+tcp://pool:3333".into()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let board = BoardState {
            name: "test-board".into(),
            model: "TestModel".into(),
            ..Default::default()
        };

        let server = start_test_server(miner_state, vec![board]).await;

        let state = server.client.get_miner().await.unwrap();
        assert_eq!(state.uptime_secs, 120);
        assert_eq!(state.hashrate, 5_000_000);
        assert_eq!(state.shares_submitted, 42);
        assert!(!state.paused);
        assert_eq!(state.boards.len(), 1);
        assert_eq!(state.boards[0].name, "test-board");
        assert_eq!(state.sources.len(), 1);
        assert_eq!(state.sources[0].name, "test-pool");
    }

    #[tokio::test]
    async fn test_get_nonexistent_endpoint() {
        let server = start_test_server(MinerState::default(), vec![]).await;

        let result = server
            .client
            .get_json::<serde_json::Value>("nonexistent")
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("404"),
            "expected 404 in error, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn test_client_url_construction() {
        let client = Client::new();
        // The default client targets 127.0.0.1:7785
        assert_eq!(client.base_url, "http://127.0.0.1:7785");

        let client = Client::with_base_url("http://10.0.0.1:9999".to_string());
        assert_eq!(client.base_url, "http://10.0.0.1:9999");
    }

    #[tokio::test]
    async fn test_get_raw_health() {
        let server = start_test_server(MinerState::default(), vec![]).await;

        let body = server.client.get_raw("health").await.unwrap();
        assert_eq!(body, "OK");
    }
}
