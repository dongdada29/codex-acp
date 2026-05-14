//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_core::config::{Config, ConfigOverrides};
use codex_model_provider_info::{ModelProviderInfo, WireApi};
use codex_utils_cli::CliConfigOverrides;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod codex_agent;
mod thread;

const NUWACLAW_PROVIDER_ID: &str = "nuwaclaw-openai-compatible";
const OPENAI_COMPATIBLE_MODEL_PREFIX: &str = "openai-compatible/";

struct NuwaclawEnvOverrides {
    model: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
}

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_model_for_provider(model: &str) -> String {
    model
        .trim()
        .strip_prefix(OPENAI_COMPATIBLE_MODEL_PREFIX)
        .unwrap_or(model.trim())
        .trim()
        .to_string()
}

fn read_nuwaclaw_env_overrides() -> NuwaclawEnvOverrides {
    NuwaclawEnvOverrides {
        model: env_non_empty("CODEX_MODEL").map(|model| normalize_model_for_provider(&model)),
        base_url: env_non_empty("CODEX_BASE_URL"),
        api_key: env_non_empty("CODEX_API_KEY"),
    }
}

fn apply_nuwaclaw_env_overrides(config: &mut Config) {
    let overrides = read_nuwaclaw_env_overrides();

    if let Some(api_key) = overrides.api_key {
        tracing::info!("CODEX_API_KEY env var overriding OPENAI_API_KEY");
        // SAFETY: This runs during single-threaded process setup before
        // CodexAgent::new() starts async session work that can read auth env.
        unsafe { std::env::set_var("OPENAI_API_KEY", api_key) };
    }

    if let Some(model) = overrides.model {
        tracing::info!("CODEX_MODEL env var overriding config model");
        config.model = Some(model);
    }

    if let Some(base_url) = overrides.base_url {
        tracing::info!("CODEX_BASE_URL env var overriding model provider base_url");
        let mut provider = config
            .model_providers
            .get(&config.model_provider_id)
            .cloned()
            .unwrap_or_else(ModelProviderInfo::default);

        provider.name = "NuwaClaw OpenAI Compatible".to_string();
        provider.base_url = Some(base_url);
        provider.env_key = Some("OPENAI_API_KEY".to_string());
        provider.env_key_instructions = None;
        provider.experimental_bearer_token = None;
        provider.auth = None;
        provider.aws = None;
        provider.wire_api = WireApi::Responses;
        provider.requires_openai_auth = false;
        provider.supports_websockets = false;

        config.model_provider_id = NUWACLAW_PROVIDER_ID.to_string();
        config.model_provider = provider.clone();
        config
            .model_providers
            .insert(NUWACLAW_PROVIDER_ID.to_string(), provider);
    }
}

/// Run the Codex ACP agent.
///
/// This sets up an ACP agent that communicates over stdio, bridging
/// the ACP protocol with the existing codex-rs infrastructure.
///
/// # Errors
///
/// If unable to parse the config or start the program.
pub async fn run_main(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
) -> std::io::Result<()> {
    // Install a simple subscriber so `tracing` output is visible.
    // Users can control the log level with `RUST_LOG`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // Parse CLI overrides and load configuration
    let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("error parsing -c overrides: {e}"),
        )
    })?;

    let config_overrides = ConfigOverrides {
        codex_linux_sandbox_exe: codex_linux_sandbox_exe.clone(),
        ..ConfigOverrides::default()
    };

    let mut config =
        Config::load_with_cli_overrides_and_harness_overrides(cli_kv_overrides, config_overrides)
            .await
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("error loading config: {e}"),
                )
            })?;

    apply_nuwaclaw_env_overrides(&mut config);

    // Apply residency requirement so the HTTP client sends the
    // x-openai-internal-codex-residency header on all requests.
    codex_login::default_client::set_default_client_residency_requirement(
        config.enforce_residency.value(),
    );

    let agent = Arc::new(codex_agent::CodexAgent::new(config, codex_linux_sandbox_exe).await?);

    let stdin = tokio::io::stdin().compat();
    let stdout = tokio::io::stdout().compat_write();

    agent
        .serve(ByteStreams::new(stdout, stdin))
        .await
        .map_err(|e| std::io::Error::other(format!("ACP error: {e}")))?;

    Ok(())
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};

#[cfg(test)]
mod tests {
    use super::normalize_model_for_provider;

    #[test]
    fn normalize_model_strips_openai_compatible_prefix() {
        assert_eq!(
            normalize_model_for_provider("openai-compatible/glm-5"),
            "glm-5"
        );
        assert_eq!(
            normalize_model_for_provider(" openai-compatible/glm-5 "),
            "glm-5"
        );
        assert_eq!(normalize_model_for_provider("glm-5"), "glm-5");
    }
}
