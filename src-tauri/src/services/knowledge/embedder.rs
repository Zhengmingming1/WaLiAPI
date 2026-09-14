use crate::db::models::Channel;
use crate::db::repository::Repository;

/// Call WaLiAPI's internal channel dispatch to get embeddings.
/// Reuses existing channel config (base_url, api_key, model_mapping) but
/// sends requests directly to the /embeddings endpoint instead of /chat/completions,
/// because all adaptors hard-code the chat completions URL.
pub async fn embed(
    texts: &[String],
    model: &str,
    repo: &Repository,
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    // 查询和文档统一修复 PDF 部首字形，覆盖 REST、MCP、管理命令入口。
    let texts: Vec<String> = texts
        .iter()
        .map(|t| super::text::normalize_radicals(t))
        .collect();

    // Get enabled channels
    let channels = repo
        .get_enabled_channels()
        .await
        .map_err(|e| format!("Failed to get channels: {}", e))?;

    // Select channels that support this model (same logic as dispatcher)
    let selected = Dispatcher::select_channels(&channels, model);

    let candidates = if selected.is_empty() {
        // Fallback: try all enabled channels
        channels.clone()
    } else {
        selected
    };

    for channel in &candidates {
        match try_embed_with_channel(&texts, model, channel).await {
            Ok(embeddings) => {
                // Log success and validate dimensions
                if !embeddings.is_empty() {
                    tracing::info!(
                        "Embedding success: channel={}, model={}, texts={}, dim={}",
                        channel.name,
                        model,
                        texts.len(),
                        embeddings[0].len()
                    );
                }
                return Ok(embeddings);
            }
            Err(e) => {
                tracing::warn!(
                    "Embedding failed on channel {} (model={}): {} — trying next channel",
                    channel.name,
                    model,
                    e
                );
                continue;
            }
        }
    }

    Err(format!(
        "All channels failed for embedding model: {}. Make sure at least one channel supports embeddings.",
        model
    ))
}

async fn try_embed_with_channel(
    texts: &[String],
    model: &str,
    channel: &Channel,
) -> Result<Vec<Vec<f32>>, String> {
    let base_url = channel.base_url.trim_end_matches('/');

    // Apply model mapping if configured
    let actual_model = apply_model_mapping(model, &channel.model_mapping);

    let url = format!("{}/embeddings", base_url);
    let body = serde_json::json!({
        "model": actual_model,
        "input": texts,
        "encoding_format": "float"
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", channel.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(format!(
            "HTTP {}: {}",
            status,
            text.chars().take(300).collect::<String>()
        ));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Parse response failed: {}", e))?;

    parse_embedding_response(&json, texts.len())
}

/// 按响应索引还原输入顺序，整批校验后才允许调用方写入文档切片。
/// 兼容完全不提供 index 的旧渠道；一旦提供索引就必须完整且唯一。
pub(crate) fn parse_embedding_response(
    response: &serde_json::Value,
    expected_count: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let data = response
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or("Invalid embedding response: missing data array")?;
    if data.len() != expected_count {
        return Err(format!(
            "Embedding count mismatch: expected {}, got {}",
            expected_count,
            data.len()
        ));
    }

    let indexed = data.iter().any(|item| item.get("index").is_some());
    let mut embeddings = vec![Vec::new(); expected_count];
    let mut dimension = None;
    for (position, item) in data.iter().enumerate() {
        let index = if indexed {
            item.get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|&index| index < expected_count)
                .ok_or_else(|| format!("Invalid embedding index at item {position}"))?
        } else {
            position
        };
        if !embeddings[index].is_empty() {
            return Err(format!("Duplicate embedding index: {index}"));
        }
        let embedding: Vec<f32> =
            serde_json::from_value(item.get("embedding").cloned().unwrap_or_default())
                .map_err(|_| format!("Embedding item {position} is not a float vector"))?;
        if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(format!("Invalid embedding vector at item {position}"));
        }
        if dimension.is_some_and(|dim| dim != embedding.len()) {
            return Err(format!(
                "Inconsistent embedding dimensions at item {position}"
            ));
        }
        dimension = Some(embedding.len());
        embeddings[index] = embedding;
    }
    Ok(embeddings)
}

fn apply_model_mapping(model: &str, mapping_json: &str) -> String {
    if mapping_json.is_empty() || mapping_json == "{}" {
        return model.to_string();
    }
    let mapping: serde_json::Value = serde_json::from_str(mapping_json).unwrap_or_default();
    if let Some(mapped) = mapping.get(model).and_then(|m| m.as_str()) {
        return mapped.to_string();
    }
    model.to_string()
}

// Re-export Dispatcher for select_channels
use crate::core::dispatcher::Dispatcher;
