//! 使用真实 HTTP 响应验证向量与输入文本的对应关系和异常响应保护。
use axum::{routing::post, Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use waliapi_lib::{db::repository::Repository, services::knowledge::embedder};

async fn model(response: Value) -> (Repository, Arc<Mutex<Value>>, tokio::task::JoinHandle<()>) {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    let response = Arc::new(Mutex::new(response));
    let state = response.clone();
    let app = Router::new().route(
        "/v1/embeddings",
        post(move || {
            let state = state.clone();
            async move { Json(state.lock().unwrap().clone()) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let repo = Repository::new(pool);
    repo.create_channel(
        &serde_json::from_value(json!({
            "name": "embedding-response-test", "type": "openai", "base_url": base_url,
            "api_key": "test-only", "models": ["embed-test"]
        }))
        .unwrap(),
    )
    .await
    .unwrap();
    (repo, response, server)
}

#[tokio::test]
async fn embedding_indexes_preserve_input_order() {
    let (repo, _, server) = model(json!({"data": [
        {"index": 1, "embedding": [0.0, 1.0]},
        {"index": 0, "embedding": [1.0, 0.0]}
    ]}))
    .await;
    let actual = embedder::embed(&["first".into(), "second".into()], "embed-test", &repo)
        .await
        .unwrap();
    server.abort();
    assert_eq!(
        actual,
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        "响应数组顺序不能改变向量所属的输入文本"
    );
}

#[tokio::test]
async fn malformed_embeddings_are_rejected_as_a_whole() {
    let (repo, response, server) = model(json!({})).await;
    for (reason, data) in [
        (
            "empty vector",
            json!([{"index": 0, "embedding": []}, {"index": 1, "embedding": []}]),
        ),
        (
            "non-number",
            json!([{"index": 0, "embedding": [1.0, "bad"]}, {"index": 1, "embedding": [1.0, "bad"]}]),
        ),
        (
            "overflow",
            json!([{"index": 0, "embedding": [1e100]}, {"index": 1, "embedding": [1e100]}]),
        ),
        (
            "duplicate index",
            json!([{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]),
        ),
        (
            "out of bounds",
            json!([{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}]),
        ),
        (
            "missing index",
            json!([{"index": 0, "embedding": [1.0]}, {"embedding": [2.0]}]),
        ),
        ("wrong count", json!([{"index": 0, "embedding": [1.0]}])),
        (
            "dimension mismatch",
            json!([{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]),
        ),
    ] {
        *response.lock().unwrap() = json!({"data": data});
        assert!(
            embedder::embed(&["first".into(), "second".into()], "embed-test", &repo)
                .await
                .is_err(),
            "必须拒绝整批异常向量: {reason}"
        );
    }
    server.abort();
}

#[tokio::test]
async fn legacy_responses_without_any_index_keep_array_order() {
    let (repo, _, server) = model(json!({"data": [
        {"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}
    ]}))
    .await;
    let actual = embedder::embed(&["first".into(), "second".into()], "embed-test", &repo)
        .await
        .unwrap();
    server.abort();
    assert_eq!(actual, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
}
