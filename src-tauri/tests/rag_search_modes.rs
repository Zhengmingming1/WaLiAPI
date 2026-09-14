//! 从实际管理命令与 REST 处理器验证搜索模式和混合权重。
use axum::{
    body::to_bytes,
    extract::{Query, State},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tauri::Manager;
use waliapi_lib::{
    commands::knowledge_base::search_knowledge_base,
    db::{repository::Repository, Database},
    server::{event_bridge::EventSink, router::SharedState},
    services::knowledge::{
        handlers,
        repository::{ChunkInsert, KbRepository},
        retriever,
    },
    settings_store::SettingsStore,
    AppState,
};

async fn fixture() -> (SharedState, String) {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    let data_dir = std::env::temp_dir().join(format!("rag-search-mode-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&data_dir).unwrap();
    let settings = SettingsStore::file(data_dir.join("settings.json"));
    settings
        .set_many(&[("kb.fusion_mode".into(), json!("weighted"))])
        .unwrap();
    let (tx, _) = tokio::sync::broadcast::channel(100);
    let state = Arc::new(AppState {
        db: Arc::new(Database { pool: pool.clone() }),
        auth_service: Arc::new(waliapi_lib::auth_provider::service::AuthService::new(
            Arc::new(Repository::new(pool.clone())),
            waliapi_lib::auth_provider::ProviderRegistry::new(),
        )),
        login_sessions: Arc::new(waliapi_lib::commands::auth::LoginSessions::new()),
        server_port: Arc::new(tokio::sync::RwLock::new(0)),
        server_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        server_handle: Arc::new(tokio::sync::RwLock::new(None)),
        test_receipts: Arc::new(waliapi_lib::services::channel_test::TestReceiptStore::new(
            std::time::Duration::from_secs(60),
        )),
        admin_sessions: waliapi_lib::server::admin_auth::SessionStore::new(),
        login_throttle: waliapi_lib::server::admin_auth::LoginThrottle::new(),
        events: EventSink::headless(tx),
        settings,
        data_dir,
    });
    let mock = Box::leak(Box::new(tauri::test::mock_app()));
    mock.manage(state.clone());
    let shared = SharedState {
        state,
        state_static: mock.state(),
        admin_token: None,
        mcp_token: None,
        #[cfg(feature = "desktop-ui")]
        desktop_app: None,
    };
    let repo = KbRepository::new(pool);
    let kb = repo
        .create_kb(
            &serde_json::from_value(json!({"name":"search modes", "embedding_model":"embed-test"}))
                .unwrap(),
        )
        .await
        .unwrap();
    for (text, vector) in [
        ("alpha lexical match", [0.0, 1.0]),
        ("semantic match", [1.0, 0.0]),
    ] {
        let doc = repo
            .create_document(&kb.id, "test.txt", None, "txt", text.len() as i64, text)
            .await
            .unwrap();
        repo.replace_document_chunks(
            &doc.id,
            &kb.id,
            &[ChunkInsert {
                id: uuid::Uuid::new_v4().to_string(),
                doc_id: doc.id.clone(),
                kb_id: kb.id.clone(),
                chunk_index: 0,
                content: text.into(),
                token_count: 1,
                embedding: retriever::encode_embedding(&vector),
                embedding_dim: 2,
                metadata: "{}".into(),
                content_hash: Some(text.into()),
                created_at: "now".into(),
            }],
            None,
        )
        .await
        .unwrap();
    }
    (shared, kb.id)
}

async fn results(
    shared: &SharedState,
    kb: &str,
    mode: &str,
) -> (Result<Vec<String>, String>, u16, String) {
    let command = search_knowledge_base(
        shared.state_static.clone(),
        serde_json::from_value(json!({
            "query":"alpha", "kb_id":kb, "top_k":1, "search_mode":mode,
            "vector_weight":0.0, "keyword_weight":1.0,
        }))
        .unwrap(),
    )
    .await
    .map(|results| results.into_iter().map(|r| r.content).collect());
    let response = handlers::search(
        State(shared.clone()),
        Query(
            serde_json::from_value(json!({
                "q":"alpha", "kb_id":kb, "top_k":1, "search_mode":mode,
                "vector_weight":0.0, "keyword_weight":1.0,
            }))
            .unwrap(),
        ),
        None,
    )
    .await;
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (command, status, String::from_utf8(bytes.to_vec()).unwrap())
}

#[tokio::test]
async fn keyword_search_needs_no_embedding_channel_in_command_or_rest() {
    let (shared, kb) = fixture().await;
    let (command, status, rest) = results(&shared, &kb, "keyword").await;
    std::fs::remove_dir_all(&shared.state.data_dir).unwrap();
    assert_eq!(command.unwrap(), ["alpha lexical match"]);
    assert_eq!(status, 200, "REST: {rest}");
    let rest: Value = serde_json::from_str(&rest).unwrap();
    assert_eq!(rest["data"][0]["content"], "alpha lexical match");
}

#[tokio::test]
async fn invalid_weights_are_rejected_before_searching() {
    let (shared, kb) = fixture().await;
    for (vector, keyword) in [
        (f32::NAN, 1.0),
        (1.0, f32::INFINITY),
        (-1.0, 1.0),
        (0.0, 0.0),
        (f32::MAX, f32::MAX),
    ] {
        assert!(retriever::search_query(
            &shared.state.db.pool,
            Some(&kb),
            "alpha",
            1,
            "keyword",
            vector,
            keyword,
            retriever::FusionMode::Weighted
        )
        .await
        .is_err());
    }
    std::fs::remove_dir_all(&shared.state.data_dir).unwrap();
}

#[tokio::test]
async fn hybrid_weights_and_vector_mode_reach_the_retriever() {
    let (shared, kb) = fixture().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/v1/embeddings",
        post(|| async { Json(json!({"data":[{"index":0,"embedding":[1.0,0.0]}]})) }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Repository::new(shared.state.db.pool.clone())
        .create_channel(
            &serde_json::from_value(json!({
                "name":"search-mode-test", "type":"openai", "base_url":base_url,
                "api_key":"test-only", "models":["embed-test"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();
    let vector = results(&shared, &kb, "vector").await;
    let hybrid = results(&shared, &kb, "hybrid").await;
    server.abort();
    std::fs::remove_dir_all(&shared.state.data_dir).unwrap();
    for ((command, status, rest), expected) in
        [(vector, "semantic match"), (hybrid, "alpha lexical match")]
    {
        assert_eq!(command.unwrap(), [expected]);
        assert_eq!(status, 200, "REST: {rest}");
        let rest: Value = serde_json::from_str(&rest).unwrap();
        assert_eq!(rest["data"][0]["content"], expected);
    }
}
