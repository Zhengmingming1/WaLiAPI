//! 导入失败后恢复渠道配置，再次导入相同文件应可恢复；有效文档仍去重。
use axum::{routing::post, Json, Router};
use serde_json::json;
use sqlx::SqlitePool;
use waliapi_lib::{
    db::repository::Repository,
    server::event_bridge::EventSink,
    services::knowledge::{importer, repository::KbRepository, retriever},
    settings_store::SettingsStore,
};

async fn pool() -> SqlitePool {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    pool
}

#[tokio::test]
async fn only_failed_documents_stop_blocking_duplicate_imports() {
    let repo = KbRepository::new(pool().await);
    let kb = repo
        .create_kb(&serde_json::from_value(json!({"name":"retry"})).unwrap())
        .await
        .unwrap();
    let doc = repo
        .create_document(&kb.id, "file.txt", None, "txt", 1, "same-hash")
        .await
        .unwrap();
    for status in ["pending", "processing", "ready"] {
        repo.update_document_status(&doc.id, status, None)
            .await
            .unwrap();
        assert_eq!(
            repo.find_document_by_hash(&kb.id, "same-hash")
                .await
                .unwrap()
                .unwrap()
                .id,
            doc.id
        );
    }
    repo.update_document_status(&doc.id, "failed", Some("embedding unavailable"))
        .await
        .unwrap();
    assert!(repo
        .find_document_by_hash(&kb.id, "same-hash")
        .await
        .unwrap()
        .is_none());
    let retry = repo
        .create_document(&kb.id, "file.txt", None, "txt", 1, "same-hash")
        .await
        .unwrap();
    assert_eq!(
        repo.find_document_by_hash(&kb.id, "same-hash")
            .await
            .unwrap()
            .unwrap()
            .id,
        retry.id
    );
    assert_eq!(
        repo.get_document(&doc.id).await.unwrap().status,
        "failed",
        "历史失败记录仍可查看"
    );
}

#[tokio::test]
async fn local_import_recovers_after_embedding_channel_is_configured() {
    let pool = pool().await;
    let repo = KbRepository::new(pool.clone());
    let kb = repo
        .create_kb(
            &serde_json::from_value(json!({
                "name":"retry", "embedding_model":"embed-test"
            }))
            .unwrap(),
        )
        .await
        .unwrap();
    let directory = std::env::temp_dir().join(format!("rag-retry-{}", kb.id));
    let source = directory.join("source");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("file.txt"), "same document for a retry").unwrap();
    let settings = SettingsStore::file(directory.join("settings.json"));
    let input =
        serde_json::from_value(json!({"source_type":"local_dir", "dir_path":source})).unwrap();
    let (tx, _) = tokio::sync::broadcast::channel(100);
    let events = EventSink::headless(tx);
    let mut receiver = events.subscribe();
    let first = importer::import_local_dir(
        &pool, &events, &kb.id, "source", &input, &settings, &directory,
    )
    .await
    .unwrap();
    assert_eq!(first, 0);
    assert_eq!(
        repo.get_documents(&kb.id).await.unwrap()[0].status,
        "failed"
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/v1/embeddings",
        post(|| async { Json(json!({"data":[{"index":0,"embedding":[1.0,0.0]}]})) }),
    );
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Repository::new(pool.clone())
        .create_channel(
            &serde_json::from_value(json!({
                "name":"retry-channel", "type":"openai", "base_url":base_url,
                "api_key":"test-only", "models":["embed-test"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();
    let retried = importer::import_local_dir(
        &pool, &events, &kb.id, "source", &input, &settings, &directory,
    )
    .await
    .unwrap();
    if retried > 0 {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let event = receiver.recv().await.unwrap();
                if event.event == "kb-index-progress" && event.payload["status"] == "ready" {
                    break;
                }
            }
        })
        .await
        .unwrap();
    }
    let repeated = importer::import_local_dir(
        &pool, &events, &kb.id, "source", &input, &settings, &directory,
    )
    .await
    .unwrap();
    let docs = repo.get_documents(&kb.id).await.unwrap();
    retriever::drop_index(&pool, &kb.id).await.unwrap();
    server.abort();
    std::fs::remove_dir_all(directory).unwrap();
    assert_eq!(retried, 1, "相同内容的失败记录不能让重试被跳过");
    assert_eq!(repeated, 0, "成功后再次导入仍需去重");
    assert_eq!(docs.iter().filter(|doc| doc.status == "ready").count(), 1);
}
