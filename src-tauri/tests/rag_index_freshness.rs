//! DB 已提交而后台索引尚未更新时，搜索不能丢失新切片或返回不足的结果。
use sqlx::SqlitePool;
use waliapi_lib::server::event_bridge::EventSink;
use waliapi_lib::services::knowledge::{
    repository::{ChunkInsert, KbRepository},
    retriever,
};

async fn fixture() -> (SqlitePool, String, EventSink) {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    let kb = KbRepository::new(pool.clone())
        .create_kb(&serde_json::from_value(serde_json::json!({"name": "index-freshness"})).unwrap())
        .await
        .unwrap();
    let (tx, _) = tokio::sync::broadcast::channel(100);
    (pool, kb.id, EventSink::headless(tx))
}

async fn document(repo: &KbRepository, kb: &str, text: &str, vector: &[f32]) -> String {
    let doc = repo
        .create_document(kb, "test.txt", None, "txt", text.len() as i64, text)
        .await
        .unwrap();
    replace(repo, kb, &doc.id, text, vector).await;
    doc.id
}

async fn replace(repo: &KbRepository, kb: &str, doc: &str, text: &str, vector: &[f32]) {
    repo.replace_document_chunks(
        doc,
        kb,
        &[ChunkInsert {
            id: uuid::Uuid::new_v4().to_string(),
            doc_id: doc.into(),
            kb_id: kb.into(),
            chunk_index: 0,
            content: text.into(),
            token_count: 1,
            embedding: retriever::encode_embedding(vector),
            embedding_dim: vector.len() as i64,
            metadata: "{}".into(),
            content_hash: Some(text.into()),
            created_at: "now".into(),
        }],
        None,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn replacement_is_searchable_before_the_background_index_update() {
    let (pool, kb, events) = fixture().await;
    let repo = KbRepository::new(pool.clone());
    document(&repo, &kb, "still indexed", &[0.8, 0.6]).await;
    let doc = document(&repo, &kb, "old body", &[0.0, 1.0]).await;
    retriever::build_index(&pool, &kb, &events).await.unwrap();
    replace(&repo, &kb, &doc, "new best match", &[1.0, 0.0]).await;
    let result = retriever::search(&pool, &kb, &[1.0, 0.0], 1).await.unwrap();
    retriever::drop_index(&pool, &kb).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].content, "new best match",
        "旧索引仍有其他有效命中时，也必须检索新正文"
    );
}

#[tokio::test]
async fn additions_and_deletions_do_not_hide_valid_candidates() {
    let (pool, kb, events) = fixture().await;
    let repo = KbRepository::new(pool.clone());
    let old = document(&repo, &kb, "old best", &[1.0, 0.0]).await;
    document(&repo, &kb, "remaining", &[0.8, 0.6]).await;
    retriever::build_index(&pool, &kb, &events).await.unwrap();
    document(&repo, &kb, "new best", &[1.0, 0.0]).await;
    repo.delete_document(&old).await.unwrap();
    let result = retriever::search(&pool, &kb, &[1.0, 0.0], 2).await.unwrap();
    retriever::drop_index(&pool, &kb).await.unwrap();
    assert_eq!(
        result.len(),
        2,
        "索引和数据库数量相同但切片已换过，也要完整返回有效结果"
    );
    assert_eq!(result[0].content, "new best");
}
