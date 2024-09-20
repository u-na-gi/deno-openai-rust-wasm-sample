

参考: 
https://developers.cloudflare.com/workers-ai/tutorials/build-a-retrieval-augmented-generation-ai

https://www.notion.so/RAG-1072cb6039ff800eb1e4e752cd7b2b54?pvs=4

## Memo

```shell
npx wrangler vectorize create vector-index --dimensions=768 --metric=cosine
```

何で768次元??あとvector-indexって何


d1にドキュメントを入れる
↓
vector-indexに保存される


d1にメッセージを保存する + vectorizeに保存する
```shell
curl -X POST https://rag-ai-tutorial.orcinusorca1758dv6932.workers.dev/notes \
  -H "Content-Type: application/json" \
  -d '{"text": "the orca (killer whale) is seen as a protector of the seas. Legend says they guard fishermen, guiding them to bountiful waters, and are considered sacred creatures representing strength, intelligence, and the balance of nature."}'
```

RAGを試す
```shell
curl -X GET https://rag-ai-tutorial.orcinusorca1758dv6932.workers.dev
```

削除
```shell
curl -X DELETE https://rag-ai-tutorial.orcinusorca1758dv6932.workers.dev/notes/2
```