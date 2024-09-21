import { Hono } from "hono";
const app = new Hono();

app.post("/notes", async (c) => {
  const { text } = await c.req.json();
  if (!text) {
    return c.text("Missing text", 400);
  }

  const { results } = await c.env.DB.prepare(
    "INSERT INTO notes (text) VALUES (?) RETURNING *",
  )
    .bind(text)
    .run();

  const record = results.length ? results[0] : null;

  if (!record) {
    return c.text("Failed to create note", 500);
  }

  const { data } = await c.env.AI.run("@cf/baai/bge-base-en-v1.5", {
    text: [text],
  });
  const values = data[0];

  if (!values) {
    return c.text("Failed to generate vector embedding", 500);
  }

  const { id } = record;
  const inserted = await c.env.VECTOR_INDEX.upsert([
    {
      id: id.toString(),
      values,
    },
  ]);

  return c.json({ id, text, inserted });
});

app.get('/', async (c) => {
  const question = c.req.query('text') || "What is the square root of 9?"
  console.log(question)

  // Embeddingsを取得
  const embeddings = await c.env.AI.run('@cf/baai/bge-base-en-v1.5', { text: question })
  const vectors = embeddings.data[0]

  // ベクトルクエリを実行
  const vectorQuery = await c.env.VECTOR_INDEX.query(vectors, { topK: 1 });
  console.log("vectorQuery", vectorQuery)
  const vecId = vectorQuery.matches[0].id

  // 保存されたノートを取得
  let notes = []
  if (vecId) {
    const query = `SELECT * FROM notes WHERE id = ?`
    const { results } = await c.env.DB.prepare(query).bind(vecId).all()
    if (results) notes = results.map(vec => vec.text)
  }

  // 文脈メッセージを作成
  const contextMessage = notes.length
    ? `Here are some related notes:\n${notes.map(note => `- ${note}`).join("\n")}`
    : "No relevant notes were found."

  // プロンプトを修正
  const systemPrompt = `You are an assistant with access to user notes. Use the following notes to help answer the question, if relevant:\n${contextMessage}`

  // AIに質問を投げる
  const { response: answer } = await c.env.AI.run(
    '@cf/meta/llama-3-8b-instruct',
    {
      messages: [
        ...(notes.length ? [{ role: 'system', content: contextMessage }] : []),
        { role: 'system', content: systemPrompt },
        { role: 'user', content: question }
      ]
    }
  )

  return c.text(answer);
});


app.delete('/notes/:id', async (c) => {
  const { id } = c.req.param();

	console.log("id", id)
	console.log("c.env.DATABASE", c.env.DB)
  const query = `DELETE FROM notes WHERE id = ?`
  await c.env.DB.prepare(query).bind(id).run()

  return c.status(204)
})

app.onError((err, c) => {
  return c.text(err);
});

export default app;