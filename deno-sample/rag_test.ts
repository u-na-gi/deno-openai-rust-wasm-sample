import OpenAI from "npm:openai";
// npm i compute-cosine-similarity
// import { cosineSimilarity } from "npm:compute-cosine-similarity";

// OpenAI APIキーを設定
const OPENAI_API_KEY = 'your-openai-api-key';

// ドキュメントサンプル（やや複雑な内容）
const documents = [
  "犬は肉を食べます。犬は忠実なペットであり、人間に使われることが多い。",
  "猫は肉を食べる傾向があり、独立した性格を持つ動物です。家庭で飼われることが多い。",
  "猿は果物、特にバナナを好んで食べます。猿は高度な社会性を持つ動物です。",
  "ライオンは肉を主に食べる捕食者で、サバンナに生息しています。非常に強力なハンターです。",
  "クマは雑食ですが、主に魚や肉を食べることが多いです。クマは力強く、危険な動物とされています。",
  "トラは肉食性の大型猫科動物で、単独で狩りをします。彼らは強力な捕食者です。",
  "象は主に草食性で、果物や葉を食べます。非常に賢く、社会的な動物です。",
  "鳥は種によって異なりますが、昆虫や果物、場合によっては小さな動物も食べます。",
  "魚は主にプランクトンや小さな生物を食べることが多いが、肉食の種も存在します。"
];

// OpenAI Embedding APIを使って各ドキュメントのエンベディングを取得
async function getEmbedding(text: string): Promise<number[]> {
  const response = await axios.post(
    "https://api.openai.com/v1/embeddings",
    {
      model: "text-embedding-ada-002",  // 埋め込み用のモデル
      input: text
    },
    {
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      }
    }
  );
  return response.data.data[0].embedding;
}

// ドキュメント全体のエンベディングを取得
function getDocumentEmbeddings(docs: string[]): Promise<number[][]> {
  return Promise.all(docs.map(doc => getEmbedding(doc)));
}

// クエリに最も近い上位3つのドキュメントを検索
function findTop3Similar(queryEmbedding: number[], documentEmbeddings: number[][]): number[] {
  let similarities = documentEmbeddings.map((embedding, index) => {
    return {
      index: index,
      similarity: cosineSimilarity(queryEmbedding, embedding)
    };
  });

  // 類似度が高い順にソートして上位3つを取得
  similarities.sort((a, b) => b.similarity - a.similarity);
  
  // 上位3つのドキュメントのインデックスを返す
  return similarities.slice(0, 3).map(item => item.index);
}

// メイン関数
async function main() {
  // クエリ
  const query = "肉食動物";
  const queryEmbedding = await getEmbedding(query);

  // ドキュメントのエンベディングを取得
  const documentEmbeddings = await getDocumentEmbeddings(documents);

  // 上位3つのドキュメントを取得
  const top3Indices = findTop3Similar(queryEmbedding, documentEmbeddings);

  // 結果を表示
  console.log("クエリ:", query);
  console.log("関連するドキュメント上位3:");
  top3Indices.forEach(index => {
    console.log(`- ${documents[index]}`);
  });
}

// 実行
main();
