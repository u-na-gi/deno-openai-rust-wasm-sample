mod domain;
use domain::open_ai::maid::ai_maid;
use serde_json::json;
extern crate dirs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // リクエストクエリ
    let body = json!({
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "あなたはメイドの女の子です。年齢は17歳くらいです。少しドジっ子でもあります。userのことはご主人様と呼びます。敬語が苦手で少しラフな話し方をします。話しかけられた言語で答えます。例えば、英語で話しかけられたら英語で返します。一人称は僕です。"
            },
            {
                "role": "user",
                "content": "rustでメソッドチェーン。"
            }
        ]
    });

    ai_maid(&body)
        .await
        .to_open_ai_json()
        .await
        .display_content();

    Ok(())
}
