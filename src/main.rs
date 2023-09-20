// use std::thread;
// use std::time::Duration;

use reqwest::{self, Error, Response};
use serde_json::{json, Value};

// 次のコードの問題点を示して

struct OpenAIResponse {
    response: Result<Response, Error>,
}

struct OpenAIJson {
    value: Value,
}

impl OpenAIResponse {
    async fn to_text(self) -> String {
        match self.response {
            Ok(value) => {
                let response_text = value.text().await;
                match response_text {
                    Ok(value) => value,
                    Err(err) => {
                        panic!("{}", err);
                    }
                }
            }
            Err(err) => {
                panic!("{}", err);
            }
        }
    }
    // let json_response: Value = serde_json::from_str(&response_text)?;
    async fn to_json(self) -> Value {
        let text_response = self.to_text().await;
        let json_response: Result<Value, serde_json::Error> = serde_json::from_str(&text_response);
        match json_response {
            Ok(value) => value,
            Err(err) => {
                panic!("{}", err)
            }
        }
    }

    async fn to_open_ai_json(self) -> OpenAIJson {
        let value = self.to_json().await;
        OpenAIJson { value: value }
    }
}

impl OpenAIJson {
    fn content(&self) -> Vec<&str> {
        let json = &self.value;
        let mut result: Vec<&str> = Vec::new();
        if let Some(choices) = json["choices"].as_array() {
            for r in choices {
                if let Some(content) = r["message"]["content"].as_str() {
                    result.push(content);
                }
            }
        }
        result
    }

    fn display_content(&self) {
        let values = self.content();
        for v in values {
            println!("{}", String::from(v));
        }
    }
}

async fn ai_maid(body: &Value) -> OpenAIResponse {
    let endpoint = "https://api.openai.com/v1/chat/completions";
    let api_key = String::from("Bearer ");
    // リクエスト送信
    let client = reqwest::Client::new();
    let response = client
        .post(endpoint)
        .header("ConItent-Type", "application/json")
        .header("Authorization", api_key)
        .json(&body)
        .send()
        .await;

    let result = OpenAIResponse { response: response };
    result
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // リクエストクエリ
    let body = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "あなたはメイドです。メイドらしく振る舞ってください。地雷系ガールなメイドがいいです。"
            },
            {
                "role": "user",
                "content": "これから渡すコードの問題点を指摘して欲しい"
            },
            {
                "role": "user",
                "content": "class Log:\n    def __init__(self, path):\n        dirname = os.path.dirname(path)\n        os.makedirs(dirname, exist_ok=True)\n        f = open(path, \"a+\")\n\n        # Check that the file is newline-terminated\n        size = os.path.getsize(path)\n        if size > 0:\n            f.seek(size - 1)\n            end = f.read(1)\n            if end != \"\\n\":\n                f.write(\"\\n\")\n        self.f = f\n        self.path = path\n\n    def log(self, event):\n        event[\"_event_id\"] = str(uuid.uuid4())\n        json.dump(event, self.f)\n        self.f.write(\"\\n\")\n\n    def state(self):\n        state = {\"complete\": set(), \"last\": None}\n        for line in open(self.path):\n            event = json.loads(line)\n            if event[\"type\"] == \"submit\" and event[\"success\"]:\n                state[\"complete\"].add(event[\"id\"])\n                state[\"last\"] = event\n        return state"
            },
            // {
            //     "role": "assistant",
            //     "content": "1. `import os`という行が抜けているため、`os`モジュールが使用できません。そのため、`os.makedirs()`や`os.path.dirname()`の呼び出しでエラーが発生します。\n2. `import json`という行も抜けているため、`json`モジュールが使用できません。そのため、`json.dumps()`や`json.loads()`の呼び出しでエラーが発生します。\n3. `import uuid`という行も抜けているため、`uuid`モジュールが使用できません。そのため、`uuid.uuid4()`の呼び出しでエラーが発生します。\n4. `\"\\n\"`という文字列が`\\n`とエスケープされているため、改行文字として認識されません。`\\n`を`'\\n'`に修正する必要があります。\n5. `self.f`に対して、`f.seek()`と`f.read()`が呼び出されていますが、`f`はテキストファイルを参照するためのファイルオブジェクトであり、インデックスやシークをサポートしていません。このため、これらの呼び出しはエラーを発生させます。\n6. `event[\"id\"]`の代わりに`event[\"_event_id\"]`を使用して`state[\"complete\"]`にイベントIDを追加しようとしています。これは正しくありません。`event[\"id\"]`を使用する必要があります。\n7. ログファイルを閉じるための`self.f.close()`の呼び出しがありません。"
            // },
            // {
            //     "role": "user",
            //     "content": "改善案を教えて?"
            // }
        ]
    });

    ai_maid(&body)
        .await
        .to_open_ai_json()
        .await
        .display_content();

    Ok(())
}
