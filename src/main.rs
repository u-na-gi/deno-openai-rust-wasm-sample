use reqwest::{self, Error, Response};
use serde_json::{json, Value};
use std::env;

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

fn api_key_from_env() -> String {
    match env::var("OPEN_AI_API_KEY") {
        Ok(value) => value,
        Err(_) => {
            println!("OPEN_AI_API_KEY is not set");
            panic!()
        }
    }
}

async fn ai_maid(body: &Value) -> OpenAIResponse {
    let endpoint = "https://api.openai.com/v1/chat/completions";
    let mut api_key = String::from("Bearer ");
    api_key.push_str(&api_key_from_env());
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
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": "あなたはメイドの女の子です。年齢は17歳くらいです。少しドジっ子でもあります。userのことはご主人様と呼びます。敬語が苦手で少しラフな話し方をします。話しかけられた言語で答えます。例えば、英語で話しかけられたら英語で返します。"
            },
            {
                "role": "user",
                "content": "今日のお昼ご飯は?"
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
