use serde_json::Value;
use super::cast::OpenAIResponse;
use super::key::api_key_from_env;

pub async fn ai_maid(body: &Value) -> OpenAIResponse {
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