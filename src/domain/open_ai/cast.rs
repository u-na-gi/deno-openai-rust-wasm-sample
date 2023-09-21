use super::log::write_log;
use reqwest::{Error, Response};
use serde_json::Value;

pub struct OpenAIResponse {
    pub response: Result<Response, Error>,
}

pub struct OpenAIJson {
    value: Value,
}

impl OpenAIResponse {
    async fn to_text(self) -> String {
        let response = match self.response {
            Err(err) => panic!("{}", err),
            Ok(response) => response
        };
        let response_text = match response.text().await {
            Err(err) => panic!("{}", err),
            Ok(response_text) => response_text
        };

        response_text
    }

    async fn to_json(self) -> Value {
        let text_response = self.to_text().await;
        write_log(&text_response);
        let json_response: Result<Value, serde_json::Error> = serde_json::from_str(&text_response);
        match json_response {
            Ok(value) => value,
            Err(err) => {
                panic!("{}", err)
            }
        }
    }

    pub async fn to_open_ai_json(self) -> OpenAIJson {
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

    pub fn display_content(&self) {
        let values = self.content();
        for v in values {
            println!("{}", String::from(v));
        }
    }
}
