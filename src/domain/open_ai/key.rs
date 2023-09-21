use std::env;

pub fn api_key_from_env() -> String {
    match env::var("OPEN_AI_API_KEY") {
        Ok(value) => value,
        Err(_) => {
            println!("OPEN_AI_API_KEY is not set");
            panic!()
        }
    }
}
