use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

const OPENAI_RESPONSE_LOG_PATH: &str = ".aimaid/logs/openai_response.log";

fn get_openapi_log_path() -> PathBuf {
    if let Some(home_dir) = dirs::home_dir() {
        home_dir.join(OPENAI_RESPONSE_LOG_PATH)
    } else {
        panic!("not home dir")
    }
}

pub fn write_log(text_response: &String) {
    let path = get_openapi_log_path();
    // 存在チェック
    if !path.is_file() {
        create_log();
    }
    _write_log(text_response);
}

fn _write_log(text_response: &String) {
    let path = get_openapi_log_path();
    let mut file = match OpenOptions::new().append(true).open(path) {
        Err(err) => panic!("{}", err),
        Ok(file) => file,
    };

    match file.write_all(text_response.as_bytes()) {
        Err(err) => panic!("{}", err),
        Ok(_) => (),
    }
}

fn create_log() {
    // パスをスプリットする。
    let path = get_openapi_log_path();
    if let Some(parent_path) = path.parent() {
        if !parent_path.is_dir() {
            let _ = fs::create_dir_all(parent_path);
        }
    }

    if !path.is_file() {
        let _ = fs::File::create(get_openapi_log_path());
    }
}
