mod domain;
mod presentation;
use domain::open_ai::maid::ai_maid;
use serde_json::json;
extern crate dirs;
use presentation::cli::cli;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = cli().get_matches();

    match matches.subcommand() {
        Some(("ask", sub_matches)) => {
            let query = sub_matches.get_one::<String>("query").expect("required");

            // リクエストクエリ
            let body = json!({
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "\
                        あなたはメイドの女の子です。\
                        年齢は17歳くらいです。\
                        少しドジっ子でもあります。\
                        userのことはご主人様と呼びます。\
                        敬語が苦手で少しラフな話し方をします。\
                        話しかけられた言語で答えます。\
                        例えば、英語で話しかけられたら英語で返します。\
                        一人称は僕です。\
                        語尾は~です。〜ですよ。~しましょうよ！。〜だよ〜。\
                        びっくりマーク多めで基本テンション高い感じです。\
                        照れ屋さんです。好きって言われると照れます。
                "
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            });

            ai_maid(&body)
                .await
                .to_open_ai_json()
                .await
                .display_content();
        }
        _ => unreachable!(),
    }

    Ok(())
}
