use clap::{arg, Command};

pub fn cli() -> Command {
    Command::new("aimaid")
        .about("AI メイドが質問にお答えします")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .allow_external_subcommands(true)
        .subcommand(
            Command::new("ask")
                .about("メイドに質問します。")
                .arg(arg!(<query> "聞きたい内容です。もし記憶がセットされていない場合、このクエリがそのまま記憶名になります。"))
                .arg_required_else_help(true),
        )
        .subcommand(
            Command::new("mem")
                .about("メモリー。会話のひとまとまり")
                .args(&[
                    arg!(-s --set <memory_name> "記憶をセットします。"),
                    arg!(-r --remove <memory_name> "現在の記憶を削除します。")
                    ])
                // .arg(arg!(-rva --remove-all "全ての記憶を削除します。"))
                // .arg(arg!(-l --list "現在保持されている記憶名を表示します。"))
                // .arg(arg!(-c --current "現在選択されている記憶名を表示します。"))
                .arg_required_else_help(true),
        )
}

// pub fn matcher() {
//     // let matches = cli().get_matches();

// }
