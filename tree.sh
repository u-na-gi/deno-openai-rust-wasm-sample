#!/bin/bash

# .gitignore ファイルが存在しない場合は終了
if [ ! -f .gitignore ]; then
  echo ".gitignore ファイルが見つかりません。"
  exit 1
fi

# .gitignore の内容を取得し、各行を -I オプション用に整形
ignore_patterns=$(grep -v '^#' .gitignore | tr '\n' '|' | sed 's/|$//')

# パターンが空でない場合に限り tree コマンドを実行
if [ -n "$ignore_patterns" ]; then
  tree -I "$ignore_patterns"
else
  tree
fi
