import {processPythonFiles} from "./concat_source_code_py.ts"

// 使用例
const inputDir = "/app/deno-sample/doll"; // inputに渡されるディレクトリパス
const outputMarkdownPath = "./concat-source-py.md"; // 出力するMarkdownファイル
await processPythonFiles(inputDir, outputMarkdownPath);