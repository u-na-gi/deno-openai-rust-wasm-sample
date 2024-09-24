import {processPythonFiles} from "./concat_source_code_py.ts"

// 使用例
const inputDir = "~/workspace/doll/doll"; // inputに渡されるディレクトリパス
const outputMarkdownPath = "./output.md"; // 出力するMarkdownファイル
await processPythonFiles(inputDir, outputMarkdownPath);