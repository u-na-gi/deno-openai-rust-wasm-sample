import { processPythonFiles } from "./concat_source_code_py.ts";

// 使用例
const inputDir = "/app/deno-sample/doll"; // inputに渡されるディレクトリパス
const outputMarkdownPath = "./concat-source-py.md"; // 出力するMarkdownファイル
await processPythonFiles(inputDir, outputMarkdownPath);

// ファイルサイズを取得してMBで表示
try {
  const fileInfo = await Deno.stat(outputMarkdownPath);

  // ファイルサイズをバイトからMBに変換
  const fileSizeMB = fileInfo.size / (1024 * 1024);
  console.log(`Markdown file size: ${fileSizeMB.toFixed(2)} MB`);
} catch (error) {
  console.error(`Error getting file size: ${error.message}`);
}
