以下は、DenoでTypeScriptを使って指定された条件を満たすコードのサンプルです。このコードは、特定のディレクトリ内のすべてのPythonファイル（`.py`）を収集し、`.gitignore`で指定されたファイルを無視し、結果をMarkdown形式でファイルに書き込みます。

```typescript
import { walk } from "https://deno.land/std/fs/mod.ts";
import { ensureFile } from "https://deno.land/std/fs/ensure_file.ts";
import { readLines } from "https://deno.land/std/io/mod.ts";

// ディレクトリパスを引数から取得
const inputDir = Deno.args[0];

async function isIgnored(filepath: string): Promise<boolean> {
  const gitignorePath = `${filepath}/.gitignore`;
  try {
    const file = await Deno.open(gitignorePath);
    const ignoredPatterns: string[] = [];

    for await (const line of readLines(file)) {
      // .gitignoreのパターンを収集
      ignoredPatterns.push(line.trim());
    }
    Deno.close(file.rid);

    // 現在のファイル名を取得
    const filename = filepath.split('/').pop() || "";

    // .gitignoreのパターンに一致するかをチェック
    return ignoredPatterns.some(pattern => {
      if (pattern.startsWith("#") || pattern === "") return false; // コメント行や空行は無視
      return filename === pattern || filename.startsWith(pattern); // パターンに合わせた条件
    });
  } catch {
    // .gitignoreが存在しない場合は無視リストに入れない
    return false;
  }
}

async function collectPythonFiles(dir: string): Promise<string[]> {
  const pythonFiles: string[] = [];

  for await (const entry of walk(dir)) {
    if (entry.isFile && entry.path.endsWith(".py") && !(await isIgnored(entry.dir))) {
      pythonFiles.push(entry.path);
    }
  }

  return pythonFiles;
}

async function main() {
  if (!inputDir) {
    console.error("Usage: deno run --allow-read script.ts <directory_path>");
    Deno.exit(1);
  }

  const pythonFiles = await collectPythonFiles(inputDir);

  const outputMarkdown = pythonFiles.map(filepath => 
    `- ${filepath}\n\`\`\`python\n\n\`\`\`\n`
  ).join("\n");

  const outputFilePath = "output.md"; // output file path
  await ensureFile(outputFilePath);
  await Deno.writeTextFile(outputFilePath, outputMarkdown);
  console.log(`Markdown file written to ${outputFilePath}`);
}

await main();
```

### 使用方法
1. このコードは `script.ts` というファイル名で保存します。
2. Denoをインストールしていない場合、Denoの公式サイトからインストールします。
3. コマンドラインで以下のように実行します：

   ```bash
   deno run --allow-read script.ts <directory_path>
   ```

`<directory_path>`には、Pythonファイルを収集するディレクトリのパスを指定します。

### 注意点
- `.gitignore`は指定したディレクトリごとにチェックされ、該当のパターンに一致するファイルは無視されます。
- 出力はカレントディレクトリに `output.md` というファイル名で保存されます。