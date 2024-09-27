import { walk } from "jsr:@std/fs";

// .gitignoreに基づいて無視するパターンを取得する関数
export async function getGitignorePatterns(dir: string): Promise<Set<string>> {
  const patterns = new Set<string>();

  try {
    const gitignorePath = `${dir}/.gitignore`;
    const gitignoreContent = await Deno.readTextFile(gitignorePath);
    gitignoreContent.split("\n").forEach((line) => {
      const pattern = line.trim();
      if (pattern && !pattern.startsWith("#")) {
        patterns.add(pattern);
      }
    });
  } catch (error) {
    if (error.name !== "NotFound") {
      console.error(`Error reading .gitignore in ${dir}: ${error.message}`);
    }
  }

  return patterns;
}

// パターンとファイルパスが後ろから一致するかどうかを確認する関数
export function isSuffixPatternMatch(
  filePath: string,
  pattern: string,
): boolean {
  const normalizedPattern = pattern.replace(/\*\*/g, "").replace(/\*/g, "")
    .replace(/\//g, "");

  // パターンがファイルパスの末尾と一致するか確認
  return filePath.endsWith(normalizedPattern);
}

// パターンから全ての '*' を削除して、部分一致を確認する関数
export function isPartialPatternMatch(
  filePath: string,
  pattern: string,
): boolean {
  const normalizedPattern = pattern.replace(/\*/g, ""); // 全ての '*' を削除
  return filePath.includes(normalizedPattern); // 部分一致を確認
}

// 指定されたファイルパスが.gitignoreに該当するか確認
export function isIgnored(
  filePath: string,
  ignorePatterns: Set<string>,
): boolean {
  const [currentPattern, ...restPatterns] = [...ignorePatterns];

  if (!currentPattern) {
    return false; // パターンが空の場合
  }

  let isMatch = false;

  // ワイルドカードを含むパターンかどうかで処理を分岐
  if (currentPattern.includes("*")) {
    // 後ろに含まれるか確認
    if (currentPattern.endsWith("*")) {
      isMatch = isPartialPatternMatch(filePath, currentPattern);
    } else {
      // ないなら前方一致ver
      isMatch = isSuffixPatternMatch(filePath, currentPattern);
    }
  } else {
    // ないなら単純な部分一致
    isMatch = filePath.includes(currentPattern);
  }

  // 現在のパターンでマッチするか、残りのパターンを再帰的に確認
  return isMatch || isIgnored(filePath, new Set(restPatterns));
}

// Markdownファイルに出力する関数
export async function writeMarkdownOutput(
  outputPath: string,
  markdown: string,
) {
  try {
    // UTF-8 でない場合の処理
    const encoder = new TextEncoder();
    const utf8Content = encoder.encode(markdown); // UTF-8 に変換

    // UTF-8にエンコードしたコンテンツを書き込む
    await Deno.writeFile(outputPath, utf8Content);

    // 成功メッセージ
    console.log(`Markdown written to ${outputPath}`);
    return true; // 成功を示す
  } catch (error) {
    // エラー処理
    console.error(`Error writing markdown to ${outputPath}:`, error);
    return false; // エラーが発生した場合は false
  }
}
// main処理
export async function processPythonFiles(dir: string, outputPath: string) {
  let markdownContent = "";
  const ignorePatterns = new Set<string>();

  // 再起的にすべての階層で.gitignoreのパターンを収集
  for await (const entry of walk(dir, { includeDirs: true })) {
    if (entry.isDirectory) {
      const patterns = await getGitignorePatterns(entry.path);
      patterns.forEach((pattern) => ignorePatterns.add(pattern));
    }
  }

  // 再起的にすべてのファイルをチェック
  for await (const entry of walk(dir, { exts: [".py"] })) {
    if (entry.isFile && isIgnored(entry.path, ignorePatterns)) {
      const fileContent = await Deno.readTextFile(entry.path);
      markdownContent +=
        `\n- ${entry.path}\n\`\`\`python\n${fileContent}\n\`\`\`\n`;
    } else {
      console.log("Is ignored. -> ", entry.path);
    }
  }

  // Markdownファイルに書き出し
  await writeMarkdownOutput(outputPath, markdownContent);
}
