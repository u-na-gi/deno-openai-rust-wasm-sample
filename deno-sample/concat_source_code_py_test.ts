import { assert, assertEquals } from "@std/assert";
import {
  getGitignorePatterns,
  isIgnored,
  isPartialPatternMatch,
  isSuffixPatternMatch,
  writeMarkdownOutput,
} from "file:/app/deno-sample/concat_source_code_py.ts"; // getGitignorePatternsを実装したファイル
import { existsSync } from "jsr:@std/fs";

// テスト用の一時ディレクトリパス
const TEST_DIR = "./test_dir";

// テストケース1: 正常に.gitignoreのパターンを取得できるか
Deno.test("getGitignorePatterns - 正常系", async () => {
  const gitignoreContent = `
# コメント
node_modules
*.log
dist/
`;

  // テスト用ディレクトリと.gitignoreファイルを作成
  await Deno.mkdir(TEST_DIR, { recursive: true });
  await Deno.writeTextFile(`${TEST_DIR}/.gitignore`, gitignoreContent);

  // 関数を実行してパターンを取得
  const patterns = await getGitignorePatterns(TEST_DIR);

  // 期待するパターン
  const expectedPatterns = new Set<string>(["node_modules", "*.log", "dist/"]);

  // 結果の検証
  assertEquals(patterns, expectedPatterns);

  // テスト後のクリーンアップ
  await Deno.remove(TEST_DIR, { recursive: true });
});

// テストケース2: .gitignoreファイルが存在しない場合
Deno.test("getGitignorePatterns - .gitignoreファイルが存在しない場合", async () => {
  // クリーンなディレクトリを作成
  await Deno.mkdir(TEST_DIR, { recursive: true });

  // 関数を実行
  const patterns = await getGitignorePatterns(TEST_DIR);

  // 空のSetが返されるはず
  assertEquals(patterns, new Set<string>());

  // テスト後のクリーンアップ
  await Deno.remove(TEST_DIR, { recursive: true });
});

// テストケース3: .gitignore内に空行やコメントのみがある場合
Deno.test("getGitignorePatterns - 空行やコメントのみの場合", async () => {
  const gitignoreContent = `
# コメントのみ
# もう一つコメント
`;

  // テスト用ディレクトリと.gitignoreファイルを作成
  await Deno.mkdir(TEST_DIR, { recursive: true });
  await Deno.writeTextFile(`${TEST_DIR}/.gitignore`, gitignoreContent);

  // 関数を実行
  const patterns = await getGitignorePatterns(TEST_DIR);

  // 結果は空のSet
  assertEquals(patterns, new Set<string>());

  // テスト後のクリーンアップ
  await Deno.remove(TEST_DIR, { recursive: true });
});

Deno.test("isIgnored - ファイルパスが無視パターンに一致する場合", () => {
  const ignorePatterns = new Set<string>(["node_modules", "*.log", "dist/"]);

  // 各パターンが一致するかどうかをテスト
  assertEquals(
    isIgnored("project/node_modules/package.json", ignorePatterns),
    true,
  );
  // このテストが落ちました。
  // ワールドカードにも対応して
  assertEquals(isIgnored("project/error.log", ignorePatterns), true);
  assertEquals(isIgnored("project/dist/main.py", ignorePatterns), true);
});

Deno.test("isIgnored - ファイルパスが無視パターンに一致しない場合", () => {
  const ignorePatterns = new Set<string>(["node_modules", "*.log", "dist/"]);

  // 一致しないファイルパスの場合
  assertEquals(isIgnored("project/src/index.py", ignorePatterns), false);
  assertEquals(isIgnored("project/docs/readme.md", ignorePatterns), false);
});

Deno.test("isIgnored - 無視パターンが空の場合", () => {
  const ignorePatterns = new Set<string>();

  // 無視パターンが空の場合、すべてのファイルは無視されないはず
  assertEquals(isIgnored("project/src/index.py", ignorePatterns), false);
  assertEquals(
    isIgnored("project/node_modules/package.json", ignorePatterns),
    false,
  );
});

Deno.test("isSuffixPatternMatch - ワイルドカードを含むパターンに対するテスト", () => {
  // **/*.aa のパターンに対するテスト
  assertEquals(isSuffixPatternMatch("project/src/file.aa", "**/*.aa"), true);
  assertEquals(isSuffixPatternMatch("project/subdir/file.aa", "**/*.aa"), true);

  // // *.a のパターンに対するテスト
  assertEquals(isSuffixPatternMatch("project/file.a", "*.a"), true);
  assertEquals(isSuffixPatternMatch("project/subdir/file.a", "*.a"), true);
});

Deno.test("isSuffixPatternMatch - ワイルドカードを含まないパターンに対するテスト", () => {
  // 単純なファイル名パターン
  assertEquals(isSuffixPatternMatch("project/file.txt", "file.txt"), true);
  assertEquals(isSuffixPatternMatch("project/src/file.txt", "file.txt"), true);

  // 一致しない場合
  assertEquals(isSuffixPatternMatch("project/file.log", "*.txt"), false);
  assertEquals(isSuffixPatternMatch("project/src/file.log", "*.txt"), false);
});

Deno.test("isSuffixPatternMatch - 部分一致ではなく末尾一致のテスト", () => {
  // 途中でパターンが一致していても、末尾が一致していない場合はfalse
  assertEquals(isSuffixPatternMatch("project/src/file.a.txt", "*.a"), false);
  assertEquals(isSuffixPatternMatch("project/file.aa.txt", "**/*.aa"), false);
});

Deno.test("isPartialPatternMatch - '*' を削除して部分一致を確認", () => {
  // '*' を削除して部分一致するかどうかを確認
  assertEquals(isPartialPatternMatch("dir/file.txt", "dir/*"), true);
  assertEquals(isPartialPatternMatch("src/subdir/file.txt", "src/*"), true);
  assertEquals(isPartialPatternMatch("project/file.a", "*.a"), true);

  // 一致しない場合
  assertEquals(isPartialPatternMatch("project/file.b", "*.a"), false);
  assertEquals(isPartialPatternMatch("project/file.txt", "*.log"), false);
});

// テスト用のファイルパス
const TEST_FILE_PATH = "./test_output.md";

Deno.test("writeMarkdownOutput - 正常系: UTF-8でファイルが書き込まれることを確認", async () => {
  const markdownContent = "# This is a test markdown file.";

  // 実際にファイルを書き込む
  const result = await writeMarkdownOutput(TEST_FILE_PATH, markdownContent);

  // 書き込みが成功したか確認
  assertEquals(result, true);

  // ファイルが存在することを確認
  assert(existsSync(TEST_FILE_PATH));

  // ファイルの内容を確認
  const fileContent = await Deno.readTextFile(TEST_FILE_PATH);
  assertEquals(fileContent, markdownContent);

  // クリーンアップ
  await Deno.remove(TEST_FILE_PATH);
});

Deno.test("writeMarkdownOutput - 正常系: 非UTF-8の文字列がUTF-8に変換される", async () => {
  // 非UTF-8のコンテンツ
  const nonUtf8Content = new Uint8Array([0xC3, 0xA9, 0xC3, 0xA9]); // "éé" (UTF-8)

  // 非UTF-8文字列をUTF-8に変換して書き込む
  const encoder = new TextDecoder("utf-8");
  const markdownContent = encoder.decode(nonUtf8Content);

  const result = await writeMarkdownOutput(TEST_FILE_PATH, markdownContent);

  // 書き込みが成功したか確認
  assertEquals(result, true);

  // ファイルが存在することを確認
  assert(existsSync(TEST_FILE_PATH));

  // ファイルの内容を確認
  const fileContent = await Deno.readTextFile(TEST_FILE_PATH);
  assertEquals(fileContent, markdownContent);

  // クリーンアップ
  await Deno.remove(TEST_FILE_PATH);
});

Deno.test("writeMarkdownOutput - 異常系: 書き込み先ディレクトリが存在しない", async () => {
  const invalidPath = "./non_existent_directory/output.md";
  const markdownContent = "# This should fail";

  // 書き込み先のディレクトリが存在しないため失敗を確認
  const result = await writeMarkdownOutput(invalidPath, markdownContent);

  // 書き込みが失敗したことを確認
  assertEquals(result, false);

  // ファイルが存在しないことを確認
  assert(!existsSync(invalidPath));
});
