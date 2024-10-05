import OpenAI from "npm:openai";
import process from "node:process";
import { developperPrompt, leaderDefaultPrompt } from "./prompt.ts";

const openai = new OpenAI();

const leader_id = "thread_3dI0rBORteg7XFSiCbOvyjBT"
const assistant_id = "asst_i4bTWicKw60VTnJjyoJzeVT1"

// const msg = "ブラウザで簡単にできるゲームを開発したいです。一人で遊べるゲームで、矢印キーで操作できます。ターゲットが動いていてそれを捕まえるだけのゲームにして";
const msg = ``;



const message = await openai.beta.threads.messages.create(
  leader_id,
  {
    role: "user",
    content:
      `${msg} ${leaderDefaultPrompt}`,
  },
);

console.log("leader message: ", message);


let jsonStr = "";

async function runStream(leader_id: string, assistant_id: string) {
  let jsonStr = "";

  return new Promise((resolve, reject) => {
    const stream = openai.beta.threads.runs.stream(leader_id, {
      assistant_id: assistant_id,
    })
      .on("textCreated", (text) => process.stdout.write("\nassistant > "))
      .on("textDelta", (textDelta, snapshot) => {
        const val = textDelta.value;
        if (val) {
          process.stdout.write(val);
          jsonStr += val;
        }
      })
      .on(
        "toolCallCreated",
        (toolCall) =>
          process.stdout.write(`\nassistant > ${toolCall.type}\n\n`),
      )
      .on("toolCallDelta", (toolCallDelta, snapshot) => {
        if (toolCallDelta.type === "code_interpreter") {
          const code_interpreter = toolCallDelta.code_interpreter;
          if (!code_interpreter) {
            return;
          }
          if (code_interpreter.input) {
            process.stdout.write(code_interpreter.input);
          }
          if (code_interpreter.outputs) {
            process.stdout.write("\noutput >\n");
            code_interpreter.outputs.forEach((output) => {
              if (output.type === "logs") {
                process.stdout.write(`\n${output.logs}\n`);
              }
            });
          }
        }
      })
      .on("end", () => {
        resolve(jsonStr);  // Streamが完了したときにresolve
      })
      .on("error", (err) => {
        reject(err);  // エラーが発生したときにreject
      });
  });
}

const result = await runStream(leader_id, assistant_id) as string;
const projectData = JSON.parse(result);

const projectDir = "./todo-app";

// ディレクトリを再帰的に作成
await Deno.mkdir(projectDir, { recursive: true });

// 4. 各ファイルを書き出し
for (const file of projectData.codes) {
  const filePath = `${projectDir}/${file.source_full_path}`;
  
  // 必要なディレクトリを作成
  const dirPath = filePath.substring(0, filePath.lastIndexOf("/"));
  await Deno.mkdir(dirPath, { recursive: true });
  
  // ファイルに内容を書き込む
  await Deno.writeTextFile(filePath, file.code);
}

console.log("プロジェクトファイルの作成が完了しました！");