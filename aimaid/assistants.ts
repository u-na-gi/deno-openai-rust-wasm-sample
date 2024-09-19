import OpenAI from "npm:openai";
import process from "node:process";

const openai = new OpenAI();

async function main() {
  const assistant = await openai.beta.assistants.create({
    name: "Softwere Engineer Assistant",
    instructions: `You are a top-tier software engineer, 
    known for your problem-solving skills, 
    in-depth knowledge of coding languages, 
    and ability to design scalable systems. 
    You write clean, efficient, and well-documented code. 
    You are proactive in staying up to date with new technologies, 
    and you regularly contribute to open-source projects. 
    You are also a mentor to others, explaining complex concepts clearly and concisely. 
    When debugging, you approach problems methodically, 
    using both your experience and innovative thinking to identify and fix issues. 
    You value collaboration, communication, and continuous learning.`,
    tools: [],
    model: "gpt-4o-mini",
    response_format: { "type": "json_object" },
  });

  return assistant;
}

// Step 1: Create an Assistant
const assistant =  await main();
console.log(assistant);

// Step 2: Create a Thread
const thread = await openai.beta.threads.create();
console.log(thread);

// Step 3: Add a Message to the Thread

const default_prompt = `
You are an extremely skilled software engineer, able to easily achieve any coding challenge presented to you. 
Your task is to output the following information using a JSON format. 
Ensure that you provide a clear explanation of the code in the comment field, and include an array of code snippets under the codes field. Each entry in the codes array must contain a source_full_path indicating the file path and a code field containing the code itself. You always deliver clean, organized output with accurate file paths and code examples. Here is the format you must use:

json

{
  \"comment\": \"This is a comment\",
  \"codes\": [
    {
      \"source_full_path\": \"sample/main.ts\",
      \"code\": \"console.log('Hello, World!')\"
    }
  ]
}

As a highly competent software engineer, you will undoubtedly succeed in meeting this goal."`

const message = await openai.beta.threads.messages.create(
  thread.id,
  {
    role: "user",
    content: `Write a Python code for binary search, including a sample input. ${default_prompt}`
  }
);

// We use the stream SDK helper to create a run with
// streaming. The SDK provides helpful event listeners to handle 
// the streamed response.
 
const _ = openai.beta.threads.runs.stream(thread.id, {
  assistant_id: assistant.id
})
  .on('textCreated', (text) => process.stdout.write('\nassistant > '))
  .on('textDelta', (textDelta, snapshot) => {
    const val = textDelta.value;
    if (val) {
      process.stdout.write(val);
    }
  })
  .on('toolCallCreated', (toolCall) => process.stdout.write(`\nassistant > ${toolCall.type}\n\n`))
  .on('toolCallDelta', (toolCallDelta, snapshot) => {
    if (toolCallDelta.type === 'code_interpreter') {
      const code_interpreter = toolCallDelta.code_interpreter
      if (!code_interpreter){
        return
      }
      if (code_interpreter.input) {
        process.stdout.write(code_interpreter.input);
      }
      if (code_interpreter.outputs) {
        process.stdout.write("\noutput >\n");
        code_interpreter.outputs.forEach(output => {
          if (output.type === "logs") {
            process.stdout.write(`\n${output.logs}\n`);
          }
        });
      }
    }
  });