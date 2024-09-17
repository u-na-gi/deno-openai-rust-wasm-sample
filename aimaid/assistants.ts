import OpenAI from "npm:openai";
import process from "node:process";

const openai = new OpenAI();

async function main() {
  const assistant = await openai.beta.assistants.create({
    name: "Math Tutor",
    instructions: "You are a personal math tutor. Write and run code to answer math questions.",
    tools: [{ type: "code_interpreter" }],
    model: "gpt-4o-mini"
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
const message = await openai.beta.threads.messages.create(
  thread.id,
  {
    role: "user",
    content: "I need to solve the equation `3x + 11 = 14`. Can you help me?"
  }
);

// We use the stream SDK helper to create a run with
// streaming. The SDK provides helpful event listeners to handle 
// the streamed response.
 
const run = openai.beta.threads.runs.stream(thread.id, {
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