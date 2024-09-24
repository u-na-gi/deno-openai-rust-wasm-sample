import OpenAI from "npm:openai";
const openai = new OpenAI();

const text = await Deno.readTextFile("/app/aimaid/input.txt");

const completion = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    {
      role: "user",
      content: text,
    },
  ],
});

const message = completion.choices[0].message;

const content = message.content;
if (content === null) {
  throw new Error("No content in message");
}
await Deno.writeTextFile("/app/aimaid/output.md", content);
