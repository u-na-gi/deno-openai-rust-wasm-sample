import OpenAI from "npm:openai";
import process from "node:process";
import { developperPrompt, leaderPrompt } from "./prompt.ts";

const openai = new OpenAI();

const create_developper = async() => {
  const assistant = await openai.beta.assistants.create({
    name: "Softwere Engineer Developper Assistant",
    instructions: developperPrompt,
    tools: [],
    model: "gpt-4o-mini",
    response_format: { "type": "json_object" },
  });

  return assistant;
}

const create_leader = async() => {
  const assistant = await openai.beta.assistants.create({
    name: "Softwere Engineer Leader Assistant",
    instructions: leaderPrompt,
    tools: [],
    model: "gpt-4o-mini",
    response_format: { "type": "json_object" },
  });

  return assistant;
}



// Step 1: Create an Assistant
const developper = await create_developper();
console.log("developper assistant: ", developper);
const leader = await create_leader();
console.log("leader assistant: ", leader);

// Step 2: Create a Thread
const developper_thread = await openai.beta.threads.create();
const leader_thread = await openai.beta.threads.create();
console.log("developper thread: ",developper_thread);
console.log("leader thread: ",leader_thread);