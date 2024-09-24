const text = await Deno.readTextFile("/app/aimaid/input.txt");

console.log(text);

await Deno.writeTextFile("/app/aimaid/output.md", text);
