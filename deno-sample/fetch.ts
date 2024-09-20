// Output: JSON Data
const jsonResponse = await fetch("https://api.github.com/users/denoland");
const jsonData = await jsonResponse.json();

console.log(jsonData, "\n");

// Output: HTML Data
const textResponse = await fetch("https://deno.land/");
const textData = await textResponse.text();

console.log(textData, "\n");

// Output: Error Message
try {
  await fetch("https://does.not.exist/");
} catch (error) {
  console.log(error);
}
