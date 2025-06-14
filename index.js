import { runMain } from './src/lib.js';

const apiKey = process.env.API_KEY;
const endSlice = 200;
const response = await runMain("what is mcp", "https://github.com/modelcontextprotocol", endSlice, apiKey);

console.log(response);