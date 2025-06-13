import { PlaywrightWebBaseLoader } from "@langchain/community/document_loaders/web/playwright";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import * as cheerio from 'cheerio';

const apiKey = process.env.API_KEY;
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});
const embeddings = new MistralAIEmbeddings({
    model: "mistral-embed",
    apiKey,
});
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  temperature: 0,
  apiKey,
});
const loader = new PlaywrightWebBaseLoader("https://snov.io/", {    
    launchOptions: {
        headless: true,        
    },
    gotoOptions: {
        waitUntil: 'domcontentloaded',        
    },
});
// const loader = new CheerioWebBaseLoader(
//     "https://erlang.org/course/history.html",
//     {
//         maxRetries: 2,
//         selector: 'p',        
//     },
// );
const vectorStore = new MemoryVectorStore(embeddings);
const promptTemplate = ChatPromptTemplate.fromTemplate(
    `Context: {context} to answer Question: {question}, use the following language based on Question's input \n`,
);

const docs = await loader.load();
const allSplits = await textSplitter.splitDocuments(docs);
const _ = await vectorStore.addDocuments(allSplits.slice(0, 100));

const question = 'what is snov.io?';
const retriever = await vectorStore.similaritySearch(question);
const prompt = await promptTemplate.invoke({
    context: retriever,
    question: question,
})
const response = await llm.invoke(prompt);

console.log(response.content)