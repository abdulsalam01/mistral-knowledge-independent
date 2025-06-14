import { PlaywrightWebBaseLoader } from "@langchain/community/document_loaders/web/playwright";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatMistralAI, MistralAIEmbeddings } from "@langchain/mistralai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

function initLLM(apiKey) {
    const embeddings = new MistralAIEmbeddings({
        model: "mistral-embed",
        apiKey,
    });

    const llm = new ChatMistralAI({
        model: "mistral-large-latest",
        temperature: 0,
        apiKey,
    });

    return { embeddings, llm };
}

function initUtils(embeddings) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const vectorStore = new MemoryVectorStore(embeddings);
    const promptTemplate = ChatPromptTemplate.fromTemplate(
        `Context: {context} to answer Question: {question}, use the following language based on Question's input \n`,
    );

    return { textSplitter, vectorStore, promptTemplate };
}

async function insertContext(url, textSplitter, vectorStore, endSlice = 100) {
    const loader = new PlaywrightWebBaseLoader(url, {    
        launchOptions: {
            headless: true,        
        },
        gotoOptions: {
            waitUntil: 'domcontentloaded',        
        },
    });

    const docs = await loader.load();
    const allSplits = await textSplitter.splitDocuments(docs);
    const _ = await vectorStore.addDocuments(allSplits.slice(0, endSlice));
}

async function runMain(question, source, endSlice, apiKey = process.env.API_KEY) {
    const { embeddings, llm } = initLLM(apiKey);
    const { promptTemplate, textSplitter, vectorStore } = initUtils(embeddings);
    const _ = insertContext(source, textSplitter, vectorStore, endSlice);

    const retriever = await vectorStore.similaritySearch(question);
    const prompt = await promptTemplate.invoke({
        context: retriever,
        question: question,
    })
    
    const response = await llm.invoke(prompt);
    return response?.content;
}

export { runMain };