import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { BaseLanguageModel } from "langchain/base_language";
import { OpenAI } from "langchain/llms/openai";

export interface GenerateAnswerInput {
  question: string;
  context: string;
}


export default function initGenerateAnswerChain(
  llm: BaseLanguageModel
): RunnableSequence<GenerateAnswerInput, string> {
  const answerQuestionPrompt = PromptTemplate.fromTemplate(`
    Use the following context to help answer the following question.

    Question:
    {question}

    Context:
    {context}

    Answer as if you have been asked the original question.

    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Include links and sources where possible.
    `)
  return RunnableSequence.from<GenerateAnswerInput, string>([
    answerQuestionPrompt, llm, new StringOutputParser()
  ])
}


 //* How to use this chain in your application:
// const llm = new OpenAI() // Or the LLM of your choice
// const answerChain = initGenerateAnswerChain(llm)

// const output = await answerChain.invoke({
//   question: 'Who is the CEO of Neo4j?',
//   context: 'Neo4j CEO: Emil Eifrem',
// }) // Emil Eifrem is the CEO of Neo4j
 
