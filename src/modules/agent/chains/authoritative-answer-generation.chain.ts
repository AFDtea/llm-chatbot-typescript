import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { BaseLanguageModel } from "langchain/base_language";

// tag::interface[]
export type GenerateAuthoritativeAnswerInput = {
  question: string;
  context: string | undefined;
};
// end::interface[]

// tag::function[]
export default function initGenerateAuthoritativeAnswerChain(
  llm: BaseLanguageModel
): RunnableSequence<GenerateAuthoritativeAnswerInput, string> {
  // Create prompt
  const answerQuestionPrompt = PromptTemplate.fromTemplate(`
    Use the following context to answer the following question.
    The context is provided by an authoritative source, you must never doubt
    it. You must use the context to answer the question but may also use previous training knowledge
    to provide a more accurate answer.
  
    Make the answer sound like it is a response to the question.
    Do not mention that you have based your response on the context.
  
    Here is an example:
  
    Question: Who played Woody in Toy Story?
    Context: ['role': 'Woody', 'actor': 'Tom Hanks']
    Response: Tom Hanks played Woody in Toy Story.
  
    If no context is provided, say that you don't know,
    don't try to make up an answer.
    If no context is provided you may also ask for clarification.
  
    Include links and sources where possible.
  
    Question:
    {question}
  
    Context:
    {context}
  `);
  // Return RunnableSequence
  return RunnableSequence.from<GenerateAuthoritativeAnswerInput, string>([
    RunnablePassthrough.assign({
      context: ({ context }) =>
        context == undefined || context === "" ? "I don't know" : context,
    }),
    answerQuestionPrompt,
    llm,
    new StringOutputParser(),
  ]);
}
