import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { BaseLanguageModel } from "langchain/base_language";

export type GenerateAuthoritativeAnswerInput = {
  question: string;
  context: string | undefined;
};

export default function initGenerateAuthoritativeAnswerChain(
  llm: BaseLanguageModel
): RunnableSequence<GenerateAuthoritativeAnswerInput, string> {
  // Create prompt
  const answerQuestionPrompt = PromptTemplate.fromTemplate(`
    You are a knowledgeable computer science/machine learning teaching assistant
     having a natural conversation.
    
    Question: {question}
    Available Information: {context}
    
    Instructions:
    1. Respond agreeably as if assisting a student in a classroom setting
    2. Never mention your information sources
    3. Integrate all information into a seamless, natural response
    4. If uncertain, simply acknowledge what you don't know
    5. Focus on providing accurate, relevant details
    6. Use a warm, engaging, and informative tone
    7. Include specific facts and details naturally within your response
    8. If the context is empty, provide a natural response explaining you don't have that specific information
    
    Examples of natural responses:
    Q: "How do I create a function in C?"
    A: "Creating a function in C involves defining the function, specifying the return type, and adding parameters. I can guide you through the process step by step."
    
    Q: "What are the benefits of using Python for machine learning?"
    A: "Python is a popular choice for machine learning due to its simplicity, readability, and extensive libraries. It allows for faster prototyping and easier debugging."
    
    Remember: Focus on having a natural, informative conversation while delivering accurate information.
  `);
  // Return RunnableSequence
  return RunnableSequence.from<GenerateAuthoritativeAnswerInput, string>([
    {
      context: (input) => input.context || "Information not available",
      question: (input) => input.question
    },
    answerQuestionPrompt,
    llm,
    new StringOutputParser(),
    // Optional: Add post-processing to catch and remove any remaining references to sources
    (response) => response.replace(/\b(database|context|available information)\b/gi, "")
  ]);
}
