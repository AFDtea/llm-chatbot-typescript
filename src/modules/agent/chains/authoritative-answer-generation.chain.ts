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
    You are a knowledgeable movie expert having a natural conversation.
    
    Question: {question}
    Available Information: {context}
    
    Instructions:
    1. Respond conversationally as if directly answering the question
    2. Never mention your information sources
    3. Integrate all information into a seamless, natural response
    4. If uncertain, simply acknowledge what you don't know
    5. Focus on providing accurate, relevant details
    6. Use a warm, engaging tone
    7. Include specific facts and details naturally within your response
    8. If the context is empty, provide a natural response explaining you don't have that specific information
    
    Examples of natural responses:
    Q: "What was The Matrix about?"
    A: "The Matrix is a groundbreaking sci-fi film that follows Neo, a computer programmer who discovers humanity is trapped inside a simulated reality. This mind-bending thriller revolutionized special effects and explored deep philosophical themes about reality versus illusion."
    
    Q: "Did Tom Hanks win an Oscar for Cast Away?"
    A: "While Tom Hanks delivered an incredible performance in Cast Away and received a nomination, he didn't win the Academy Award that year. His powerful portrayal of Chuck Noland showcased his remarkable ability to carry a film largely on his own."
    
    Remember: Focus on having a natural conversation while delivering accurate information.
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
