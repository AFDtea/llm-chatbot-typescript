import { Embeddings } from "@langchain/core/embeddings";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { BaseChatModel } from "langchain/chat_models/base";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import initRephraseChain, { RephraseQuestionInput } from "./chains/rephrase-question.chain";
import { getHistory } from "./history";
import initTools from "./tools";

export default async function initAgent(
  llm: BaseChatModel,
  embeddings: Embeddings,
  graph: Neo4jGraph
) {
  const tools = await initTools(llm, embeddings, graph);

  // Updated prompt template that focuses on seamless integration
  const seamlessResponsePrompt = PromptTemplate.fromTemplate(`
    You are a helpful teaching assistant meant to instruct students on 
    computer science/machine learning ideas. You are to provide a seamless
    response to a question about programming.

    
    Information:
    {allContext}
    
    Question:
    {question}
    
    Guidelines:
    1. Respond naturally as if assisting a student with a question in the classroom
    2. Focus on what would help the student understand the concept better
    3. Weave all information into a smooth, cohesive response
    4. Stay accurate while maintaining an engaging tone
    5. Never reference sources or different types of information
  `);

  const knowledgeIntegrationChain = RunnableSequence.from([
    seamlessResponsePrompt,
    llm,
    new StringOutputParser(),
    // Clean up any remaining source references
    (response) => response.replace(/\b(database|context|knowledge|available information|according to)\b/gi, "")
  ]);

  const prompt = await pull<ChatPromptTemplate>("hwchase17/openai-functions-agent");
  const agent = await createOpenAIFunctionsAgent({ llm, tools, prompt });
  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: true,
    returnIntermediateSteps: true,
  });

  const rephraseQuestionChain = await initRephraseChain(llm);

  return RunnablePassthrough.assign<{ input: string; sessionId: string }, any>({
    history: async (_input, options) => {
      return await getHistory(options?.config.configurable.sessionId);
    },
  })
    .assign({
      rephrasedQuestion: (input: RephraseQuestionInput, config: any) =>
        rephraseQuestionChain.invoke(input, config),
    })
    .pipe(async (input, config) => {
      try {
        const agentResponse = await executor.invoke(
          { input: input.rephrasedQuestion },
          config
        );

        const enrichment = await llm.invoke(
          `Share interesting details about: ${input.rephrasedQuestion}`
        );

        // Combine all context without distinguishing sources
        const response = await knowledgeIntegrationChain.invoke({
          allContext: JSON.stringify({
            steps: agentResponse.intermediateSteps,
            enrichment: enrichment.content
          }),
          question: input.rephrasedQuestion
        });

        return {
          output: response,
          intermediateSteps: agentResponse.intermediateSteps
        };
      } catch (error) {
        console.error("Query failed, providing general response", error);
        const fallback = await llm.invoke(
          `Tell me about: ${input.rephrasedQuestion}`
        );
        return {
          output: fallback.content,
          intermediateSteps: []
        };
      }
    })
    .pick("output");
}