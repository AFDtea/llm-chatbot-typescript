import { BaseLanguageModel } from "langchain/base_language";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";

export default async function initCypherGenerationChain(
  graph: Neo4jGraph,
  llm: BaseLanguageModel
) {
  //Create Prompt Template
  const cypherPrompt = PromptTemplate.fromTemplate(`
     You are a Neo4j Developer translating user questions into Cypher to answer questions
  about movies and provide recommendations.
  Convert the user's question into a Cypher statement based on the schema.

  You must:
  * Only use the nodes, relationships and properties mentioned in the schema.
  * When required, \`IS NOT NULL\` to check for property existence.
  * Use the \`elementId()\` function to return the unique identifier for a node or relationship as \`_id\`.
  * Include extra information about the papers that may help provide a more informative answer,
    such as publication date, citations, and abstracts.
  * Limit the maximum number of results to 10.
  * Respond with only a Cypher statement. No preamble.

  Example Question: Who authored the paper about transformers?
  Example Cypher:
  MATCH (p:Paper)<-[:AUTHORED]-(a:Person)
  WHERE toLower(p.title) CONTAINS 'transformer'
  RETURN p.title AS Paper, collect(a.name) AS Authors, p.abstract AS Abstract,
  elementId(p) AS _id
  LIMIT 10

  Schema:
  {schema}

  Question:
  {question}
    `);

    //Create Runnable Sequence
    return RunnableSequence.from<string, string>([
      {
        // Take the input and assign it to the question key
        question: new RunnablePassthrough(),
        // Get the schema
        schema: () => graph.getSchema(),
      },
      cypherPrompt,
      llm,
      new StringOutputParser(),
    ]);
}
