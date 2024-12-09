import { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

/**
 * Create a new vector search index that uses the existing
 * `paperAbstracts` index.
 *
 * @param {EmbeddingsInterface} embeddings  The embeddings model
 * @returns {Promise<Neo4jVectorStore>}
 */
export default async function initVectorStore(
  embeddings: EmbeddingsInterface
): Promise<Neo4jVectorStore> {
  const vectorStore = await Neo4jVectorStore.fromExistingIndex(embeddings, { 
    url: process.env.NEO4J_URI as string,
    username: process.env.NEO4J_USERNAME as string,
    password: process.env.NEO4J_PASSWORD as string,
    indexName: "index_f7700477",
    textNodeProperty: "abstract",
    embeddingNodeProperty: "embedding",
    retrievalQuery: `
      RETURN
        node.abstract AS text,
        score,
        {
          _id: elementid(node),
          id: node.id,
          title: node.title,
          abstract: node.abstract,
          category: node.category,
          pdf_url: node.pdf_url,
        } AS metadata
    `,
   })
  return vectorStore;
}
