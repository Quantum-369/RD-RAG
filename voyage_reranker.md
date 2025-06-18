Voyage reranker is accessible in Python through the voyageai package. Please first install the voyageai package and setup the API key.

Voyage reranker receives as input a query and a list of candidate documents, e.g., the documents retrieved by a nearest neighbor search with embeddings. It reranks the candidate documents according to their semantic relevances to the search query, and returns the list of relevance scores. To access the reranker, create a voyageai.Client object and use its rerank() method.

voyageai.Client.rerank (query: str, documents: List[str], model: str, top_k: Optional[int] = None, truncation: bool = True)

Parameters

query (str) - The query as a string. The query can contain a maximum of 4,000 tokens for rerank-2, 2,000 tokens for rerank-2-lite and rerank-1, and 1,000 tokens for rerank-lite-1.
documents (List[str]) - The documents to be reranked as a list of strings.
The number of documents cannot exceed 1,000.
The sum of the number of tokens in the query and the number of tokens in any single document cannot exceed 16,000 for rerank-2; 8,000 for rerank-2-lite and rerank-1; and 4,000 for rerank-lite-1.
The total number of tokens, defined as "the number of query tokens × the number of documents + sum of the number of tokens in all documents", cannot exceed 600K for rerank-2 and rerank-2-lite, and 300K for rerank-1 and rerank-lite-1. Please see our FAQ.
model (str) - Name of the model. Recommended options: rerank-2, rerank-2-lite.
top_k (int, optional, defaults to None) - The number of most relevant documents to return. If not specified, the reranking results of all documents will be returned.
truncation (bool, optional, defaults to True) - Whether to truncate the input to satisfy the "context length limit" on the query and the documents.
If True, the query and documents will be truncated to fit within the context length limit, before processed by the reranker model.
If False, an error will be raised when the query exceeds 4,000 tokens for rerank-2; 2,000 tokens rerank-2-lite and rerank-1; and 1,000 tokens for rerank-lite-1, or the sum of the number of tokens in the query and the number of tokens in any single document exceeds 16,000 for rerank-2; 8,000 for rerank-2-lite and rerank-1; and 4,000 for rerank-lite-1.
Returns

A RerankingObject, containing the following attributes:
results (List[RerankingResult]) - A list of RerankingResult, with format specified below, sorted by the descending order of relevance scores. The length of the list equals to top_k if this argument is specified, otherwise the number of the input documents. Each element in the list is a RerankingResult object, which contains attributes:
index (int) - The index of the document in the input list.
document (str) - The document as a string.
relevance_score (float) - The relevance score of the document with respect to the query.
total_tokens (int) - The total number of tokens in the input, which is defined as "the number of query tokens × the number of documents + sum of the number of tokens in all documents".
Example

Python
Output

import voyageai

vo = voyageai.Client()
# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")

query = "When is Apple's conference call scheduled?"
documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
]

reranking = vo.rerank(query, documents, model="rerank-2", top_k=3)
for r in reranking.results:
    print(f"Document: {r.document}")
    print(f"Relevance Score: {r.relevance_score}")
    print()

Rerankers
post
https://api.voyageai.com/v1/rerank
Voyage reranker endpoint receives as input a query, a list of documents, and other arguments such as the model name, and returns a response containing the reranking results.

Body Params
query
string
required
The query as a string. The query can contain a maximum of 4,000 tokens for rerank-2, 2,000 tokens for rerank-2-lite and rerank-1, and 1,000 tokens for rerank-lite-1.

documents
array of strings
required
The documents to be reranked as a list of strings.

The number of documents cannot exceed 1,000.
The sum of the number of tokens in the query and the number of tokens in any single document cannot exceed 16,000 for rerank-2; 8,000 for rerank-2-lite and rerank-1; and 4,000 for rerank-lite-1.
The total number of tokens, defined as "the number of query tokens × the number of documents + sum of the number of tokens in all documents", cannot exceed 600K for rerank-2 and rerank-2-lite, and 300K for rerank-1 and rerank-lite-1. Please see our FAQ.
model
string
required
Name of the model. Recommended options: rerank-2, rerank-2-lite.

top_k
integer | null
Defaults to null
The number of most relevant documents to return. If not specified, the reranking results of all documents will be returned.

return_documents
boolean
Defaults to false
Whether to return the documents in the response. Defaults to false.

If false, the API will return a list of {"index", "relevance_score"} where "index" refers to the index of a document within the input list.
If true, the API will return a list of {"index", "document", "relevance_score"} where "document" is the corresponding document from the input list.
truncation
boolean
Defaults to true
Whether to truncate the input to satisfy the "context length limit" on the query and the documents. Defaults to true.

If true, the query and documents will be truncated to fit within the context length limit, before processed by the reranker model.
If false, an error will be raised when the query exceeds 4,000 tokens for rerank-2; 2,000 tokens rerank-2-lite and rerank-1; and 1,000 tokens for rerank-lite-1, or the sum of the number of tokens in the query and the number of tokens in any single document exceeds 16,000 for rerank-2; 8,000 for rerank-2-lite and rerank-1; and 4,000 for rerank-lite-1.