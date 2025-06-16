import os
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import gc


load_dotenv()
class PineconeVectorStore(BaseModel):
    index_name: str
    query: str

class QueryResult(BaseModel):
    text: str
    metadata: Dict
    score: float



def execute_query_acts(query: str):
    """
    Query the Pinecone vector database and return results with full metadata
    
    Args:
        query (str): The query text
    
    Returns:
        Dict: Returns a dictionary containing "texts" and "metadata_list" on success, or an "error" key on failure.

    """
    namespace = "casone-acts-1"
    pc = None
    index = None
    try:
        index_name = "casone-acts"

        # Initialize embeddings and Pinecone
        print(f"Getting index and project for namespace: {namespace}")
        print(f"Retrieved index_name: {index_name}")
        
        if not index_name:
            raise ValueError(f"No index or project found for namespace: {namespace}")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(index_name)
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=30,
            include_metadata=True,
            namespace=namespace,
        )
        
        # Extract results and metadata
        query_results = []
        for match in results["matches"]:
            text = match["metadata"].get("text", "")
            metadata = {
                "page": match["metadata"].get("page", "Unknown"),
                "score": match["score"],
                "document_name": match["metadata"].get("document_name", "Unknown"),
                # Add any other metadata fields you want to track
                "chunk_index": match["metadata"].get("chunk_index", "Unknown"),
            }
            query_results.append(QueryResult(text=text, metadata=metadata, score=match["score"]))
        
        # Return both texts and full metadata
        texts = [result.text for result in query_results]
        metadata_list = [result.metadata for result in query_results]
        return {"texts": texts, "metadata": metadata_list}
        
    except Exception as e:
        print(f"An error occurred in pinecone vector database query: {str(e)}")
        import traceback
        print("Full error details:")
        print(traceback.format_exc())
        return {"error": str(e)}
    
    finally:
        # Help garbage collection by clearing references and forcing collection
        embeddings = None
        query_embedding = None
        results = None
        query_results = None
        
        # No explicit cleanup needed for Pinecone clients
        pc = None
        index = None
        
        # Force garbage collection
        gc.collect()
        st.toast("acts query completed")



def execute_query_laws(query: str):
    """
    Query the Pinecone vector database and return results with full metadata
    
    Args:
        query (str): The query text
    
    Returns:
        Dict: Returns a dictionary containing "texts" and "metadata_list" on success, or an "error" key on failure.

    """
    namespace = "caseone-global-768"
    pc = None
    index = None

    try:
        index_name = "caseone-global-768"

        # Initialize embeddings and Pinecone
        print(f"Getting index and project for namespace: {namespace}")
        print(f"Retrieved index_name: {index_name}")
        
        if not index_name:
            raise ValueError(f"No index or project found for namespace: {namespace}")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY_SECOND"])
        index = pc.Index(index_name)
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=30,
            include_metadata=True,
            namespace=namespace,
        )
        
        # Extract results and metadata
        query_results = []
        for match in results["matches"]:
            text = match["metadata"].get("text", "")
            metadata = {
                "page": match["metadata"].get("page", "Unknown"),
                "score": match["score"],
                "document_name": match["metadata"].get("document_name", "Unknown"),
                # Add any other metadata fields you want to track
                "chunk_index": match["metadata"].get("chunk_index", "Unknown"),
            }
            query_results.append(QueryResult(text=text, metadata=metadata, score=match["score"]))
        
        # Return both texts and full metadata
        texts = [result.text for result in query_results]
        metadata_list = [result.metadata for result in query_results]
        return {"texts": texts, "metadata": metadata_list}
        
    except Exception as e:
        print(f"An error occurred in pinecone vector database query: {str(e)}")
        import traceback
        print("Full error details:")
        print(traceback.format_exc())
        return {"error": str(e)}
    
    finally:
        # Help garbage collection by clearing references and forcing collection
        embeddings = None
        query_embedding = None
        results = None
        query_results = None
        
        # No explicit cleanup needed for Pinecone clients
        pc = None
        index = None
        
        # Force garbage collection
        gc.collect()

        st.toast("laws query completed")

