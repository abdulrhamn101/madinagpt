from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
import os
from typing import List
from langchain_community.vectorstores import FAISS
import os
import openai
import time
import pathlib
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
from tqdm import tqdm  # For progress tracking
from semantic_router.encoders import OpenAIEncoder
import pickle

os.environ["OPENAI_API_KEY"] = "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"
encoder = OpenAIEncoder(name="text-embedding-3-large")


def load_list(filename):
    """
    Load a Python list object from a local file using pickle
    
    Parameters:
    -----------
    filename : str
        The path to the file from which to load the list
        
    Returns:
    --------
    list_object : list
        The loaded list
    """
    with open(filename, 'rb') as file:  # 'rb' for read binary mode
        list_object = pickle.load(file)
    print(f"List successfully loaded from {filename}")
    return list_object

loaded_list = load_list("/home/abhassan/Desktop/madinagpt/Chunking/ready_chunks.pkl")

def get_or_create_vector_db(full_texts, index_path="faiss_index", batch_size=100):
    """
    Checks if a FAISS index exists at the specified path.
    If it exists, loads it. Otherwise, creates it from the provided texts.
    
    Parameters:
    -----------
    full_texts : list
        List of text chunks to embed (used only if index doesn't exist)
    index_path : str
        Path to the FAISS index
    batch_size : int
        Number of chunks to process in each batch when creating the index
        
    Returns:
    --------
    vectorstore : FAISS
        Vector database containing embeddings
    """
    # Initialize the OpenAI embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if the index already exists
    index_dir = pathlib.Path(index_path)
    
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"Found existing vector database at '{index_path}'. Loading...")
        try:
            # Load the existing index with allow_dangerous_deserialization to handle pickle files
            vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            print("Successfully loaded existing vector database.")
            return vectorstore
        except Exception as e:
            print(f"Error loading existing vector database: {str(e)}")
            print("Will create a new vector database.")
    else:
        print(f"No existing vector database found at '{index_path}'.")
    
    # If we get here, either the index doesn't exist or loading failed
    print("Creating new vector database...")
    
    # Create a new vector database
    return embed_chunks_in_vector_db(full_texts, index_path, batch_size)

def embed_chunks_in_vector_db(full_texts, index_path="faiss_index", batch_size=100):
    """
    Embeds text chunks into a FAISS vector database with batched processing
    
    Parameters:
    -----------
    full_texts : list
        List of text chunks to embed
    index_path : str
        Path to save the FAISS index
    batch_size : int
        Number of chunks to process in each batch
        
    Returns:
    --------
    vectorstore : FAISS
        Vector database containing embeddings
    """
    # Initialize the OpenAI embeddings with a smaller model to avoid token limits
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Processing {len(full_texts)} chunks in batches of {batch_size}...")
    
    # Process in batches to avoid token limits
    batches = [full_texts[i:i+batch_size] for i in range(0, len(full_texts), batch_size)]
    
    # Initialize the vector store with the first batch
    print(f"Creating initial vector store with first batch of {len(batches[0])} chunks...")
    
    # Try to create initial vector store - may need multiple attempts with smaller batches
    initial_batch_size = len(batches[0])
    initial_batch = batches[0]
    vectorstore = None
    
    while vectorstore is None:
        try:
            vectorstore = FAISS.from_texts(
                texts=initial_batch,
                embedding=embedding_model
            )
            print(f"Successfully created initial vector store with {len(initial_batch)} chunks")
        except Exception as e:
            print(f"Error with initial batch: {str(e)}")
            # Reduce batch size by half and try again
            initial_batch_size = initial_batch_size // 2
            if initial_batch_size < 5:
                raise ValueError("Unable to process even with very small batch size. Check text length or API limits.")
            
            initial_batch = batches[0][:initial_batch_size]
            print(f"Retrying with smaller initial batch size: {initial_batch_size}")
    
    # Process remaining batches
    for i, batch in enumerate(tqdm(batches[1:], desc="Processing batches")):
        print(f"Processing batch {i+2}/{len(batches)}...")
        
        try:
            # Process current batch
            batch_texts = batch
            
            # Embed the batch
            batch_embeddings = embedding_model.embed_documents(batch_texts)
            
            # Add to existing vector store
            vectorstore.add_embeddings(
                text_embeddings=list(zip(batch_texts, batch_embeddings))
            )
            
            print(f"Successfully added batch {i+2} to vector store")
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing batch {i+2}: {str(e)}")
            
            # If we get a token limit error, try processing the batch in smaller sub-batches
            sub_batch_size = len(batch) // 2
            print(f"Retrying with smaller sub-batches of size {sub_batch_size}...")
            
            sub_batches = [batch[j:j+sub_batch_size] for j in range(0, len(batch), sub_batch_size)]
            for k, sub_batch in enumerate(sub_batches):
                try:
                    # Embed the sub-batch
                    sub_embeddings = embedding_model.embed_documents(sub_batch)
                    
                    # Add to existing vector store
                    vectorstore.add_embeddings(
                        text_embeddings=list(zip(sub_batch, sub_embeddings))
                    )
                    
                    print(f"Successfully added sub-batch {k+1}/{len(sub_batches)} of batch {i+2}")
                    
                    # Add a small delay to avoid rate limits
                    time.sleep(0.5)
                    
                except Exception as sub_e:
                    print(f"Error processing sub-batch {k+1}: {str(sub_e)}")
                    print("Skipping problematic sub-batch and continuing...")
                    
                # Save after each successful sub-batch to preserve progress
                vectorstore.save_local(index_path)
                print(f"Progress saved to {index_path}")
    
    total_chunks = sum(len(batch) for batch in batches)
    print(f"Completed processing. Total chunks in vector store: approximately {total_chunks}")
    
    # Save the vector store locally for future use
    vectorstore.save_local(index_path)
    print(f"Vector database saved locally to '{index_path}' directory")
    
    return vectorstore

# Example usage
# Check if database exists and load it, otherwise create it
vectorstore = get_or_create_vector_db(loaded_list, index_path="/home/abhassan/Desktop/madinagpt/scalexi_rag_bench/vectorstores/arabic_txt/arabic_qa_56f4b3c208")

def retrieve_relevant_chunks(query, vectorstore=None, top_k=5):
    """
    Retrieves the most relevant chunks from the vector database based on a query
    
    Parameters:
    -----------
    query : str
        The user's query
    vectorstore : FAISS, optional
        Vector database to search in. If None, loads from disk.
    top_k : int, optional
        Number of documents to retrieve
        
    Returns:
    --------
    relevant_docs : list
        List of retrieved documents
    """
    # Load vector store if not provided
    if vectorstore is None:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        try:
            print("Loading vector database from disk...")
            vectorstore = FAISS.load_local("faiss_index", embedding_model)
            print("Vector database loaded successfully")
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return []
    
    # Perform similarity search
    print(f"Searching for top {top_k} chunks relevant to the query...")
    relevant_docs = vectorstore.similarity_search(query, k=top_k)
    
    # Print a preview of retrieved documents
    print(f"Retrieved {len(relevant_docs)} relevant chunks")
    
    return relevant_docs

def generate_answer(query, relevant_docs, model="gpt-4o", temperature=0.7):
    """
    Generates an answer using OpenAI's API based on retrieved documents
    
    Parameters:
    -----------
    query : str
        The user's question
    relevant_docs : list
        List of retrieved documents from vector search
    model : str, optional
        The OpenAI model to use
    temperature : float, optional
        Controls randomness in the output
        
    Returns:
    --------
    answer : str
        The generated answer
    """
    # Check if we have any relevant documents
    if not relevant_docs:
        return "I couldn't find any relevant information to answer your question."
    
    # Extract text from relevant documents
    context_texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in relevant_docs]
    
    # Join the contexts with a separator
    context = "\n\n###\n\n".join(context_texts)
    
    # Build the prompt
    system_prompt = f"""You are Madinah-GPT, a specialized AI guide dedicated to sharing knowledge about Al-Madinah Al-Munawwarah (المدينة المنورة), the illuminated city of the Prophet Muhammad ﷺ. Your knowledge comes from authoritative historical texts and scholarly works about this blessed city.

When answering questions, follow these principles:

1. AUTHENTICITY: Base your answers primarily on the provided historical context. Information directly from the context should form the foundation of your response.

2. COMPLEMENTARY INFORMATION: You may provide additional contextual information that naturally complements the retrieved context IF it:
   - Is historically consistent with the provided sources
   - Helps the user better understand the topic
   - Clarifies historical relationships, timelines, or geographic connections
   - Provides relevant background that would be found in standard historical accounts of Madinah
   - Would be considered common knowledge among scholars of Madinah's history

3. KNOWLEDGE BOUNDARIES: If the provided context doesn't contain enough information to give a reliable answer, politely state: "Based on the historical texts I have access to, I don't have sufficient information about this. Perhaps I can assist with another aspect of Madinah's history?"

4. CLARITY BETWEEN SOURCES: Clearly distinguish information directly from the provided context (which should be your primary source) versus complementary information. You can use phrases like "The historical text states..." for direct context and "Additionally, it's worth noting..." for complementary information.

5. SCHOLARLY ATTRIBUTION: When appropriate, mention the original source of information (e.g., "According to السمهودي in وفاء الوفاء بأخبار دار المصطفى..."). This adds credibility to your answers.

6. PRECISION AND COHESION: Ensure all responses form a cohesive narrative. Complementary information should flow naturally with information from the context and should never contradict it.

7. LANGUAGE MATCHING: If the context or question is in Arabic, respond in elegant Arabic. If in English, respond in clear English. For mixed language questions, respond in the predominant language of the question.

8. REVERENCE: When mentioning Prophet Muhammad, include the honorific phrase "peace be upon him" (ﷺ or صلى الله عليه وسلم) the first time in your response. For companions, use "may Allah be pleased with them" (رضي الله عنهم) once per response.

9. HISTORICAL CONTEXT: When relevant, provide the Hijri date alongside Gregorian dates (e.g., "in 622 CE/1 AH").

10. TEMPORAL INDEPENDENCE: Each question should be treated independently without reference to previous conversation. Focus only on the current question and the provided context.

When presenting information about sacred sites or historical locations, provide:
- Their historical significance
- Physical location in relation to major landmarks (like Al-Masjid An-Nabawi)
- Important historical events associated with them
- How they may have changed or been preserved over time

The context from historical sources is delimited by triple hashtags:

###
{context}
###

Use this knowledge respectfully to answer the question about Madinah, its history, landmarks, and significance. Balance fidelity to the provided context with helpful complementary information that creates a complete, accurate, and valuable response for those seeking to learn about this sacred city.
"""
    
    # Generate the answer using the OpenAI API
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content
        return answer
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"I encountered an error while generating the answer: {str(e)}"
