"""
Mini RAG System for Movie Plots
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from utils import get_logger, sanitize_text

import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

logger = get_logger()
load_dotenv()


@dataclass
class ChunkMetadata:
    """Metadata for each plot chunk"""
    chunk_id: str
    title: str
    chunk_index: int
    chunked_plot: str


@dataclass
class RAGResponse:
    """Structured RAG output"""
    answer: str
    contexts: List[str]
    reasoning: str


@dataclass
class ModerationResult:
    """OpenAI moderation result"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    reason: Optional[str] = None


class ContentModerator:
    def __init__(self, client: OpenAI):
        self.client = client
        self.cache = {}  # Cache moderation results
    
    def moderate_text(self, text: str) -> ModerationResult:
        """
        Guardrail: Check content using OpenAI Moderation API
        """
        # Check cache first
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.cache:
            logger.debug("Using cached moderation result")
            return self.cache[text_hash]
        
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]

            flagged_categories = [
                cat for cat, flagged in result.categories.__dict__.items()
                if flagged
            ]

            mod_result = ModerationResult(
                flagged=result.flagged,
                categories=dict(result.categories.__dict__),
                category_scores=dict(result.category_scores.__dict__),
                reason=", ".join(flagged_categories) if flagged_categories else None
            )

            self.cache[text_hash] = mod_result  # Cache result
            
            if mod_result.flagged:
                logger.warning(f"Content flagged: {mod_result.reason}")
            
            return mod_result
            
        except Exception as e:
            logger.error(f"Moderation API error: {e}")
            # Allow content if API fails
            return ModerationResult(
                flagged=False,
                categories={},
                category_scores={},
                reason="Moderation API unavailable"
            )
    
    def is_safe(self, text: str, strict: bool = False) -> Tuple[bool, Optional[str]]:
        result = self.moderate_text(text)
        
        if not result.flagged:
            return True, None
        
        if strict:
            return False, f"Content flagged: {result.reason}"

        # Check severity scores
        high_severity = any(
            score > 0.8 for cat, score in result.category_scores.items()
            if cat in ['hate', 'hate/threatening', 'self-harm', 'violence']
        )

        if high_severity:
            return False, f"High-severity content detected: {result.reason}"
        
        return True, None


class EmbeddingCache:
    """Cache embeddings to avoid redundant API calls"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get(self, text: str) -> Optional[List[float]]:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self.cache.get(text_hash)
    
    def set(self, text: str, embedding: List[float]):
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.cache[text_hash] = embedding
        self._save_cache()


class RAGSystem:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
        enable_moderation: bool = True
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k
        self.enable_moderation = enable_moderation
        
        # Initialize components
        self.moderator = ContentModerator(self.client) if enable_moderation else None
        self.embedding_cache = EmbeddingCache()
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[ChunkMetadata] = []
        
        logger.info(f"RAG System initialized")
        logger.info(f"  Embedding: {embedding_model}")
        logger.info(f"  LLM: {llm_model}")
        logger.info(f"  Moderation: {'Enabled' if enable_moderation else 'Disabled'}")
    
    def load_preprocessed_data(self, csv_path: str):
        logger.info(f"Loading preprocessed chunks from {csv_path}")

        try:
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')

            required_cols = ['chunk_id', 'title', 'chunked_plot', 'chunk_index']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must be a preprocessed chunks CSV with columns: {required_cols}")

            chunks = []
            for _, row in df.iterrows():
                try:
                    chunk = ChunkMetadata(
                        chunk_id=str(row['chunk_id']),
                        title=str(row['title']),
                        chunk_index=int(row['chunk_index']),
                        chunked_plot=str(row['chunked_plot']),
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Skipping malformed chunk row: {e}")

            self.chunks = chunks
            logger.info(f"Loaded {len(self.chunks)} chunks from {csv_path}")
            return len(self.chunks)

        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
    def create_embeddings(self, batch_size: int = 100):
        logger.info("Generating embeddings...")
        
        embeddings = []
        texts_to_embed = []
        cached_count = 0
        
        # Check cache first
        for chunk in self.chunks:
            cached_embedding = self.embedding_cache.get(chunk.chunked_plot)
            if cached_embedding:
                embeddings.append(cached_embedding)
                cached_count += 1
            else:
                texts_to_embed.append(chunk.chunked_plot)
                embeddings.append(None)  # Placeholder
        
        logger.info(f"Using {cached_count} cached embeddings")
        
        # Generate new embeddings in batches
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_model
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    
                    # Cache new embeddings
                    for text, embedding in zip(batch, batch_embeddings):
                        self.embedding_cache.set(text, embedding)
                    
                    # Fill in placeholders
                    embed_idx = 0
                    for j, emb in enumerate(embeddings):
                        if emb is None and embed_idx < len(batch_embeddings):
                            embeddings[j] = batch_embeddings[embed_idx]
                            embed_idx += 1
                    
                    logger.info(f"Embedded batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    raise
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array
    
    def build_index(self, embeddings: np.ndarray):
        logger.info("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # Create index (L2 distance)
        
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[ChunkMetadata]:
        # Guardrail: Moderate user query
        if self.enable_moderation and self.moderator:
            is_safe, reason = self.moderator.is_safe(query, strict=True)
            if not is_safe:
                raise ValueError(f"Query rejected: {reason}")
        
        k = k or self.top_k
        
        # Generate query embedding
        try:
            query_embedding = self.client.embeddings.create(
                input=[query],
                model=self.embedding_model
            ).data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
        
        # Search
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        # Guardrail: Validate results
        retrieved_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Guardrail: Filter by relevance threshold
                if distance < 1.5:  # Reasonable threshold for L2 distance
                    retrieved_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks
    
    def generate_answer(self, query: str, contexts: List[ChunkMetadata]) -> RAGResponse:
        if not contexts:
            return RAGResponse(
                answer="I couldn't find relevant information in the movie database to answer your question.",
                contexts=[],
                reasoning="No relevant movie plots were found for this query."
            )
        
        # Prepare context
        context_text = "\n\n".join([
            f"[{i+1}] Movie: {chunk.title}\n{chunk.chunked_plot}"
            for i, chunk in enumerate(contexts)
        ])
        
        # Construct prompt with guardrails
        system_prompt = """
            You are a helpful movie expert assistant. Answer questions based ONLY on the provided movie plot contexts.

            Guidelines:
            1. Only use information from the provided contexts
            2. If the answer isn't in the contexts, say so clearly
            3. Be concise and accurate
            4. Cite which movie(s) you're referencing
            5. Provide your reasoning for the answer
            6. Do not include any inappropriate, offensive, or harmful content

            Respond in valid JSON format with these fields:
            {
            "answer": "Your answer here as a single paragraph",
            "reasoning": "Explain what question asks in your words and explain how you formed the answer as a single paragraph"
            }
        """
        
        user_prompt = f"""
            Question: {query}

            Movie Plot Contexts:
            {context_text}

            Provide your response in JSON format.
        """
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Guardrail: Validate response structure
            answer = result.get("answer", "Unable to generate answer")
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Guardrail: Moderate the generated answer
            if self.enable_moderation and self.moderator:
                is_safe, reason = self.moderator.is_safe(answer, strict=False)
                if not is_safe:
                    logger.warning(f"Generated answer flagged: {reason}")
                    answer = "I cannot provide an appropriate answer to this question based on the available content."

            return RAGResponse(
                answer=answer,
                contexts=[f"{chunk.title} - ... {chunk.chunked_plot} ..." for chunk in contexts],
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def query(self, query: str) -> Dict:
        logger.info(f"Processing query: {query}")
        
        try:
            contexts = self.retrieve(query)  # Retrieve relevant chunks
            response = self.generate_answer(query, contexts)
            return asdict(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "An error occurred processing your query.",
                "contexts": [],
                "reasoning": str(e)
            }


def main():
    rag = RAGSystem(enable_moderation=True)

    preproc_path = Path("preprocessed_wiki_movies.csv")
    if preproc_path.exists():
        try:
            num_chunks = rag.load_preprocessed_data(str(preproc_path))
            if num_chunks > 0:
                embeddings = rag.create_embeddings()
                rag.build_index(embeddings)
                logger.info(f"Loaded {num_chunks} preprocessed chunks and built FAISS index.")
        except Exception as e:
            logger.error(f"Failed to build vector embedding from {preproc_path}: {e}")

    # Interactive mode
    print("Mini RAG for Movie Plots — interactive mode. Type 'End' or 'Quite' to exit.")
    for _ in range(10**6):
        try:
            query = input("\nEnter your question: ")
            query = sanitize_text(query)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("end", "quite"):
            print("Goodbye.")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        # Run the RAG pipeline and get results
        result = rag.query(query)
        
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
