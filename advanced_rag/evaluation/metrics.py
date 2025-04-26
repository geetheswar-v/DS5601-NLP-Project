"""Evaluation metrics for the RAG system."""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import nltk
from datasets import Dataset
# Import the main evaluate function from ragas
from ragas import evaluate as ragas_evaluate
# Import specific metrics if needed, but also Ragas LLM/Embeddings wrappers
from ragas.metrics import (
    answer_relevancy, context_precision, context_recall, faithfulness
)
# Import Langchain components for Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Import your config
from advanced_rag.config import RAGAS_LLM_MODEL, RAGAS_EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class RagEvaluator:
    """Evaluator for RAG systems."""
    
    def __init__(self):
        """Initialize the evaluator and download necessary resources."""
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Load traditional evaluation metrics
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        
        # Define the RAGAS metrics list
        self.ragas_metrics_list = [
            faithfulness, 
            answer_relevancy, 
            context_precision, # Requires 'ground_truth'
            context_recall
        ]
        
        # Initialize Ollama LLM and Embeddings for RAGAS
        # RAGAS uses Langchain interfaces for custom models
        logger.info(f"Initializing RAGAS with Ollama LLM: {RAGAS_LLM_MODEL} and Embeddings: {RAGAS_EMBEDDING_MODEL} at {OLLAMA_BASE_URL}")
        try:
            self.ragas_llm = ChatOllama(model=RAGAS_LLM_MODEL, base_url=OLLAMA_BASE_URL)
            # self.ragas_llm.invoke("Test prompt") 
            self.ragas_embeddings = OllamaEmbeddings(model=RAGAS_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            # self.ragas_embeddings.embed_query("Test query")
            logger.info("RAGAS LLM and Embeddings initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama models for RAGAS: {e}", exc_info=True)
            logger.error("Please ensure Ollama is running and the models are pulled.")
            # Here i can i add OpenAPI key for OpenAI models
            # but here we remove this part
            # For now, set to None and let evaluate_ragas_async handle it
            self.ragas_llm = None
            self.ragas_embeddings = None
    
    def evaluate_traditional(
        self, 
        predictions: List[str], 
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate using traditional NLG metrics.
        
        Args:
            predictions: List of generated answers
            references: List of reference (ground truth) answers
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        if not predictions or not references:
            logger.error("Empty predictions or references")
            return {"error": "Empty predictions or references"}
        
        if len(predictions) != len(references):
            logger.error(f"Predictions ({len(predictions)}) and references ({len(references)}) have different lengths")
            return {"error": "Mismatched lengths"}
        
        # BLEU
        bleu_results = self.bleu.compute(
            predictions=predictions, 
            references=references
        )
        results["bleu"] = bleu_results["bleu"]
        
        # ROUGE
        rouge_results = self.rouge.compute(
            predictions=predictions, 
            references=references
        )
        for key, value in rouge_results.items():
            results[key] = value
        
        # METEOR
        meteor_results = self.meteor.compute(
            predictions=predictions, 
            references=references
        )
        results["meteor"] = meteor_results["meteor"]
        
        return results
    
    async def evaluate_ragas_async(
        self, 
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str], 
    ) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics asynchronously.
        Note: ragas.evaluate might run synchronously depending on backend.
        """
        # Check if LLM/Embeddings were initialized
        if self.ragas_llm is None or self.ragas_embeddings is None:
             logger.error("RAGAS LLM or Embeddings not initialized. Skipping RAGAS evaluation.")
             return {"error": "RAGAS LLM/Embeddings not initialized"}

        # Ensure inputs are valid
        if not questions or not answers or not contexts or not ground_truths: 
            logger.error("Empty inputs for RAGAS evaluation")
            return {"error": "Empty inputs for RAGAS evaluation"}
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            logger.error(f"Mismatched lengths: questions ({len(questions)}), answers ({len(answers)}), contexts ({len(contexts)}), ground_truths ({len(ground_truths)})")
            return {"error": "Mismatched input lengths for RAGAS"}

        
        try:
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths, 
            }
            dataset = Dataset.from_dict(data)
            
            ragas_results = ragas_evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics_list,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            ) 
            
            # Calculate a combined score if needed (adjust based on actual keys)
            required_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            if all(key in ragas_results for key in required_keys):
                 ragas_results["combined_score"] = sum(ragas_results[key] for key in required_keys) / len(required_keys)
            else:
                 ragas_results["combined_score"] = None
                 missing = [k for k in required_keys if k not in ragas_results]
                 logger.warning(f"Could not calculate combined RAGAS score. Missing metrics: {missing}")

            return ragas_results

        except ValueError as ve: 
             logger.error(f"ValueError during RAGAS evaluation (likely missing column or LLM issue): {ve}", exc_info=True)
             return {"error": f"RAGAS ValueError: {str(ve)}"}
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}", exc_info=True)
            return {"error": f"RAGAS evaluation failed: {str(e)}"}
    
    async def evaluate_all(
        self,
        questions: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        contexts: List[List[str]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all evaluations and return combined results. (Now async)
        """
        traditional_metrics = self.evaluate_traditional(
            predictions=generated_answers,
            references=reference_answers,
        )
        
        ragas_metrics = {} 
        has_references = all(ref and isinstance(ref, str) and len(ref.strip()) > 0 for ref in reference_answers) 

        if has_references:
            if self.ragas_llm and self.ragas_embeddings:
                try:
                    ragas_metrics = await self.evaluate_ragas_async( 
                        questions=questions,
                        answers=generated_answers,
                        contexts=contexts,
                        ground_truths=reference_answers, 
                    )
                except Exception as e: 
                     logger.error(f"Error during RAGAS evaluation call: {e}", exc_info=True)
                     ragas_metrics = {"error": f"RAGAS evaluation failed: {str(e)}"}
            else:
                logger.warning("Skipping RAGAS evaluation because Ollama models failed to initialize.")
                ragas_metrics = {"error": "Skipped - RAGAS models not initialized"}
        else:
            logger.warning("Skipping RAGAS evaluation because reference_answers (ground_truth) are missing or invalid.")
            ragas_metrics = {"error": "Skipped - Missing ground_truth"}


        return {
            "traditional": traditional_metrics,
            "ragas": ragas_metrics,
        }
