"""Benchmark evaluation for the RAG system."""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from advanced_rag.config import DATA_DIR
from advanced_rag.evaluation.metrics import RagEvaluator
from advanced_rag.models.document import Document
from advanced_rag.rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)


class RagBenchmark:
    """Benchmark for evaluating RAG system."""
    
    def __init__(self, rag_pipeline: RagPipeline, evaluator: RagEvaluator):
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator
        self.benchmark_dir = DATA_DIR / "benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
    async def load_or_create_benchmark_data(
        self,
        dataset_path: Optional[Path] = None,
        num_questions: int = 3,
    ) -> pd.DataFrame:
        """
        Load existing benchmark data or create new data.
        
        Args:
            dataset_path: Path to existing dataset
            num_questions: Number of questions for new dataset
            
        Returns:
            DataFrame with benchmark data
        """
        if dataset_path and dataset_path.exists():
            logger.info(f"Loading benchmark data from {dataset_path}")
            try:
                df = pd.read_csv(dataset_path)
                
                # Validate minimum required columns
                if "question" not in df.columns:
                    logger.error(f"CSV file {dataset_path} is missing 'question' column")
                    raise ValueError(f"CSV file {dataset_path} is missing 'question' column")
                
                # Add reference_answer column if missing
                if "reference_answer" not in df.columns:
                    logger.warning(f"CSV file {dataset_path} is missing 'reference_answer' column. Adding empty column.")
                    df["reference_answer"] = ["" for _ in range(len(df))]
                
                return df
            except Exception as e:
                logger.error(f"Error loading CSV file {dataset_path}: {e}")
                raise
        
        # If no dataset provided, create a default path
        if not dataset_path:
            dataset_path = self.benchmark_dir / "engineering_benchmark.csv"
            if dataset_path.exists():
                logger.info(f"Loading benchmark data from {dataset_path}")
                return pd.read_csv(dataset_path)
        
        # Each question from different branch we thought of
        # for now here we didn't add answers, we manually add them later
        engineering_questions = [
            "What are the principles of reinforcement learning in AI?",
            "Explain the concept of impedance in electrical engineering",
            "What is the difference between supervised and unsupervised learning?",
            "What is the significance of the Navier-Stokes equations in fluid dynamics?",
            "Explain how blockchain technology works",
        ]
        
        # Take the requested number of questions
        questions = engineering_questions[:num_questions]
        
        # Create the benchmark DataFrame
        df = pd.DataFrame({
            "question": questions,
            "reference_answer": ["" for _ in questions],  # Empty for now
        })
        
        # Save the benchmark data
        df.to_csv(dataset_path, index=False)
        logger.info(f"Created benchmark data with {len(questions)} questions at {dataset_path}")
        
        return df
    
    async def run_benchmark(
        self,
        dataset_path: Optional[Path] = None,
        use_web: bool = True,
        num_questions: int = 20,
        output_path: Optional[Path] = None,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run benchmark evaluation.
        
        Args:
            dataset_path: Path to benchmark dataset
            use_web: Whether to use web retrieval
            num_questions: Number of questions for new dataset
            output_path: Path to save results
            
        Returns:
            Tuple of (evaluation results, detailed results DataFrame)
        """
        # Load benchmark data
        benchmark_df = await self.load_or_create_benchmark_data(dataset_path, num_questions)
        
        # Limit to the requested number of questions if creating a new dataset
        if dataset_path is None and len(benchmark_df) > num_questions:
            benchmark_df = benchmark_df.iloc[:num_questions]
            
        questions = benchmark_df["question"].tolist()
        
        # Set default output path if not provided
        if not output_path:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = self.benchmark_dir / f"results_{timestamp}.json"
        
        # Run queries
        results = []
        generated_answers = []
        contexts_list = []
        
        logger.info(f"Running benchmark with {len(questions)} questions")
        for i, question in enumerate(tqdm(questions, desc="Processing questions")):
            # Process the query (already async)
            start_time = time.time()
            answer, retrieved_docs = await self.rag_pipeline.process_query(
                query=question,
                generate_answer=True,
                use_web=use_web,
            )
            end_time = time.time()
            
            # Extract contexts
            contexts = [doc.content for doc in retrieved_docs]
            contexts_list.append(contexts)
            generated_answers.append(answer)
            
            # Store result
            result = {
                "question": question,
                "answer": answer,
                "processing_time": end_time - start_time,
                "num_docs_retrieved": len(retrieved_docs),
                "sources": [
                    {
                        "title": doc.metadata.title,
                        "url": doc.metadata.url,
                        "source_type": doc.metadata.source_type.value if hasattr(doc.metadata.source_type, 'value') else str(doc.metadata.source_type), 
                        "relevance_score": doc.metadata.relevance_score,
                    }
                    for doc in retrieved_docs
                ],
            }
            results.append(result)
            
            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Processed {i+1}/{len(questions)} questions")
        
        # Compute evaluation metrics if reference answers exist
        reference_answers = benchmark_df["reference_answer"].tolist()
        # Improved check for valid reference answers
        has_references = all(ref and isinstance(ref, str) and len(ref.strip()) > 5 for ref in reference_answers) 
        
        evaluation_results = {}
        if has_references:
            logger.info("Computing evaluation metrics")
            # Use await here
            evaluation_results = await self.evaluator.evaluate_all( 
                questions=questions,
                generated_answers=generated_answers,
                reference_answers=reference_answers,
                contexts=contexts_list,
            )
        else:
            logger.warning("No valid reference answers found for quantitative evaluation. Skipping RAGAS/traditional metrics.")
            evaluation_results = {"message": "No reference answers provided"}
        
        # Save detailed results to JSON
        try:
            with open(output_path, "w") as f:
                json.dump({
                    "results": results,
                    "evaluation": evaluation_results, # Now this should be a dict
                    "metadata": {
                        "num_questions": len(questions),
                        "use_web": use_web,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                }, f, indent=2)
            logger.info(f"Benchmark results saved to {output_path}")
            
            # Also save a CSV with questions and generated answers for easier review
            results_df = pd.DataFrame({
                "question": questions,
                "generated_answer": generated_answers,
                "reference_answer": reference_answers if has_references else [""] * len(questions)
            })
            csv_path = output_path.with_suffix('.csv')
            results_df.to_csv(csv_path, index=False)
            logger.info(f"CSV results saved to {csv_path}")
            
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}. Check data types in results or evaluation.", exc_info=True)
            raise
        
        results_df = pd.DataFrame(results)
        
        return evaluation_results, results_df
