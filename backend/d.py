import os
import json
import importlib
import pandas as pd
from typing import List, Dict, Any

# Non-LLM Imports
import evaluate as hf_eval
from nubia_score import Nubia

# RAGAS Imports
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def evaluate_llm_metrics(
    data: List[Dict[str, Any]],
    metrics_to_evaluate: List[str],
    openai_api_key: str,
    openai_model: str,
    output_file: str = "ragas_scores.json"
) -> Dict[str, Any]:
    """
    Evaluate LLM-based metrics using RAGAS framework.
    
    Args:
        data: List of dictionaries containing evaluation data
        metrics_to_evaluate: List of RAGAS metric names to evaluate
        openai_api_key: OpenAI API key
        openai_model: OpenAI model name
        output_file: Output JSON file path
    
    Returns:
        Dictionary containing evaluation results
    """
    # Setup OpenAI environment
    os.environ["OPENAI_API_KEY"] = openai_api_key
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=openai_model))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # Create RAGAS dataset
    dataset = EvaluationDataset.from_list(data)

    # Evaluate each metric
    evaluation_results = []
    for metric in metrics_to_evaluate:
        def _ragas_evaluate(dataset, metric_name):
            metric_class = getattr(importlib.import_module('ragas.metrics'), metric_name)
            return evaluate(dataset=dataset, metrics=[metric_class()], llm=evaluator_llm)
        
        result = _ragas_evaluate(dataset, metric)
        evaluation_results.append(result)

    # Process results
    df = pd.DataFrame([res.to_pandas() for res in evaluation_results])
    scores = pd.concat([df['score'], df['scores']], axis=1)

    # Create final JSON structure
    output_json = {"scores": []}
    for _, row in scores.iterrows():
        output_json["scores"].extend(row["scores"])
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"LLM metrics saved to {output_file}")
    return output_json

def evaluate_non_llm_metrics(
    data: List[Dict[str, Any]],
    metrics_to_evaluate: List[str],
    output_file: str = "quant_scores.json"
) -> List[Dict[str, Any]]:
    """
    Evaluate non-LLM based metrics using HuggingFace Evaluate and Nubia.
    
    Args:
        data: List of dictionaries containing evaluation data
        metrics_to_evaluate: List of metric names to evaluate
        output_file: Output JSON file path
    
    Returns:
        List of combined evaluation results
    """
    metrics_results = []
    
    for metric in metrics_to_evaluate:
        metric_scores = []
        
        if metric == 'nubia':
            n = Nubia()
            for entry in data:
                score = n.score(entry['reference'], entry['response'], verbose=True, get_features=True)
                metric_scores.append(score)
        else:
            eval_metric = hf_eval.load(metric)
            for entry in data:
                score = eval_metric.compute(
                    references=[entry['reference']], 
                    predictions=[entry['response']]
                )
                metric_scores.append(score)
        
        metrics_results.append(metric_scores)

    # Combine results across metrics
    def _combine_results(results: List[List[Dict]]]) -> List[Dict]:
        combined = []
        for i in range(len(results[0])):
            entry = {}
            for metric in results:
                entry.update(metric[i])
            combined.append(entry)
        return combined
    
    combined_results = _combine_results(metrics_results)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"Non-LLM metrics saved to {output_file}")
    return combined_results


if __name__ == "__main__":
    # Load eval_config.json
    with open("eval_config.json", mode="r", encoding="utf-8") as config_file:
        eval_config = json.load(config_file)

    # Extract LLM and non-LLM metrics based on 'true' values
    llm_metrics_to_evaluate = [
        metric for metric, enabled in eval_config["LLM_METRICS"].items() 
        if enabled is True and metric not in ["OpenAI_API", "OpenAI_Model"]
    ]
    non_llm_metrics_to_evaluate = [
        metric for metric, enabled in eval_config["METRICS"].items() 
        if enabled is True
    ]

    # Load OpenAI API key and model from eval_config
    openai_api_key = eval_config["LLM_METRICS"]["OpenAI_API"]
    openai_model = eval_config["LLM_METRICS"]["OpenAI_Model"]

    # Load data from DATA_FILE specified in eval_config
    data_file = eval_config["DATA_FILE"]
    with open(data_file, mode="r", encoding="utf-8") as data_file:
        data = json.load(data_file)

    # Evaluate LLM metrics if any are enabled
    if llm_metrics_to_evaluate:
        print("Evaluating LLM metrics:", llm_metrics_to_evaluate)
        llm_results = evaluate_llm_metrics(
            data=data,
            metrics_to_evaluate=llm_metrics_to_evaluate,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            output_file="llm_metrics_results.json"
        )
        print("LLM Metrics Evaluation Complete. Results saved to llm_metrics_results.json")

    # Evaluate non-LLM metrics if any are enabled
    if non_llm_metrics_to_evaluate:
        print("Evaluating non-LLM metrics:", non_llm_metrics_to_evaluate)
        non_llm_results = evaluate_non_llm_metrics(
            data=data,
            metrics_to_evaluate=non_llm_metrics_to_evaluate,
            output_file="non_llm_metrics_results.json"
        )
        print("Non-LLM Metrics Evaluation Complete. Results saved to non_llm_metrics_results.json")