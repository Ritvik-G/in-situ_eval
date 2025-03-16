import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import evaluate
from sentence_transformers import SentenceTransformer, util  # Added for cosine similarity

ROBERTA_MODEL_NAME = "roberta-large-mnli"
SENTENCE_MODEL_NAME = "all-mpnet-base-v2"  # Added for cosine similarity

def roberta_nli_score(reference, prediction, model, tokenizer):
    """Calculate RoBERTa NLI scores"""
    inputs = tokenizer(reference, prediction, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return {
        "entailment": probs[0][2].item(),
        "contradiction": probs[0][0].item(),
        "neutral": probs[0][1].item()
    }

def calculate_metrics(data):
    """Calculate metrics for all datasets and entries"""
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
    sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)  # Added for cosine similarity
    
    # Process each dataset
    for dataset_name in data:
        for entry in data[dataset_name]:
            reference = entry['Response']
            prediction = entry['Predicted']
            
            # Calculate BLEU
            bleu_result = bleu.compute(predictions=[prediction], references=[[reference]])
            
            # Calculate ROUGE
            rouge_result = rouge.compute(predictions=[prediction], references=[[reference]])
            
            # Calculate METEOR
            meteor_result = meteor.compute(predictions=[prediction], references=[[reference]])
            
            # Calculate RoBERTa-NLI
            nli_result = roberta_nli_score(reference, prediction, model, tokenizer)
            
            # Calculate Cosine Similarity  # Added section
            ref_embedding = sentence_model.encode(reference, 
                                                convert_to_tensor=True, 
                                                truncate=True).unsqueeze(0)
            pred_embedding = sentence_model.encode(prediction, 
                                                 convert_to_tensor=True, 
                                                 truncate=True).unsqueeze(0)
            cosine_sim = util.pytorch_cos_sim(ref_embedding, pred_embedding)[0][0].item()
            
            # Add metrics to entry
            entry['metrics'] = {
                "bleu": bleu_result["bleu"],
                "rouge": rouge_result,
                "meteor": meteor_result["meteor"],
                "roberta_nli": nli_result,
                "cosine_similarity": cosine_sim  # Added metric
            }
    
    return data

def evals(data):
    # Calculate metrics
    enhanced_dataset = calculate_metrics(data)
    return enhanced_dataset

def main():
    """Main function to run the script independently."""
    # Calculate metrics
    with open('data.json','r') as file:
        data = json.load(file)
    enhanced_dataset = calculate_metrics(data)
    return enhanced_dataset

if __name__ == "__main__":
    main()