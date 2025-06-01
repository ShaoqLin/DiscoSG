import json
import re
from tqdm import tqdm
import argparse

def clean_text_caparena(text: str) -> str:
    """Clean and normalize text."""
    # strip
    text = text.strip()
    text = re.sub(r'"+', '"', text)
    text = text.replace('"', "'")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\t+", "", text)
    # text = re.sub("*", "", text)
    text = text.replace("*", "")
    assert "*" not in text, f"Error: {text}"
    assert "\n" not in text, f"Error: {text}"
    assert "\t" not in text, f"Error: {text}"
    return text

def find_matching_metrics(original_data, metrics_data):
    """
    Match original data with metrics data based on captions.
    Returns a dictionary mapping original data index to metrics data.
    """
    matches = {}
    
    # Create cleaned versions of metrics captions for matching
    metrics_clean_captions = []
    for i, metric in enumerate(metrics_data):
        caption1 = clean_text_caparena(metric["cand"].get("caption1", ""))
        caption2 = clean_text_caparena(metric["cand"].get("caption2", ""))
        metric_ref = clean_text_caparena(metric["ref"].get("ref", ""))
        metrics_clean_captions.append((i, caption1, caption2, metric_ref))
    
    # Match original data with metrics data
    for i, item in tqdm(enumerate(original_data), desc="Matching metrics", ncols=88):
        clean_caption1 = clean_text_caparena(item.get("caption1", ""))
        clean_caption2 = clean_text_caparena(item.get("caption2", ""))
        clean_ref = clean_text_caparena(item.get("ref", ""))
        count = 0
        
        for metric_idx, metric_caption1, metric_caption2, metric_ref in metrics_clean_captions:
            # Check if captions match (allowing for partial matches)
            # if (metric_caption1 in clean_caption1 or clean_caption1 in metric_caption1) and \
            #    (metric_caption2 in clean_caption2 or clean_caption2 in metric_caption2):
            if metric_caption1 == clean_caption1 and metric_caption2 == clean_caption2 and clean_ref == metric_ref:
                matches[i] = metric_idx
                count += 1
                # if count > 1:
                #     print(f"Warning: Multiple matches found for item {i}.")
                #     break
                # break
        if count == 0:
            # print(f"Warning: No match found for item {i}.")
            print(item.get("winner"))
    
    return matches

def determine_winner(item, metrics):
    """
    Determine the winner based on all available metrics scores.
    Returns a dictionary with metric names as keys and winner sources as values.
    Also returns a dictionary with metric judgments formatted as "Caption X is better."
    """
    # List of all metrics to consider
    metric_types = [
        "original_spice_score",
        "sub_sentences_spice_score",
        "delete_spice_score",
        "insert_spice_score",
        "combined_spice_score",
        "original_soft_spice_score",
        "sub_sentences_soft_spice_score",
        "delete_soft_spice_score",
        "insert_soft_spice_score",
        "combined_soft_spice_score",
        "sub_sent_capture_score",
        "delete_capture_score",
        "insert_capture_score",
        "combined_capture_score"
    ]
    
    winners = {}
    judgments = {}
    
    # Handle all metrics that have array structure [score1, score2]
    for metric_type in metric_types:
        if metric_type in metrics:
            scores = metrics[metric_type]
            # Higher score is better
            if scores[0] > scores[1]:
                winner = item["source1"]
                judgment = "Caption 1 is better."
            elif scores[1] > scores[0]:
                winner = item["source2"]
                judgment = "Caption 2 is better."
            else:
                winner = "tie"
                judgment = "Tie."
            
            winners[metric_type] = winner
            judgments[metric_type] = judgment
    
    return winners, judgments

def filter_data(data):
    """
    Filter data according to the specified criteria.
    """
    filtered_data = []
    
    for item in data:
        # Skip if no judge field
        if "judge" not in item:
            continue
            
        # Skip if either source is human
        if item["source1"] == "human" or item["source2"] == "human":
            continue
            
        # Skip if judge is not in expected format
        if item["judge"] not in ["Caption 1 is better.", "Caption 1 is better", 
                                "Caption 2 is better.", "Caption 2 is better", 
                                "Tie", "Tie."]:
            continue
            
        filtered_data.append(item)
    
    return filtered_data

def main(args):
    """
    Main function to process the JSON files and add new winner fields.
    """
    # Load original data
    with open('caparena_annots_eval_gpt_ref.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load metrics data
    score_file = args.score_file
    if not score_file.endswith("correlation_score.json"):
        raise ValueError("The score file must end with 'correlation_score.json'")
    with open(score_file, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
    
    # Filter data according to criteria
    filtered_data = filter_data(original_data)
    print(f"Filtered data: {len(filtered_data)} out of {len(original_data)} items")
    
    # Match filtered data with metrics data
    matches = find_matching_metrics(filtered_data, metrics_data)
    print(f"Found matches for {len(matches)} out of {len(filtered_data)} items")
    
    # Add new winner fields based on different metrics
    for i, item in tqdm(enumerate(filtered_data), desc="Processing matches", ncols=88):
        if i in matches:
            metric_idx = matches[i]
            metrics = metrics_data[metric_idx]
            
            # Determine winners and judgments based on different metrics
            winners, judgments = determine_winner(item, metrics)
            
            # Add winners and judgments to original data
            for metric_type, winner in winners.items():
                item[f"winner_{metric_type}"] = winner
                item[f"judge_{metric_type}"] = judgments[metric_type]
            
            # Calculate agreement with human judgment for each metric
            human_judgment = item["judge"]
            for metric_type, judgment in judgments.items():
                # Consider "Caption X is better" and "Caption X is better." as the same
                metric_agrees = (judgment == human_judgment) or \
                               (judgment.rstrip('.') == human_judgment.rstrip('.'))
                item[f"agreement_{metric_type}"] = 1 if metric_agrees else 0
            
            # Add a consensus winner based on majority vote across all metrics
            if winners:
                winner_counts = {}
                for winner in winners.values():
                    winner_counts[winner] = winner_counts.get(winner, 0) + 1
                
                # Find the winner with the most votes
                consensus_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
                item["winner_consensus"] = consensus_winner
                
                # Format consensus judgment
                if consensus_winner == item["source1"]:
                    item["judge_consensus"] = "Caption 1 is better."
                elif consensus_winner == item["source2"]:
                    item["judge_consensus"] = "Caption 2 is better."
                else:
                    item["judge_consensus"] = "Tie."
                
                # Calculate agreement between consensus and human judgment
                consensus_judgment = item["judge_consensus"]
                consensus_agrees = (consensus_judgment == human_judgment) or \
                                  (consensus_judgment.rstrip('.') == human_judgment.rstrip('.'))
                item["agreement_consensus"] = 1 if consensus_agrees else 0
    
    # Calculate overall agreement statistics
    agreement_stats = {metric_type: 0 for metric_type in ["consensus"] + list(winners.keys())}
    agreement_counts = {metric_type: 0 for metric_type in ["consensus"] + list(winners.keys())}
    
    cluster_agreement = {}
    
    for item in filtered_data:
        for metric_type in agreement_stats.keys():
            agreement_key = f"agreement_{metric_type}"
            if agreement_key in item:
                agreement_stats[metric_type] += item[agreement_key]
                agreement_counts[metric_type] += 1
                
                # Track agreement by cluster
                cluster = item.get("cluster", "unknown")
                if cluster not in cluster_agreement:
                    cluster_agreement[cluster] = {metric_type: {"agree": 0, "total": 0} for metric_type in agreement_stats.keys()}
                
                cluster_agreement[cluster][metric_type]["agree"] += item[agreement_key]
                cluster_agreement[cluster][metric_type]["total"] += 1
    
    # Print agreement statistics
    print("\nOverall Agreement Statistics:")
    for metric_type, count in agreement_counts.items():
        if count > 0:
            agreement_rate = agreement_stats[metric_type] / count
            print(f"{metric_type}: {agreement_rate:.4f} ({agreement_stats[metric_type]}/{count})")
    
    print("\nAgreement by Cluster:")
    for cluster, metrics in cluster_agreement.items():
        print(f"\n{cluster}:")
        for metric_type, stats in metrics.items():
            if stats["total"] > 0:
                agreement_rate = stats["agree"] / stats["total"]
                print(f"  {metric_type}: {agreement_rate:.4f} ({stats['agree']}/{stats['total']})")
    
    output_path = score_file.replace("correlation_score.json", "correlation_score_winner_style.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nProcessing complete. Updated data saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and add new winner fields.")
    parser.add_argument('--score_file', type=str, required=True, help="Path to the input JSON file.")
    args = parser.parse_args()
    main(args)
