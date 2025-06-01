# Calculate caption-level agreement and model-level agreement based on metrics annotation results
# Usage: python caparena_metrics.py --eval_dir data/eval/caparena_annots_eval_gpt_ref.json --metric_type combined_spice_score
import json
from cal_ranking import calculate_elo_rankings
from scipy.stats import spearmanr, kendalltau
import argparse
import os
import copy


def cal_agreement(caption_eval_pair_list, metric_type=None, include_tie=False, in_400=False):
    """
    Calculate agreement between human judgments and either GPT or metric judgments.
    If metric_type is provided, calculate agreement with that metric instead of GPT.
    """
    agreement_level = {"overall": [], "level 1": [], "level 2": [], "level 3": [], "level 4": []}
    tie_num = {"level 1": 0, "level 2": 0, "level 3": 0, "level 4": 0}
    
    for item in caption_eval_pair_list:
        # Skip items without necessary fields
        if "winner" not in item:
            continue

        # Skip human vs model comparisons
        if item["source1"] == "human" or item["source2"] == "human":
            continue

        # Filter for in-400 subset if requested
        if in_400 and not item.get("in-400", False):
            continue
        
        # Get the metric winner if metric_type is specified
        if metric_type:
            metric_winner_key = f"winner_{metric_type}"
            if metric_winner_key not in item:
                continue
            metric_winner = item[metric_winner_key]
            
            # Convert metric winner to judgment format
            if metric_winner == item["source1"]:
                metric_judgment = "Caption 1 is better."
            elif metric_winner == item["source2"]:
                metric_judgment = "Caption 2 is better."
            else:
                metric_judgment = "Tie."
        else:
            # Use GPT judgment
            if "judge" not in item:
                continue
            metric_judgment = item["judge"]
            if metric_judgment not in ["Caption 1 is better.", "Caption 1 is better", "Caption 2 is better.", "Caption 2 is better", "Tie", "Tie."]:
                continue
        
        # Get human judgment
        if item["winner"] == item["source1"]:
            human_judgment = "Caption 1 is better."
        elif item["winner"] == item["source2"]:
            human_judgment = "Caption 2 is better."
        else:
            human_judgment = "Tie."

        # Calculate agreement
        if not include_tie:
            if human_judgment != "Tie." and metric_judgment not in ["Tie.", "Tie"]:
                agree = 1 if (metric_judgment in human_judgment or human_judgment in metric_judgment) else 0
                agreement_level["overall"].append(agree)
                agreement_level[item["cluster"]].append(agree)
        else:
            agree = 1 if (metric_judgment in human_judgment or human_judgment in metric_judgment) else 0
            agreement_level["overall"].append(agree)
            agreement_level[item["cluster"]].append(agree)

            if metric_judgment in ["Tie.", "Tie"]:
                tie_num[item["cluster"]] += 1

    # Calculate agreement percentages
    overall = sum(agreement_level["overall"]) / len(agreement_level["overall"]) if len(
        agreement_level["overall"]) > 0 else None
    level1 = sum(agreement_level["level 1"]) / len(agreement_level["level 1"]) if len(
        agreement_level["level 1"]) > 0 else None
    level2 = sum(agreement_level["level 2"]) / len(agreement_level["level 2"]) if len(
        agreement_level["level 2"]) > 0 else None
    level3 = sum(agreement_level["level 3"]) / len(agreement_level["level 3"]) if len(
        agreement_level["level 3"]) > 0 else None
    level4 = sum(agreement_level["level 4"]) / len(agreement_level["level 4"]) if len(
        agreement_level["level 4"]) > 0 else None
    
    # Count items in each level
    overall_num = len(agreement_level["overall"])
    level1_num = len(agreement_level["level 1"])
    level2_num = len(agreement_level["level 2"])
    level3_num = len(agreement_level["level 3"])
    level4_num = len(agreement_level["level 4"])

    # Format and print results
    result = (
        f"Overall: {overall if overall is None else f'{overall:.3f}'} ({overall_num}), "
        f"Level 1: {level1 if level1 is None else f'{level1:.3f}'} ({level1_num}), "
        f"Level 2: {level2 if level2 is None else f'{level2:.3f}'} ({level2_num}), "
        f"Level 3: {level3 if level3 is None else f'{level3:.3f}'} ({level3_num}), "
        f"Level 4: {level4 if level4 is None else f'{level4:.3f}'} ({level4_num})"
    )
    print(result)

    # Print tie counts if requested
    if include_tie:
        level1_num = tie_num["level 1"]
        level2_num = tie_num["level 2"]
        level3_num = tie_num["level 3"]
        level4_num = tie_num["level 4"]
        result_tie_num = f"Level 1: {level1_num}, Level 2: {level2_num}, Level 3: {level3_num}, Level 4: {level4_num}"
        print(f"Tie counts: {result_tie_num}")
    
    return overall, agreement_level


def prepare_data_for_elo(data, metric_type=None):
    """
    Prepare data for ELO calculation by converting metric winners to judge format.
    """
    processed_data = copy.deepcopy(data)
    
    if metric_type:
        metric_winner_key = f"winner_{metric_type}"
        
        for item in processed_data:
            if metric_winner_key in item:
                metric_winner = item[metric_winner_key]
                
                # Convert metric winner to judgment format
                if metric_winner == item["source1"]:
                    item["judge"] = "Caption 1 is better."
                elif metric_winner == item["source2"]:
                    item["judge"] = "Caption 2 is better."
                else:
                    item["judge"] = "Tie."
    
    return processed_data


def cal_model_level_agreement(sorted_model_names, ranking_human=["GPT-4o-0806", "human", "Gemini-2.0-flash-exp", "InternVL2-26B", "Gemini-1.5-pro-002",
                                                                "Claude-3.5-Sonnet-0620", "GPT-4o-mini-0718", "LLama-3.2-90B", "Qwen2-VL-72B-Instruct", 
                                                                "CogVLM2-llama3-chat-19B", "MiniCPM-V2.6-8B", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct",
                                                                "LLaVA-1.6-34B", "LLaVA-1.5-7B"]):
    """
    Calculate model-level agreement between metric rankings and human rankings.
    """
    print(f"Num models: {len(ranking_human)}")
    print("Human ranking:")
    print(ranking_human)

    if "human" in sorted_model_names:
        sorted_model_names.remove("human")
    print("Metrics ranking:")
    print(sorted_model_names)
    
    # Only consider models present in both rankings
    common_models = [model for model in sorted_model_names if model in ranking_human]
    
    # Get rankings for common models
    human_ranking = [ranking_human.index(model) + 1 for model in common_models]
    metric_ranking = [sorted_model_names.index(model) + 1 for model in common_models]
    
    print(f"Common models: {len(common_models)}")
    print(f"Human ranking of common models: {human_ranking}")
    print(f"Metric ranking of common models: {metric_ranking}")

    # Calculate correlation coefficients
    if len(human_ranking) > 1:
        # Calculate Spearman correlation coefficient
        rho, p_value = spearmanr(human_ranking, metric_ranking)
        print(f"Spearman ρ: {rho:.3f} (p-value: {p_value:.3f})")

        # Calculate Kendall Tau correlation coefficient
        tau, kendall_p_value = kendalltau(human_ranking, metric_ranking)
        print(f"Kendall Tau: {tau:.3f} (p-value: {kendall_p_value:.3f})")
        
        return rho, tau
    else:
        print("Not enough common models to calculate correlations")
        return None, None


def cal_metrics_agreement(eval_dir, metric_type=None):
    """
    Calculate agreement between human judgments and metric judgments.
    """
    # Load evaluation data
    data = json.load(open(eval_dir, 'r'))
    
    # Calculate caption-level agreement
    print(f"Caption-level agreement for {metric_type if metric_type else 'GPT judgments'}:")
    overall, _ = cal_agreement(data, metric_type=metric_type, include_tie=True, in_400=False)
    
    # Prepare data for ELO calculation
    processed_data = prepare_data_for_elo(data, metric_type)
    
    # Create a temporary file for calculate_elo_rankings
    temp_file = f"temp_{metric_type if metric_type else 'original'}.json"
    with open(temp_file, 'w') as f:
        json.dump(processed_data, f)
    
    # Calculate model rankings
    print(f"Model-level agreement for {metric_type if metric_type else 'GPT judgments'}:")
    sorted_model_names = calculate_elo_rankings(temp_file)
    
    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Calculate model-level agreement
    rho, tau = cal_model_level_agreement(sorted_model_names)
    
    return {
        "metric_type": metric_type if metric_type else "original",
        "caption_agreement": overall,
        "spearman": rho,
        "kendall_tau": tau
    }


def compare_metrics(eval_dir, metric_types):
    """
    Compare different metrics by calculating agreement for each.
    """
    results = []
    
    # Calculate agreement for original GPT judgments
    print("\n===== Original GPT Judgments =====")
    original_result = cal_metrics_agreement(eval_dir)
    results.append(original_result)
    
    # Calculate agreement for each metric type
    for metric_type in metric_types:
        print(f"\n===== {metric_type} =====")
        metric_result = cal_metrics_agreement(eval_dir, metric_type)
        results.append(metric_result)
    
    # Print comparison table
    print("\n===== Metric Comparison =====")
    print(f"{'Metric':<30} {'Caption Agreement':<20} {'Spearman ρ':<15} {'Kendall τ':<15}")
    print("-" * 80)
    for result in results:
        spearman = result['spearman']
        kendall = result['kendall_tau']
        print(f"{result['metric_type']:<30} {f'{result['caption_agreement']:.3f}':<20} " + 
              f"{spearman if spearman is None else f'{spearman:.3f}':<15} " +
              f"{kendall if kendall is None else f'{kendall:.3f}':<15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate metrics agreement')
    parser.add_argument('--eval_dir', type=str, default="caparena_correlation_score_winner_style.json", help='Path to JSON file containing caption evaluation candidates with metric winners')
    parser.add_argument('--metric_type', type=str, help='Specific metric type to evaluate')
    parser.add_argument('--compare_all', action='store_true', default=True, help='Compare all available metrics')
    parser.add_argument('--with_equal', action='store_true', default=False, help='Include ties in agreement calculation')
    args = parser.parse_args()
    
    if args.compare_all:
        # List of all metrics to consider
        all_metrics = [
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
        compare_metrics(args.eval_dir, all_metrics)
    elif args.metric_type:
        cal_metrics_agreement(args.eval_dir, args.metric_type)
    else:
        cal_metrics_agreement(args.eval_dir)
