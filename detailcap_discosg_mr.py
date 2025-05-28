import os
import sys
import json
import copy
import argparse
import torch

from disco_sg_camera.src.factual_scene_graph.dataset_utils import (
    load_detailcaps_dataset,
    load_longparse_dataset,
    collect_unique_captions,
)
from disco_sg_camera.src.factual_scene_graph.parse_utils import (
    load_parsed_captions,
    parse_captions,
    parse_captions_fix,
)
from disco_sg_camera.src.factual_scene_graph.utils import (
    seed_everything,
)
from disco_sg_camera.src.factual_scene_graph.parser.scene_graph_parser_insert_delete import (
    SceneGraphParser,
)
from disco_sg_camera.src.factual_scene_graph.parser.DualTaskSceneGraphParser import (
    DualTaskSceneGraphParser,
)
from disco_sg_camera.src.factual_scene_graph.evaluation.evaluator import Evaluator
from disco_sg_camera.src.factual_scene_graph.eval_utils import (
    evaluate_graphs,
    print_original_metrics,
    print_sub_sentences_metrics,
    print_three_task_metrics,
    print_sub_sentences_metrics_capture,
    print_three_task_metrics_capture,
    evaluate_graphs_capture,
)
from disco_sg_camera.src.factual_scene_graph.triple_utils import (
    merge_delete_insert_results,
    graph_string_to_object,
)
from disco_sg_camera.src.factual_scene_graph.log_utils import (
    save_results_to_json,
)
from modified_capture.capture.capture_metric.capture import CAPTURE

from datetime import datetime
now = datetime.now()
time_tag = now.strftime("%d-%m_%H-%M-%S")


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def compute_correlation_main(args):
    """Main correlation computation function."""
    # Load dataset
    if "DetailCaps" in args.dataset:
        refs, candidates, human_scores = load_detailcaps_dataset(
            dataset_name_or_path="foundation-multimodal-models/DetailCaps-4870",
            split="test",
        )
    elif "discosg" in args.dataset:
        refs, candidates, human_scores = load_longparse_dataset(
            dataset_name_or_path="<path to human_anno_test_100.json>",
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    # for debug
    if args.debug:
        refs = refs[:2]
        candidates = candidates[:2]
        human_scores = human_scores[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = SceneGraphParser(
        "lizhuang144/flan-t5-base-VG-factual-sg",   # we use the flan-t5-base-VG-factual-sg model, which does not break the anonymous policy
        device=device,
        lemmatize=False,
        lowercase=True,
    )
    evaluator = Evaluator(
        parser=parser,
        text_encoder_checkpoint="all-MiniLM-L6-v2",
        device=device,
        lemmatize=True,
    )

    # Process captions
    if "DetailCaps" in args.dataset:
        caption_list = collect_unique_captions(candidates, refs)
    elif "discosg" in args.dataset:
        caption_list = collect_unique_captions(candidates, [])
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.sub_sentence_parse_dict:
        sub_sentences_parse_dict = load_parsed_captions(args.sub_sentence_parse_dict)
        print(f"Loaded sub-sentence parse dict from {args.sub_sentence_parse_dict}")
    else:
        sub_sentences_parse_dict = parse_captions(caption_list, parser, sub_sentence=True, batch_size=16 * args.bs_scale, num_beams=3)
    if args.original_parse_dict:
        parse_dict = load_parsed_captions(args.original_parse_dict)
        print(f"Loaded original parse dict from {args.original_parse_dict}")
    else:
        parse_dict = parse_captions(caption_list, parser, batch_size=64 * args.bs_scale, num_beams=3)

    sub_sentences_parse_dict_for_delete_and_insert_together = copy.deepcopy(sub_sentences_parse_dict)

    dual_task_parser = DualTaskSceneGraphParser(
        model_path=args.model_path,
        device=device,
        lemmatize=False,
        lowercase=True,
    )
    for i_round in range(args.round):
        print(f"Round {i_round + 1} of {args.round}...")
        if i_round == 0 and args.combined_parse_dict:
            dual_task_parse_results = load_parsed_captions(args.combined_parse_dict)
            print(f"Loaded combined parse dict from {args.combined_parse_dict}")
        else:
            dual_task_parse_results = parse_captions_fix(
                descriptions=caption_list,
                graph_to_fix=sub_sentences_parse_dict_for_delete_and_insert_together,
                parser=dual_task_parser,
                task="delete_before_insert",
                batch_size=4 * args.bs_scale,
                max_input_len=args.max_input_length,
                max_output_len=args.max_output_length,
                max_triples_num=args.max_triples_num,
                skip_toolong=args.skip_toolong,
                skip_len=args.skip_len,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )
            dual_task_parse_results["combined"] = {}
            for desc in caption_list:
                _, dual_task_parse_results["combined"][desc] = merge_delete_insert_results(
                    insert_res=dual_task_parse_results["insert"][desc],
                    delete_res=dual_task_parse_results["delete"][desc],
                )

        print("Evaluating parsing results...")

        original_metrics_spice, original_metrics_soft_spice = evaluate_graphs(
            candidates=candidates,
            refs=refs,
            parse_dict=parse_dict,
            evaluator=evaluator,
            return_graphs=True,
        )
        spice_score_original, _, _ = original_metrics_spice
        soft_spice_score_original, _, _ = original_metrics_soft_spice

        sub_sentences_metrics, sub_sentences_metrics_soft_spice = evaluate_graphs(
            candidates=candidates,
            refs=refs,
            parse_dict=sub_sentences_parse_dict,
            evaluator=evaluator,
            return_graphs=True,
        )
        spice_score_sub_sentences, _, _ = sub_sentences_metrics
        soft_spice_score_sub_sentences, _, _ = sub_sentences_metrics_soft_spice

        delete_metrics, delete_metrics_soft_spice = evaluate_graphs(
            candidates=candidates,
            refs=refs,
            parse_dict=dual_task_parse_results["delete"],
            evaluator=evaluator,
            return_graphs=True,
        )
        spice_score_delete, _, _ = delete_metrics
        soft_spice_score_delete, _, _ = delete_metrics_soft_spice

        insert_metrics, insert_metrics_soft_spice = evaluate_graphs(
            candidates=candidates,
            refs=refs,
            parse_dict=dual_task_parse_results["insert"],
            evaluator=evaluator,
            return_graphs=True,
        )
        spice_score_insert, _, _ = insert_metrics
        soft_spice_score_insert, _, _ = insert_metrics_soft_spice

        combined_metrics, combined_metrics_soft_spice = evaluate_graphs(
            candidates=candidates,
            refs=refs,
            parse_dict=dual_task_parse_results["combined"],
            evaluator=evaluator,
            return_graphs=True,
        )
        spice_score_combined, _, _ = combined_metrics
        soft_spice_score_combined, _, _ = combined_metrics_soft_spice

        if isinstance(human_scores[0], dict):
            human_scores_flat = [score for scores in human_scores for score in scores.values()]
        else:
            human_scores_flat = human_scores

        print_original_metrics(spice_score_original, soft_spice_score_original, human_scores_flat)
        print_sub_sentences_metrics(spice_score_sub_sentences, soft_spice_score_sub_sentences, human_scores_flat)
        print_three_task_metrics(
            spice_score_delete,
            soft_spice_score_delete,
            spice_score_insert,
            soft_spice_score_insert,
            spice_score_combined,
            soft_spice_score_combined,
            human_scores_flat,
        )
        if args.capture:
            print("-" * 66)
            print("Evaluating CAPTURE results...")
            capture = CAPTURE()
            sub_sentences_parse_dict_object = {}
            if args.dataset == "DetailCaps":
                for i in range(len(refs)):
                    for key_ref in refs[i].keys():
                        sub_sentences_parse_dict_object[refs[i][key_ref]] = capture.parse_results_post_editing(graph_string_to_object(sub_sentences_parse_dict[refs[i][key_ref]]), refs[i][key_ref])
                for i in range(len(candidates)):
                    for key_cand in candidates[i].keys():
                        sub_sentences_parse_dict_object[candidates[i][key_cand]] = capture.parse_results_post_editing(graph_string_to_object(sub_sentences_parse_dict[candidates[i][key_cand]]), candidates[i][key_cand])
                capture_score_subsentence = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=refs,
                    parse_dict_capture=sub_sentences_parse_dict_object,
                    capture_evaluator=capture,
                )
            else:
                for i in range(len(refs)):
                    sub_sentences_parse_dict_object[refs[i]] = capture.parse_results_post_editing(graph_string_to_object(refs[i]), refs[i])
                for i in range(len(candidates)):
                    sub_sentences_parse_dict_object[candidates[i]] = capture.parse_results_post_editing(graph_string_to_object(sub_sentences_parse_dict[candidates[i]]), candidates[i])
                capture_score_subsentence = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=[sub_sentences_parse_dict_object[ref] for ref in refs],
                    parse_dict_capture=sub_sentences_parse_dict_object,
                    capture_evaluator=capture,
                )
            assert len(refs) == len(candidates) == len(human_scores), f"Length mismatch: {len(refs)}, {len(candidates)}, {len(human_scores)}"

            
            print_sub_sentences_metrics_capture(capture_score_subsentence, human_scores_flat)

            dual_task_parse_results_object = {}
            dual_task_parse_results_object["delete"] = {}
            dual_task_parse_results_object["insert"] = {}
            dual_task_parse_results_object["combined"] = {}

            if args.dataset == "DetailCaps":
                for i in range(len(refs)):
                    for key in dual_task_parse_results.keys():
                        for key_ref in refs[i].keys():
                            dual_task_parse_results_object[key][refs[i][key_ref]] = capture.parse_results_post_editing(graph_string_to_object(dual_task_parse_results[key][refs[i][key_ref]]), refs[i][key_ref])
                for i in range(len(candidates)):
                    for key in dual_task_parse_results.keys():
                        for key_cand in candidates[i].keys():
                            dual_task_parse_results_object[key][candidates[i][key_cand]] = capture.parse_results_post_editing(graph_string_to_object(dual_task_parse_results[key][candidates[i][key_cand]]), candidates[i][key_cand])
                capture_score_insert = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=refs,
                    parse_dict_capture=dual_task_parse_results_object["insert"],
                    capture_evaluator=capture,
                )
                capture_score_delete = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=refs,
                    parse_dict_capture=dual_task_parse_results_object["delete"],
                    capture_evaluator=capture,
                )
                capture_score_combined = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=refs,
                    parse_dict_capture=dual_task_parse_results_object["combined"],
                    capture_evaluator=capture,
                )
            else:
                for i in range(len(refs)):
                    for key in dual_task_parse_results.keys():
                        dual_task_parse_results_object[key][refs[i]] = capture.parse_results_post_editing(graph_string_to_object(refs[i]), refs[i])
                for i in range(len(candidates)):
                    for key in dual_task_parse_results.keys():
                        dual_task_parse_results_object[key][candidates[i]] = capture.parse_results_post_editing(graph_string_to_object(dual_task_parse_results[key][candidates[i]]), candidates[i])        
                capture_score_insert = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=[dual_task_parse_results_object["insert"][ref] for ref in refs],
                    parse_dict_capture=dual_task_parse_results_object["insert"],
                    capture_evaluator=capture,
                )
                capture_score_delete = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=[dual_task_parse_results_object["delete"][ref] for ref in refs],
                    parse_dict_capture=dual_task_parse_results_object["delete"],
                    capture_evaluator=capture,
                )
                capture_score_combined = evaluate_graphs_capture(
                    candidates=candidates,
                    refs=[dual_task_parse_results_object["combined"][ref] for ref in refs],
                    parse_dict_capture=dual_task_parse_results_object["combined"],
                    capture_evaluator=capture,
                )
            assert len(refs) == len(candidates), "refs and candidates should have the same length"

            
            print_three_task_metrics_capture(
                capture_score_delete,
                capture_score_insert,
                capture_score_combined,
                human_scores_flat,
            )
            
        print("-" * 66)
        print("Finished evaluating CAPTURE results.")

        # Save results to JSON
        save_results_to_json(
            args=args,
            candidates=candidates,
            refs=refs,
            human_scores=human_scores_flat,
            dual_task_parse_results=dual_task_parse_results,
            sub_sentences_parse_dict=sub_sentences_parse_dict,
            parse_dict=parse_dict,
            spice_score_original=spice_score_original,
            spice_score_sub_sentences=spice_score_sub_sentences,
            spice_score_delete=spice_score_delete,
            spice_score_insert=spice_score_insert,
            spice_score_combined=spice_score_combined,
            soft_spice_score_original=soft_spice_score_original,
            soft_spice_score_sub_sentences=soft_spice_score_sub_sentences,
            soft_spice_score_delete=soft_spice_score_delete,
            soft_spice_score_insert=soft_spice_score_insert,
            soft_spice_score_combined=soft_spice_score_combined,
            save_path=f"{args.save_folder}/{args.dataset}/correlation_results_{args.dataset}_{args.model_path.split('/')[-3]}_skip_toolong_{args.skip_toolong}_skip_len_{args.skip_len}_max_input_len_{args.max_input_length}_max_output_len_{args.max_output_length}_max_triples_num_{args.max_triples_num}_bs_scale_{args.bs_scale}_do_sample_{args.do_sample}_num_beams_{args.num_beams}_top_k_{args.top_k}_top_p_{args.top_p}_temperature_{args.temperature}/round_{i_round}",
            time_tag=time_tag,
            capture_score_subsentence=capture_score_subsentence if args.capture else None,
            capture_score_delete=capture_score_delete if args.capture else None,
            capture_score_insert=capture_score_insert if args.capture else None,
            capture_score_combined=capture_score_combined if args.capture else None
        )

        # update for multi round
        if i_round < args.round - 1:
            sub_sentences_parse_dict_for_delete_and_insert_together = copy.deepcopy(dual_task_parse_results["combined"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlation between human and model scores")
    parser.add_argument("--dataset", type=str, default="DetailCaps", help="Dataset name", choices=["DetailCaps", "discosg"])
    parser.add_argument("--model_path", type=str, default="None", help="Path to the dual task model")
    parser.add_argument("--max_input_length", type=int, default=2048, help="Max input length for the model")
    parser.add_argument("--max_output_length", type=int, default=512, help="Max output length for the model")
    parser.add_argument("--max_triples_num", type=int, default=256, help="Max number of triples for the model")
    parser.add_argument("--capture", action="store_true", default=False, help="Whether to use CAPTURE metric")
    parser.add_argument("--skip_toolong", action="store_true", default=False, help="Whether to skip long sentences")
    parser.add_argument("--skip_len", type=int, default=-1, help="Max length of sentences")
    parser.add_argument("--bs_scale", type=int, default=1, help="Batch size scale")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to sample the dataset")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search, 1 for greedy search")
    parser.add_argument("--top_k", type=int, default=50, help="Top k for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p for nucleus sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    parser.add_argument("--save_folder", type=str, default="eval_res", help="Folder to save results")
    parser.add_argument("--round", type=int, default=3, help="Multi round")
    parser.add_argument("--original_parse_dict", type=str, default=None, help="Path to the original parse dict file")
    parser.add_argument("--sub_sentence_parse_dict", type=str, default=None, help="Path to the sub sentence parse dict file")
    parser.add_argument("--combined_parse_dict", type=str, default=None, help="Path to the combined parse dict file")

    args = parser.parse_args()
    
    seed_everything(args.seed)
    compute_correlation_main(args)
