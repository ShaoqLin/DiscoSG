import scipy

from typing import Dict, List, Any, Tuple
from disco_sg_camera.src.factual_scene_graph.evaluation.evaluator import Evaluator

def print_correlation(avg_score, correlation_c, correlation_b, pearson_correlation, score_name):
    """Print the correlation results."""
    print(f"{score_name} score: {avg_score:.3f}")
    print(f"{score_name} Tau-c: {100 * correlation_c:.3f}")
    print(f"{score_name} Tau-b: {100 * correlation_b:.3f}")
    print(f"{score_name} Pearson correlation: {100 * pearson_correlation:.3f}")

def compute_correlation(scores, human_scores, score_name):
    """Compute all kendall and pearson correlation between model scores and human scores."""
    avg_score = sum(scores) / len(scores)
    correlation_c = scipy.stats.kendalltau(scores, human_scores, variant="c")[0]
    correlation_b = scipy.stats.kendalltau(scores, human_scores, variant="b")[0]
    pearson_correlation = scipy.stats.pearsonr(scores, human_scores)[0]
    print_correlation(avg_score, correlation_c, correlation_b, pearson_correlation, score_name)
    return avg_score, correlation_c, correlation_b, pearson_correlation

def print_original_metrics(spice_score_original, soft_spice_score_original, human_scores_flat):
    print("---- Original SPICE score ----")
    spice_score_original, Tau_c_ori, Tau_b_ori, Pearson_ori = compute_correlation(spice_score_original, human_scores_flat, score_name="SPICE score original")
    print("---- Original Soft SPICE score ----")
    soft_spice_score_original, Tau_c_ori_soft, Tau_b_ori_soft, Pearson_ori_soft = compute_correlation(soft_spice_score_original, human_scores_flat, score_name="Soft SPICE score original")

def print_sub_sentences_metrics(spice_score_sub_sentences, soft_spice_score_sub_sentences, human_scores_flat):
    print("---- Sub-sentences SPICE score ----")
    spice_score_sub_sentences, Tau_c_sub_sentences, Tau_b_sub_sentences, Pearson_sub_sentences = compute_correlation(spice_score_sub_sentences, human_scores_flat, score_name="SPICE score sub-sentences")
    print("---- Sub-sentences Soft SPICE score ----")
    soft_spice_score_sub_sentences, Tau_c_sub_sentences_soft, Tau_b_sub_sentences_soft, Pearson_sub_sentences_soft = compute_correlation(soft_spice_score_sub_sentences, human_scores_flat, score_name="Soft SPICE score sub-sentences")

def print_three_task_metrics(spice_score_delete, soft_spice_score_delete, spice_score_insert, soft_spice_score_insert, spice_score_combined, soft_spice_score_combined, human_scores_flat):
    print("---- Delete SPICE score ----")
    spice_score_delete, Tau_c_delete, Tau_b_delete, Pearson_delete = compute_correlation(spice_score_delete, human_scores_flat, score_name="SPICE score for delete task")
    print("---- Delete Soft SPICE score ----")
    soft_spice_score_delete, Tau_c_delete_soft, Tau_b_delete_soft, Pearson_delete_soft = compute_correlation(soft_spice_score_delete, human_scores_flat, score_name="Soft SPICE score for delete task")
    print("---- Insert SPICE score ----")
    spice_score_insert, Tau_c_insert, Tau_b_insert, Pearson_insert = compute_correlation(spice_score_insert, human_scores_flat, score_name="SPICE score for insert task")
    print("---- Insert Soft SPICE score ----")
    soft_spice_score_insert, Tau_c_insert_soft, Tau_b_insert_soft, Pearson_insert_soft = compute_correlation(soft_spice_score_insert, human_scores_flat, score_name="Soft SPICE score for insert task")
    print("---- Combined SPICE score ----")
    spice_score_combined, Tau_c_combined, Tau_b_combined, Pearson_combined = compute_correlation(spice_score_combined, human_scores_flat, score_name="SPICE score for combined task")
    print("---- Combined Soft SPICE score ----")
    soft_spice_score_combined, Tau_c_combined_soft, Tau_b_combined_soft, Pearson_combined_soft = compute_correlation(soft_spice_score_combined, human_scores_flat, score_name="Soft SPICE score for combined task")

def print_original_metrics_capture(capture_score_original, human_scores_flat):
    print("---- Original CAPTURE score ----")
    capture_score_original, Tau_c_ori, Tau_b_ori, Pearson_ori = compute_correlation(capture_score_original[1], human_scores_flat, score_name="CAPTURE score original")

def print_sub_sentences_metrics_capture(capture_score_sub_sentences, human_scores_flat):
    print("---- Sub-sentences CAPTURE score ----")
    capture_score_sub_sentences, Tau_c_sub_sentences, Tau_b_sub_sentences, Pearson_sub_sentences = compute_correlation(capture_score_sub_sentences[1], human_scores_flat, score_name="CAPTURE score sub-sentences")

def print_three_task_metrics_capture(capture_score_delete, capture_score_insert, capture_score_combined, human_scores_flat):
    print("---- Delete CAPTURE score ----")
    capture_score_delete, Tau_c_delete, Tau_b_delete, Pearson_delete = compute_correlation(capture_score_delete[1], human_scores_flat, score_name="CAPTURE score for delete task")
    print("---- Insert CAPTURE score ----")
    capture_score_insert, Tau_c_insert, Tau_b_insert, Pearson_insert = compute_correlation(capture_score_insert[1], human_scores_flat, score_name="CAPTURE score for insert task")
    print("---- Combined CAPTURE score ----")
    capture_score_combined, Tau_c_combined, Tau_b_combined, Pearson_combined = compute_correlation(capture_score_combined[1], human_scores_flat, score_name="CAPTURE score for combined task")

def evaluate_graphs(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict: Dict[str, Any],
    evaluator: Evaluator,
    return_graphs: bool,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""

    cand_graphs = [
        (
            [parse_dict[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [parse_dict[cand]]
        )
        for cand in candidates
    ]
    ref_graphs = [
        [parse_dict[ref] for ref in ref.values()] if isinstance(ref, dict) else [ref]
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    length_diff_times = len(cand_graphs) // len(ref_graphs)
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(length_diff_times)]
    assert len(cand_graphs) == len(ref_graphs), "Candidate and reference lengths do not match"

    spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
    )

    soft_spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="soft_spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
        bidirectional=True,
    )

    return spice_res, soft_spice_res

def evaluate_graphs_flickr8k(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict: Dict[str, Any],
    evaluator: Evaluator,
    return_graphs: bool,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        [parse_dict[cand] for cand in cand.values()] if isinstance(cand, dict) else [parse_dict[cand]] for cand in candidates
    ]
    ref_graphs = [[parse_dict[ref_i] for ref_i in ref] for ref in refs]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(3)]

    spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
    )

    soft_spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="soft_spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
        bidirectional=True,
    )

    return spice_res, soft_spice_res

def evaluate_graphs_sft(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict: Dict[str, Any],
    evaluator: Evaluator,
    return_graphs: bool,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        (
            [parse_dict[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [cand]
        )
        for cand in candidates
    ]
    ref_graphs = [
        [parse_dict[ref] for ref in ref.values()] if isinstance(ref, dict) else [ref]
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(3)]

    spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
    )

    soft_spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="soft_spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
        bidirectional=True,
    )

    return spice_res, soft_spice_res

def evaluate_graphs_caparena(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict: Dict[str, Any],
    evaluator: Evaluator,
    return_graphs: bool,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        (
            [parse_dict[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [parse_dict[cand]]
        )
        for cand in candidates
    ]
    ref_graphs = [
        [parse_dict[ref] for ref in ref.values()] if isinstance(ref, dict) else [ref]
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(2)]

    spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
    )

    soft_spice_res = evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="soft_spice",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
        bidirectional=True,
    )

    return spice_res, soft_spice_res

def evaluate_graphs_capture(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict_capture: Dict[str, Any],
    capture_evaluator: Evaluator,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        (
            [parse_dict_capture[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [parse_dict_capture[cand]]
        )
        for cand in candidates
    ]
    ref_graphs = [
        (
            [parse_dict_capture[ref] for ref in ref.values()]
            if isinstance(ref, dict)
            else [ref]
        )
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(3)]

    assert len(cand_graphs) == len(
        ref_graphs
    ), "Candidate and reference lengths do not match"
    refs_example_dict = {}
    cands_example_dict = {}
    for i in range(len(ref_graphs)):
        example_id = f"example_{i}"
        refs_example_dict[example_id] = ref_graphs[i]
        cands_example_dict[example_id] = cand_graphs[i]

    capture_score = capture_evaluator.compute_score_post(
        refs_example_dict,
        cands_example_dict,
    )

    return capture_score

def evaluate_graphs_ori(candidates, refs, parse_dict, evaluator, return_graphs):
    """Evaluate the graphs and return the results."""
    cand_graphs = [parse_dict[cand] for cand in candidates]
    ref_graphs = [[parse_dict[ref_i] for ref_i in ref] for ref in refs]
    spice_out = evaluator.evaluate(cand_graphs, ref_graphs, method='spice',
                              beam_size=1, batch_size=64, max_input_len=512,
                              max_output_len=512, return_graphs=return_graphs)
    soft_spice_out = evaluator.evaluate(cand_graphs, ref_graphs, method='soft_spice',
                                beam_size=1, batch_size=64, max_input_len=512,
                                max_output_len=512, return_graphs=return_graphs, bidirectional=True,)
    return spice_out, soft_spice_out

def evaluate_graphs_capture_flicke8k(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict_capture: Dict[str, Any],
    capture_evaluator: Evaluator,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        (
            [parse_dict_capture[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [parse_dict_capture[cand]]
        )
        for cand in candidates
    ]
    ref_graphs = [
        (
            [parse_dict_capture[ref] for ref in ref.values()]
            if isinstance(ref, dict)
            else [parse_dict_capture[r] for r in ref]
        )
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(3)]

    assert len(cand_graphs) == len(
        ref_graphs
    ), "Candidate and reference lengths do not match"
    refs_example_dict = {}
    cands_example_dict = {}
    for i in range(len(ref_graphs)):
        example_id = f"example_{i}"
        refs_example_dict[example_id] = ref_graphs[i]
        cands_example_dict[example_id] = cand_graphs[i]

    capture_score = capture_evaluator.compute_score_post(
        refs_example_dict,
        cands_example_dict,
    )

    return capture_score

def evaluate_graphs_capture_caparena(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict_capture: Dict[str, Any],
    capture_evaluator: Evaluator,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    cand_graphs = [
        (
            [parse_dict_capture[cand] for cand in cand.values()]
            if isinstance(cand, dict)
            else [parse_dict_capture[cand]]
        )
        for cand in candidates
    ]
    ref_graphs = [
        (
            [parse_dict_capture[ref] for ref in ref.values()]
            if isinstance(ref, dict)
            else [ref]
        )
        for ref in refs
    ]

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    if len(cand_graphs) != len(ref_graphs):
        ref_graphs = [item for item in ref_graphs for _ in range(2)]

    assert len(cand_graphs) == len(
        ref_graphs
    ), "Candidate and reference lengths do not match"
    refs_example_dict = {}
    cands_example_dict = {}
    for i in range(len(ref_graphs)):
        example_id = f"example_{i}"
        refs_example_dict[example_id] = ref_graphs[i]
        cands_example_dict[example_id] = cand_graphs[i]

    capture_score = capture_evaluator.compute_score_post(
        refs_example_dict,
        cands_example_dict,
    )

    return capture_score

def set_match_evaluate_graphs(
    candidates: List[Dict[str, str]],
    refs: List[str],
    parse_dict: Dict[str, Any],
    evaluator: Evaluator,
    return_graphs: bool,
) -> Tuple[List[float], Any, Any]:
    """Enhanced graph evaluation."""
    # check if candidates is a list of dicts
    if isinstance(candidates[0], dict):
        cand_graphs = [
            [parse_dict[value] for value in cand.values()] for cand in candidates
        ]
        ref_graphs = [[parse_dict[ref]] for ref in refs]
    else:
        cand_graphs = [[parse_dict[cand]] for cand in candidates]
        ref_graphs = [[ref] for ref in refs]

    # Normalize lengths
    # check if cand_graphs is a list of lists
    if isinstance(candidates[0], dict):
        for i in range(len(cand_graphs)):
            if len(cand_graphs[i]) != 3:
                print(
                    f"Warning: Candidate {i} has {len(cand_graphs[i])} graphs instead of 3"
                )
                continue
            ref_graphs[i] = ref_graphs[i] * 3
    else:
        ref_graphs = ref_graphs  # do nothing
        assert len(cand_graphs) == len(
            ref_graphs
        ), "Candidate and reference lengths do not match"

    # Flatten lists
    cand_graphs = [item for sublist in cand_graphs for item in sublist]
    ref_graphs = [[item] for sublist in ref_graphs for item in sublist]

    return evaluator.evaluate(
        cand_graphs,
        ref_graphs,
        method="set_match",
        # num_beams=1,
        batch_size=32,
        max_input_len=512,
        max_output_len=512,
        return_graphs=return_graphs,
        # num_return_sequences=5,
        # temperature=0.5,
        # top_k=50,
        # top_p=0.95,
        do_sample=False,
    )
