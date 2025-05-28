import json
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List
from disco_sg_camera.src.factual_scene_graph.triple_utils import (
    clean_text,
    clean_text_caparena,
    extract_triples,
)

def collect_unique_captions(
    candidates: List[Dict[str, str]], refs: List[str]
) -> List[str]:
    """Collect unique captions from candidates and references."""
    caption_set = set()
    for cand in candidates:
        # check if cand is a dict
        if isinstance(cand, dict):
            for value in cand.values():
                caption_set.add(value)
        else:
            caption_set.add(cand)
    for ref in refs:
        # check if ref is a dict
        if isinstance(ref, dict):
            for value in ref.values():
                caption_set.add(value)
        else:
            caption_set.add(ref)
    return list(caption_set)

def collect_unique_captions_flickr8k(candidates, refs):
    """Collect unique captions from candidates and references."""
    caption_set = set(candidates)  # Add all candidates to the set
    for ref_list in refs:
        caption_set.update(ref_list)  # Add all elements from each list in refs
    return list(caption_set)

def load_detailcaps_dataset(
    dataset_name_or_path="foundation-multimodal-models/DetailCaps-4870", split="test"
):
    dataset = load_dataset(dataset_name_or_path, split=split)

    refs = []
    candidates = []
    human_scores = []

    # Process dataset
    for data in tqdm(dataset, desc="Processing dataset"):
        if data["GPT4_Eval"] is None:
            print("Skipping entry with None eval results")
            continue

        candidate = {
            "CogVLM": clean_text(data["CogVLM"]),
            "ShareCaptioner": clean_text(data["ShareCaptioner"]),
            "LLaVA_v15": clean_text(data["LLaVA_v15"]),
        }
        candidates.append(candidate)

        ref = {
            "GT_Caption_GPT4O": clean_text(data["GT_Caption_GPT4O"]),
            "GT_Caption_GPT4V": clean_text(data["GT_Caption_GPT4V"]),
            "GT_Caption_Gemini15Pro": clean_text(data["GT_Caption_Gemini15Pro"]),
        }
        refs.append(ref)

        human_score = {
            "CogVLM": json.loads(data["GPT4_Eval"])["CogVLM"],
            "ShareCaptioner": json.loads(data["GPT4_Eval"])["ShareCaptioner"],
            "LLaVA_v15": json.loads(data["GPT4_Eval"])["LLaVA_v15"],
        }
        human_scores.append(human_score)

    print(f"Processed {len(refs)} references and {len(candidates)} candidates")
    return refs, candidates, human_scores


def load_longparse_dataset(
    dataset_name_or_path="XXX",
):
    dataset = load_dataset("json", data_files=dataset_name_or_path, split="train")
    refs = []
    candidates = []
    human_scores = []

    # Process dataset
    for data in tqdm(dataset, desc="Processing dataset"):
        candidates.append(clean_text(data["caption"]))
        refs.append(clean_text(data["scene_graph"]))
        human_scores.append(1.0)

    return refs, candidates, human_scores


def load_sft_prompted_longparse_dataset(
    dataset_name_or_path="XXX",
):
    dataset = load_dataset("json", data_files=dataset_name_or_path, split="train")
    refs = []
    candidates = []
    human_scores = []

    # Process dataset
    for data in tqdm(dataset, desc="Processing dataset"):
        cand = clean_text(data["generated_scene_graph"])
        cand_triples = extract_triples(cand)
        if len(cand_triples) > 1:
            for triple in cand_triples:
                if (
                    "(" in triple[0]
                    or ")" in triple[0]
                    or "(" in triple[1]
                    or ")" in triple[1]
                    or "(" in triple[2]
                    or ")" in triple[2]
                ):
                    cand_triples.remove(triple)
            cand_triples = " , ".join(
                [
                    f"( {triple[0]} , {triple[1]} , {triple[2]} )"
                    for triple in cand_triples
                ]
            )
            candidates.append(cand_triples)
        else:
            candidates.append("")
        refs.append(clean_text(data["scene_graph"]))
        human_scores.append(1.0)

    return refs, candidates, human_scores


def load_caparena_dataset(
    dataset_name_or_path="XXX",
):
    dataset = load_dataset("json", data_files=dataset_name_or_path, split="train")
    refs = []
    candidates = []
    human_scores = []

    # Process dataset
    for data in tqdm(dataset, desc="Processing dataset"):
        if data["winner"] == "skip" or data["winner"] == "bad":
            continue
        candidates.append(
            {
                "caption1": clean_text_caparena(data["caption1"]),
                "caption2": clean_text_caparena(data["caption2"]),
            }
        )
        refs.append({"ref": clean_text_caparena(data["ref"])})
        if data["winner"] == data["source1"]:
            human_scores.append({"caption1": 1.0, "caption2": 0.0})
        elif data["winner"] == data["source2"]:
            human_scores.append({"caption1": 0.0, "caption2": 1.0})
        elif data["winner"] == "equal":
            human_scores.append({"caption1": 0.5, "caption2": 0.5})
        else:
            raise Exception("winner is not in source1 or source2")

    return refs, candidates, human_scores


def load_discourseFOIL_dataset(
    dataset_name_or_path="XXX",
):
    dataset = load_dataset("json", data_files=dataset_name_or_path, split="train")
    refs = []
    candidates = []
    human_scores = []

    # Process dataset
    for data in tqdm(dataset, desc="Processing dataset"):
        if data["fixed"] != "true":
            continue
        candidates.append(
            {
                "hall_caption": clean_text(data["hall_caption"]),
                "human_corrected_hall_caption": clean_text(data["human_corrected_hall_caption"]),
            }
        )
        refs.append({"caption": clean_text(data["caption"])})

        human_scores.append({"hall_caption": 0.0, "human_corrected_hall_caption": 1.0})

    return refs, candidates, human_scores


def load_flickr8k_dataset(
    dataset_name_or_path="XXX",
):
    data = {}
    with open(dataset_name_or_path) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue

            candidate = ' '.join(human_judgement['caption'].split())
            candidates.append(candidate)

            ref = [' '.join(gt.split()) for gt in v['ground_truth']]
            refs.append(ref)
            human_scores.append(human_judgement['rating'])
    print('Loaded {} references and {} candidates'.format(len(refs), len(candidates)))
    assert len(candidates) == len(refs)

    return refs, candidates, human_scores
