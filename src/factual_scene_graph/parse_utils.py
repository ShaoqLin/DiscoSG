import json
from typing import List, Dict, Any

def load_parsed_captions(
    file_path: str,
) -> Dict[str, Any]:
    """Load parsed captions from a file."""
    if file_path.endswith(".json"):
        print(f"Loading parsed captions from {file_path}")
        with open(file_path, "r") as f:
            parsed_captions = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return parsed_captions

def parse_captions(
    captions: List[str], parser, sub_sentence: bool = False, batch_size: int = 32, num_beams: int = 3
) -> Dict[str, Any]:
    """Parse captions with enhanced processing."""
    parse_results = parser.parse(
        captions,
        batch_size=batch_size,
        max_input_len=512 if sub_sentence else 256,
        max_output_len=512 if sub_sentence else 256,
        num_beams=num_beams,
        return_text=True,
        sub_sentence=sub_sentence,
        num_return_sequences=5,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    return {caption: result for caption, result in zip(captions, parse_results)}

def parse_captions_fix(
    descriptions: List[str],
    graph_to_fix,
    parser,
    skip_toolong,
    skip_len,
    max_input_len: int = 2048,
    max_output_len: int = 512,
    max_triples_num: int = 256,
    task: str = "insert_delete",
    batch_size: int = 32,
    num_beams: int = 1,
    do_sample: bool = False,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Parse captions with enhanced processing."""
    parse_results = parser.parse(
        descriptions=descriptions,
        graph_to_fix=graph_to_fix,
        batch_size=batch_size,
        task=task,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_triples_num=max_triples_num,
        skip_toolong=skip_toolong,
        skip_len=skip_len,
        num_beams=num_beams,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    if task == "insert" or task == "delete":
        return {caption: result for caption, result in zip(descriptions, parse_results)}
    elif task == "insert_delete":
        delete_results = parse_results["delete"]
        insert_results = parse_results["insert"]
        return {"delete": delete_results, "insert": insert_results}
    elif task == "delete_before_insert":
        delete_results = parse_results["delete"]
        insert_results = parse_results["insert"]
        return {"delete": delete_results, "insert": insert_results}
    else:
        raise ValueError(f"Unsupported task: {task}")

