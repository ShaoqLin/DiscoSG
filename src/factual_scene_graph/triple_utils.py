import re
from typing import List, Dict


def remove_space(text: str) -> str:
    """Remove unnecessary spaces in text."""
    text = text.replace(" ,", ",").replace(", ", ",")
    text = text.replace(" (", "(").replace("( ", "(")
    text = text.replace(" )", ")").replace(") ", ")")
    return text

def normalize_triple(triple_str: str) -> List[str]:
    """Normalize a triple string into a list."""
    triple = triple_str.strip("()").split(",")
    return [part.strip() for part in triple]

def extract_triples(scene_graph: str) -> List[List[str]]:
    """Extract triples from scene graph string."""
    scene_graph = remove_space(scene_graph)
    raw_triples = scene_graph.split("),(")
    processed = []
    for triple in raw_triples:
        clean_triple = triple.strip().strip("()")
        if clean_triple:
            normalized = normalize_triple(clean_triple)
            if len(normalized) == 3:
                processed.append(normalized)
    return processed

def compare_triples(triple1: List[str], triple2: List[str]) -> bool:
    """Compare two triples regardless of order."""
    return sorted(triple1) == sorted(triple2)

def count_matching_triples(
    predictions: List[List[str]], ground_truths: List[List[str]]
) -> int:
    """Count matching triples between predictions and ground truths."""
    count = 0
    for pred in predictions:
        for gt in ground_truths:
            if compare_triples(pred, gt):
                count += 1
                break
    return count

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'"+', '"', text)
    text = text.replace('"', "'")
    text = re.sub(r"\n+", " ", text)
    return text

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

def merge_delete_insert_results(
    insert_res: str, delete_res: str
) -> List[List[str]]:
    """Merge delete and insert results into a single list of triples."""
    insert_res_list = extract_triples(insert_res)
    delete_res_list = extract_triples(delete_res)
    merged_res_list = insert_res_list + delete_res_list
    merged_res_list = list(set([tuple(triple) for triple in merged_res_list]))
    merged_res_list_str = " , ".join(
        [f"( {' , '.join(triple)} )" for triple in merged_res_list]
    )
    return merged_res_list, merged_res_list_str

def get_or_create_entity_index(entity_name, graph, entity_map):
        if entity_name not in entity_map:
            new_index = len(graph['entities'])
            graph['entities'].append({'head': entity_name, 'quantity': '', 'attributes': set()})
            entity_map[entity_name] = new_index
        else:
            new_index = entity_map[entity_name]

        return new_index

def graph_string_to_object(graph_text):
    graph = {'entities': [], 'relations': []}
    entity_map = {}  # Entity name to index mapping

    # Process each relation in the description
    relation_strs = graph_text.strip().split(') , ')
    for relation_str in relation_strs:
        relation_str = relation_str.strip().strip('()')
        parts = [part.strip() for part in relation_str.split(',')]

        if len(parts) != 3 and len(relation_strs) > 1:
            continue  # Skip malformed relations
        elif len(parts) != 3 and len(relation_strs) == 1:
            get_or_create_entity_index(parts[0], graph, entity_map)
        else:
            subject, relationship, object_ = parts

            subject_index = get_or_create_entity_index(subject, graph, entity_map)

            if relationship == 'is':
                if object_.isdigit():  # Quantity
                    graph['entities'][subject_index]['quantity'] = object_
                else:  # Attribute
                    graph['entities'][subject_index]['attributes'].add(object_)
            else:
                object_index = get_or_create_entity_index(object_, graph, entity_map)
                # Add relation
                graph['relations'].append({'subject': subject_index, 'relation': relationship, 'object': object_index})

    return graph

def collect_unique_captions_merge_refs_into_one_sentence(
    candidates: List[Dict[str, str]], refs: List[str]
) -> List[str]:
    """Collect unique captions from candidates and references."""
    caption_set = set()
    # merged_refs = [' '.join(ref_list) for ref_list in refs]
    # make sure each ref ends with a period during merging
    # for ref in refs:
    #     if ref and not ref.endwith('.'):
    #         ref += '.'
    for ref_list in refs:
        for ref in ref_list:
            if ref and not ref.endswith('.'):
                ref += '.'
    merged_refs = [' '.join(ref_list) for ref_list in refs]
    caption_set.update(merged_refs)
    return list(caption_set)

def collect_unique_captions(candidates, refs):
    """Collect unique captions from candidates and references."""
    caption_set = set(candidates)  # Add all candidates to the set
    for ref_list in refs:
        caption_set.update(ref_list)  # Add all elements from each list in refs
    return list(caption_set)