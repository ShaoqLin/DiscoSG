import os
import json
from typing import Dict, List, Any
import torch
import spacy
from transformers import AutoTokenizer
from disco_sg_camera.src.arch.T5ForDualTask import T5ForDualTask
from disco_sg_camera.src.factual_scene_graph.triple_utils import extract_triples

from tqdm import tqdm


class DualTaskSceneGraphParser:
    """
    DualTaskSceneGraphParser, functionalities:
    1. Insert triples into a scene graph based on a caption.
    2. Delete triples from a scene graph based on a caption.
    3. Delete triples from a scene graph based on a caption and then insert new triples.
    4. Insert triples into a scene graph based on a caption and then delete existing triples.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        lemmatize: bool = False,
        lowercase: bool = True,
    ):
        """
        Initialize the DualTaskSceneGraphParser.

        Args:
            model_path: Path to the model directory.
            device: Device to run the model on (e.g., "cuda" or "cpu").
            lemmatize: Whether to perform lemmatization on the input text.
            lowercase: Whether to convert the input text to lowercase.
        """
        self.device = device
        self.lemmatize = lemmatize
        self.lowercase = lowercase

        # load model
        self.model = T5ForDualTask.from_pretrained(model_path, device=device)
        self.model.eval()

        # get base model name
        base_model_name = None
        model_info_path = os.path.join(model_path, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                base_model_name = model_info.get(
                    "base_model_name", "google/flan-t5-base"
                )
        else:
            base_model_name = "google/flan-t5-base"  # 默认模型

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # load spacy if lemmatize is True
        if self.lemmatize:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")

    def _preprocess_text(self, text: str) -> str:
        """preprocess the input text"""
        if self.lowercase:
            text = text.lower()
        if self.lemmatize:
            doc = self.nlp(text)
            text = " ".join([token.lemma_ for token in doc])
        return text

    def _extract_triples_from_text(self, text: str) -> List[List[str]]:
        """extract triples from the input text"""
        return extract_triples(text)

    def _delete_triples(self, scene_graph: str, caption: str, **kwargs) -> str:
        """execute delete task
        Args:
            scene_graph: The scene graph to delete triples from.
            caption: The caption to use for deletion.
        """
        triples = self._extract_triples_from_text(scene_graph)
        if not triples:
            return scene_graph

        input_text = f"Delete Task:\nCaption: {caption}\nCandidate Graph: {scene_graph}"

        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        triple_spans = []
        for triple_str in [f"( {' , '.join(triple)} )" for triple in triples]:
            triple_start_char = input_text.find(triple_str)
            if triple_start_char == -1:
                continue
            triple_end_char = triple_start_char + len(triple_str) - 1

            token_start = None
            token_end = None
            for i, (start, end) in enumerate(offset_mapping):
                if start == end == 0:
                    continue
                if token_start is None and start <= triple_start_char < end:
                    token_start = i
                if end > 0 and start <= triple_end_char < end:
                    token_end = i
                if (
                    start > triple_end_char
                    and token_end is None
                    and token_start is not None
                ):
                    token_end = i - 1
                    break

            if token_start is not None and token_end is not None:
                triple_spans.append([token_start, token_end])

        if not triple_spans:
            return scene_graph

        max_triples = 128
        spans_tensor = torch.full((1, max_triples, 2), -1, dtype=torch.long).to(
            self.device
        )
        spans_mask = torch.zeros(1, max_triples, dtype=torch.bool).to(self.device)

        for i, (start, end) in enumerate(triple_spans):
            if i >= max_triples:
                break
            spans_tensor[0, i, 0] = start
            spans_tensor[0, i, 1] = end
            spans_mask[0, i] = True

        with torch.no_grad():
            outputs = self.model.forward_delete(
                input_ids=input_ids,
                attention_mask=attention_mask,
                triple_spans=spans_tensor,
                span_mask=spans_mask,
            )

        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

        kept_triples = []
        pred_idx = 0
        for i, _ in enumerate(triple_spans):
            if i >= len(predictions):
                kept_triples.append(triples[i])
                continue

            if predictions[pred_idx] == 0:  # KEEP
                kept_triples.append(triples[i])
            pred_idx += 1

        result = " , ".join([f"( {' , '.join(triple)} )" for triple in kept_triples])
        return result

    def _batch_delete_triples(
        self,
        scene_graphs: List[str],
        captions: List[str],
        max_input_len=2048,
        max_triples_num=128,
        skip_toolong=False,
        skip_len=-1,
        **kwargs,
    ) -> List[str]:
        if not scene_graphs:
            return []

        # assert when skip_toolong is True, skip_len should be > 0
        # if skip_toolong and skip_len <= 0:
        #     raise ValueError("When skip_toolong is True, skip_len should be > 0")

        batch_size = len(scene_graphs)
        results = [""] * batch_size

        all_triples = []
        valid_indices = []

        for i, scene_graph in enumerate(scene_graphs):
            triples = self._extract_triples_from_text(scene_graph)
            if triples:
                all_triples.append(triples)
                valid_indices.append(i)
            else:
                results[i] = scene_graph

        if not valid_indices:
            return results

        valid_scene_graphs = [scene_graphs[i] for i in valid_indices]
        valid_captions = [captions[i] for i in valid_indices]
        input_texts = [
            f"Delete Task:\nCaption: {caption}\nCandidate Graph: {graph}"
            for caption, graph in zip(valid_captions, valid_scene_graphs)
        ]

        # encode the input texts
        encodings = self.tokenizer(
            input_texts,
            max_length=max_input_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mappings = encodings.pop("offset_mapping").tolist()
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # set max_triples: we process 128 triples at most at a time
        max_triples = max_triples_num
        # get triple spans
        batch_triple_spans = []

        for batch_idx, (input_text, triples, offset_mapping) in enumerate(
            zip(input_texts, all_triples, offset_mappings)
        ):
            triple_spans = []
            for triple_str in [f"( {' , '.join(triple)} )" for triple in triples]:
                triple_start_char = input_text.find(triple_str)
                if triple_start_char == -1:
                    continue
                triple_end_char = triple_start_char + len(triple_str) - 1

                token_start = None
                token_end = None
                for i, (start, end) in enumerate(offset_mapping):
                    if start == end == 0:
                        continue
                    if token_start is None and start <= triple_start_char < end:
                        token_start = i
                    if end > 0 and start <= triple_end_char < end:
                        token_end = i
                    if (
                        start > triple_end_char
                        and token_end is None
                        and token_start is not None
                    ):
                        token_end = i - 1
                        break

                if token_start is not None and token_end is not None:
                    triple_spans.append([token_start, token_end])

            batch_triple_spans.append(triple_spans)

        # prepare spans tensor and mask for model input
        spans_tensor = torch.full(
            (len(valid_indices), max_triples, 2), -1, dtype=torch.long
        ).to(self.device)
        spans_mask = torch.zeros(len(valid_indices), max_triples, dtype=torch.bool).to(
            self.device
        )

        for batch_idx, triple_spans in enumerate(batch_triple_spans):
            for i, (start, end) in enumerate(triple_spans):
                if i >= max_triples:
                    break
                spans_tensor[batch_idx, i, 0] = start
                spans_tensor[batch_idx, i, 1] = end
                spans_mask[batch_idx, i] = True

        # forward pass
        with torch.no_grad():
            outputs = self.model.forward_delete(
                input_ids=input_ids,
                attention_mask=attention_mask,
                triple_spans=spans_tensor,
                span_mask=spans_mask,
            )

        # get logits
        logits = outputs["logits"]

        triple_counts = [min(len(spans), max_triples) for spans in batch_triple_spans]

        all_predictions = torch.argmax(logits, dim=1).cpu().numpy()

        start_idx = 0
        for batch_idx, count in enumerate(triple_counts):
            if count == 0:
                continue

            sample_predictions = all_predictions[start_idx : start_idx + count]
            start_idx += count

            kept_triples = []
            for i, pred in enumerate(sample_predictions):
                if i < len(all_triples[batch_idx]) and pred == 0:  # KEEP
                    kept_triples.append(all_triples[batch_idx][i])

            result = " , ".join(
                [f"( {' , '.join(triple)} )" for triple in kept_triples]
            )
            results[valid_indices[batch_idx]] = result

        # debug
        for result in results:
            if result == "" or len(result) < 5:
                print("Empty result")

        # skip too long encoded encodings
        # TODO: filter out too long encodings before encoding and ner deletion
        # if skip_toolong:
        #     # calculate the valid length of encoding, using sum(mask)
        #     # for i, scene_graph, encoding in enumerate(zip(scene_graphs, encodings["input_ids"])):
        #     for i, scene_graph, encoding, mask in zip(
        #         valid_indices,
        #         scene_graphs,
        #         encodings["input_ids"],
        #         encodings["attention_mask"],
        #     ):
        #         if sum(mask) > skip_len:
        #             results[i] = scene_graph

        return results

    def _batch_delete_triples_skip_len(
        self,
        scene_graphs: List[str],
        captions: List[str],
        max_input_len=2048,
        max_triples_num=128,
        skip_toolong=False,
        skip_len=-1,
        **kwargs,
    ) -> List[str]:
        if not scene_graphs:
            return []

        # assert when skip_toolong is True, skip_len should be > 0
        if skip_toolong and skip_len <= 0:
            raise ValueError("When skip_toolong is True, skip_len should be > 0")

        batch_size = len(scene_graphs)
        results = [""] * batch_size

        all_triples = []
        valid_indices = []

        for i, scene_graph in enumerate(scene_graphs):
            triples = self._extract_triples_from_text(scene_graph)
            if triples:
                all_triples.append(triples)
                valid_indices.append(i)
            else:
                results[i] = scene_graph

        if not valid_indices:
            return results

        valid_scene_graphs = [scene_graphs[i] for i in valid_indices]
        valid_captions = [captions[i] for i in valid_indices]
        input_texts = [
            f"Delete Task:\nCaption: {caption}\nCandidate Graph: {graph}"
            for caption, graph in zip(valid_captions, valid_scene_graphs)
        ]

        # encode the input texts
        encoding_for_selection = self.tokenizer(
            input_texts,
            max_length=max_input_len * 4,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        encodings = self.tokenizer(
            input_texts,
            max_length=max_input_len * 4,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        offset_mappings = encodings.pop("offset_mapping").tolist()
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        attention_mask_for_selection = encoding_for_selection["attention_mask"].to(
            self.device
        )

        if skip_toolong:
            skip_indices = []
            for batch_idx, mask in enumerate(attention_mask_for_selection):
                # print(sum(mask).item())
                if sum(mask).item() > skip_len:
                    original_idx = valid_indices[batch_idx]
                    results[original_idx] = scene_graphs[original_idx]
                    skip_indices.append(batch_idx)

            if skip_indices:
                keep_indices = [
                    i for i in range(len(valid_indices)) if i not in skip_indices
                ]
                if not keep_indices:
                    return results

                input_ids = input_ids[keep_indices]
                attention_mask = attention_mask[keep_indices]
                offset_mappings = [offset_mappings[i] for i in keep_indices]
                input_texts = [input_texts[i] for i in keep_indices]
                all_triples = [all_triples[i] for i in keep_indices]
                valid_indices = [valid_indices[i] for i in keep_indices]

        # set max_triples: we process max_triples_num triples at most at a time
        max_triples = max_triples_num
        # get triple spans
        batch_triple_spans = []

        for batch_idx, (input_text, triples, offset_mapping) in enumerate(
            zip(input_texts, all_triples, offset_mappings)
        ):
            triple_spans = []
            for triple_str in [f"( {' , '.join(triple)} )" for triple in triples]:
                triple_start_char = input_text.find(triple_str)
                if triple_start_char == -1:
                    continue
                triple_end_char = triple_start_char + len(triple_str) - 1

                token_start = None
                token_end = None
                for i, (start, end) in enumerate(offset_mapping):
                    if start == end == 0:
                        continue
                    if token_start is None and start <= triple_start_char < end:
                        token_start = i
                    if end > 0 and start <= triple_end_char < end:
                        token_end = i
                    if (
                        start > triple_end_char
                        and token_end is None
                        and token_start is not None
                    ):
                        token_end = i - 1
                        break

                if token_start is not None and token_end is not None:
                    triple_spans.append([token_start, token_end])

            batch_triple_spans.append(triple_spans)

        # prepare spans tensor and mask for model input
        spans_tensor = torch.full(
            (len(valid_indices), max_triples, 2), -1, dtype=torch.long
        ).to(self.device)
        spans_mask = torch.zeros(len(valid_indices), max_triples, dtype=torch.bool).to(
            self.device
        )

        for batch_idx, triple_spans in enumerate(batch_triple_spans):
            for i, (start, end) in enumerate(triple_spans):
                if i >= max_triples:
                    break
                spans_tensor[batch_idx, i, 0] = start
                spans_tensor[batch_idx, i, 1] = end
                spans_mask[batch_idx, i] = True

        # forward pass
        with torch.no_grad():
            outputs = self.model.forward_delete(
                input_ids=input_ids,
                attention_mask=attention_mask,
                triple_spans=spans_tensor,
                span_mask=spans_mask,
            )

        # get logits
        logits = outputs["logits"]

        triple_counts = [min(len(spans), max_triples) for spans in batch_triple_spans]

        # pred label for each triple, 0 for KEEP, 1 for DELETE
        all_predictions = torch.argmax(logits, dim=1).cpu().numpy()

        # allocate the predictions to each sample
        start_idx = 0
        for batch_idx, count in enumerate(triple_counts):
            if count == 0:
                continue

            sample_predictions = all_predictions[start_idx : start_idx + count]
            start_idx += count

            kept_triples = []
            for i, pred in enumerate(sample_predictions):
                if i < len(all_triples[batch_idx]) and pred == 0:  # KEEP
                    kept_triples.append(all_triples[batch_idx][i])

            result = " , ".join(
                [f"( {' , '.join(triple)} )" for triple in kept_triples]
            )
            results[valid_indices[batch_idx]] = result

        # debug
        # for i, result in enumerate(results):
        #     if result == "" or len(result) < 5:
        #         print("\nEmpty result, replace with original scene graph\n")
        #         results[i] = scene_graphs[valid_indices[i]] # TODO：check

        return results

    def _batch_delete_triples_maskBIO(
        self, scene_graphs: List[str], captions: List[str], **kwargs
    ) -> List[str]:
        if not scene_graphs:
            return []

        batch_size = len(scene_graphs)
        results = [""] * batch_size

        all_triples = []
        valid_indices = []

        for i, scene_graph in enumerate(scene_graphs):
            triples = self._extract_triples_from_text(scene_graph)
            if triples:
                all_triples.append(triples)
                valid_indices.append(i)
            else:
                results[i] = scene_graph

        if not valid_indices:
            return results

        valid_scene_graphs = [scene_graphs[i] for i in valid_indices]
        valid_captions = [captions[i] for i in valid_indices]
        input_texts = [
            f"Delete Task:\nCaption: {caption}\nCandidate Graph: {graph}"
            for caption, graph in zip(valid_captions, valid_scene_graphs)
        ]

        encodings = self.tokenizer(
            input_texts,
            max_length=512,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        offset_mapping = encodings["offset_mapping"].to(self.device)

        max_triples = 128
        valid_batch_size = len(valid_indices)

        spans_tensor = torch.full(
            (valid_batch_size, max_triples, 2), -1, dtype=torch.long, device=self.device
        )
        spans_mask = torch.zeros(
            (valid_batch_size, max_triples), dtype=torch.bool, device=self.device
        )

        for batch_idx, (input_text, triples) in enumerate(
            zip(input_texts, all_triples)
        ):
            if not triples:
                continue

            triple_strings = [f"( {' , '.join(triple)} )" for triple in triples]

            for i, triple_str in enumerate(triple_strings):
                if i >= max_triples:
                    break

                triple_start_char = input_text.find(triple_str)
                if triple_start_char == -1:
                    continue
                triple_end_char = triple_start_char + len(triple_str) - 1

                sample_offset = offset_mapping[batch_idx]

                char_starts = sample_offset[:, 0]
                char_ends = sample_offset[:, 1]

                start_mask = (
                    (char_starts <= triple_start_char)
                    & (triple_start_char < char_ends)
                    & (char_ends > 0)
                )
                if not torch.any(start_mask):
                    continue
                token_start = torch.nonzero(start_mask, as_tuple=True)[0][0].item()

                end_mask = (
                    (char_starts <= triple_end_char)
                    & (triple_end_char < char_ends)
                    & (char_ends > 0)
                )
                if not torch.any(end_mask):
                    end_mask = (
                        (char_ends > 0)
                        & (char_starts > triple_start_char)
                        & (char_starts <= triple_end_char)
                    )
                    if not torch.any(end_mask):
                        continue
                    token_end = torch.nonzero(end_mask, as_tuple=True)[0][-1].item()
                else:
                    token_end = torch.nonzero(end_mask, as_tuple=True)[0][0].item()

                spans_tensor[batch_idx, i, 0] = token_start
                spans_tensor[batch_idx, i, 1] = token_end
                spans_mask[batch_idx, i] = True

        if not torch.any(spans_mask):
            return results

        with torch.no_grad():
            outputs = self.model.forward_delete_maskBIO(
                input_ids=input_ids,
                attention_mask=attention_mask,
                triple_spans=spans_tensor,
                span_mask=spans_mask,
            )

        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1).cpu()

        batch_indices = outputs.get("batch_indices", None)

        if batch_indices is not None:
            batch_indices = batch_indices.cpu()

            batch_predictions = [[] for _ in range(valid_batch_size)]

            for i, batch_idx in enumerate(batch_indices):
                batch_idx = batch_idx.item()
                if batch_idx < valid_batch_size:
                    batch_predictions[batch_idx].append(predictions[i].item())

            for batch_idx, (preds, triples) in enumerate(
                zip(batch_predictions, all_triples)
            ):
                kept_triples = [
                    triples[i]
                    for i, pred in enumerate(preds)
                    if i < len(triples) and pred == 0
                ]
                result = " , ".join(
                    [f"( {' , '.join(triple)} )" for triple in kept_triples]
                )
                results[valid_indices[batch_idx]] = result
        else:
            triple_counts = spans_mask.sum(dim=1).cpu().tolist()

            start_idx = 0
            for batch_idx, count in enumerate(triple_counts):
                count = int(count)
                if count == 0:
                    continue

                sample_predictions = predictions[start_idx : start_idx + count].tolist()
                start_idx += count

                kept_triples = []
                for i, pred in enumerate(sample_predictions):
                    if i < len(all_triples[batch_idx]) and pred == 0:  # KEEP
                        kept_triples.append(all_triples[batch_idx][i])

                result = " , ".join(
                    [f"( {' , '.join(triple)} )" for triple in kept_triples]
                )
                results[valid_indices[batch_idx]] = result

        return results

    def _insert_triples(self, scene_graph: str, caption: str, **kwargs) -> str:
        """执行插入任务"""
        input_text = f"Insert Task:\nCaption: {caption}\nCorrupted Graph: {scene_graph}"

        input_encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                max_length=512,
                num_beams=3,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
            )

        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        if generated_text.strip():
            original_triples = self._extract_triples_from_text(scene_graph)
            generated_triples = self._extract_triples_from_text(generated_text)

            all_triples = original_triples.copy()
            for triple in generated_triples:
                if triple not in all_triples:
                    all_triples.append(triple)

            result = " , ".join([f"( {' , '.join(triple)} )" for triple in all_triples])
            return result

        return scene_graph

    def _batch_insert_triples(
        self,
        scene_graphs: List[str],
        captions: List[str],
        max_input_len=2048,
        max_output_len=512,
        skip_toolong=False,
        skip_len=-1,
        **kwargs,
    ) -> List[str]:
        if not scene_graphs:
            return []

        batch_size = len(scene_graphs)
        results = [""] * batch_size

        valid_indices = []
        valid_scene_graphs = []
        valid_captions = []

        for i, (scene_graph, caption) in enumerate(zip(scene_graphs, captions)):
            if scene_graph.strip():
                valid_indices.append(i)
                valid_scene_graphs.append(scene_graph)
                valid_captions.append(caption)
            else:
                results[i] = ""

        if not valid_indices:
            return results

        input_texts = [
            f"Insert Task:\nCaption: {caption}\nCorrupted Graph: {scene_graph}"
            for caption, scene_graph in zip(valid_captions, valid_scene_graphs)
        ]

        input_encoding = self.tokenizer(
            input_texts,
            max_length=max_input_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                max_length=max_output_len,
                num_beams=3,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
            )

        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        for i, (scene_graph, generated_text, idx) in enumerate(
            zip(valid_scene_graphs, generated_texts, valid_indices)
        ):
            if generated_text.strip():
                original_triples = self._extract_triples_from_text(scene_graph)
                generated_triples = self._extract_triples_from_text(generated_text)

                all_triples = original_triples.copy()
                for triple in generated_triples:
                    if triple not in all_triples:
                        all_triples.append(triple)

                result = " , ".join(
                    [f"( {' , '.join(triple)} )" for triple in all_triples]
                )
                results[idx] = result
            else:
                results[idx] = scene_graph

        # skip too long encoded encodings
        # TODO: filter out too long encodings before encoding and ner deletion
        if skip_toolong:
            # calculate the valid length of encoding, using sum(mask)
            # for i, scene_graph, encoding in enumerate(zip(scene_graphs, encodings["input_ids"])):
            for i, scene_graph, encoding, mask in zip(
                valid_indices,
                scene_graphs,
                input_encoding["input_ids"],
                input_encoding["attention_mask"],
            ):
                if sum(mask) > skip_len:
                    results[i] = scene_graph

        return results

    def _batch_insert_triples_skip_len(
        self,
        scene_graphs: List[str],
        captions: List[str],
        max_input_len=2048,
        max_output_len=512,
        skip_toolong=False,
        skip_len=-1,
        num_beams=1,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=1.0,
        **kwargs,
    ) -> List[str]:
        """Batch execution of insert tasks"""
        if not scene_graphs:
            return []

        # Ensure skip_len is valid when skip_toolong is True
        if skip_toolong and skip_len <= 0:
            raise ValueError("When skip_toolong is True, skip_len should be > 0")

        batch_size = len(scene_graphs)
        results = [""] * batch_size

        # Filter out empty scene graphs
        valid_indices = []
        valid_scene_graphs = []
        valid_captions = []

        for i, (scene_graph, caption) in enumerate(zip(scene_graphs, captions)):
            if scene_graph.strip():
                valid_indices.append(i)
                valid_scene_graphs.append(scene_graph)
                valid_captions.append(caption)
            else:
                results[i] = ""  # Return empty string for empty scene graphs

        if not valid_indices:
            return results

        # Prepare batch inputs
        input_texts = [
            f"Insert Task:\nCaption: {caption}\nCorrupted Graph: {scene_graph}"
            for caption, scene_graph in zip(valid_captions, valid_scene_graphs)
        ]

        # Encode input texts with max_length + 1 for select the over length samples
        input_encoding_for_selection = self.tokenizer(
            input_texts,
            max_length=max_input_len * 4,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        # Encode input texts
        input_encoding = self.tokenizer(
            input_texts,
            max_length=max_input_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Check encoding length and skip too long samples before model processing
        if skip_toolong:
            skip_indices = []
            for batch_idx, mask in enumerate(
                input_encoding_for_selection["attention_mask"]
            ):
                if sum(mask).item() > skip_len:
                    original_idx = valid_indices[batch_idx]
                    results[original_idx] = scene_graphs[original_idx]
                    skip_indices.append(batch_idx)
                    print(
                        f"Skipping sample {original_idx} due to length: {sum(mask).item()}"
                    )

            # Remove skipped samples from processing data
            if skip_indices:
                # Create indices for samples to keep
                keep_indices = [
                    i for i in range(len(valid_indices)) if i not in skip_indices
                ]
                if not keep_indices:  # If all samples are skipped
                    return results

                # Update all relevant data structures
                input_ids = input_encoding["input_ids"][keep_indices]
                attention_mask = input_encoding["attention_mask"][keep_indices]
                input_encoding = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                # Update valid_indices and other related lists
                valid_scene_graphs = [valid_scene_graphs[i] for i in keep_indices]
                valid_indices = [valid_indices[i] for i in keep_indices]

        # Return early if no valid samples remain
        if len(valid_indices) == 0:
            return results

        # Generate triples in batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                max_length=max_output_len,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=1,
            )
            # generated_ids = self.model.generate(
            #     input_ids=input_encoding["input_ids"],
            #     attention_mask=input_encoding["attention_mask"],
            #     max_length=max_output_len,
            #     num_beams=3,
            #     temperature=0.7,
            #     top_k=50,
            #     top_p=0.95,
            #     do_sample=True,
            #     num_return_sequences=1,
            # )

        # Decode generated text
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        # Process each scene graph
        for i, (scene_graph, generated_text, idx) in enumerate(
            zip(valid_scene_graphs, generated_texts, valid_indices)
        ):
            # Add generated triples to original scene graph if not empty
            if generated_text.strip() and len(generated_text.strip()) > 5:
                # Extract original and generated triples
                original_triples = self._extract_triples_from_text(scene_graph)
                generated_triples = self._extract_triples_from_text(generated_text)

                # Merge triples (remove duplicates)
                all_triples = original_triples.copy()
                for triple in generated_triples:
                    if triple not in all_triples:
                        all_triples.append(triple)

                # Convert merged triples back to string format
                result = " , ".join(
                    [f"( {' , '.join(triple)} )" for triple in all_triples]
                )
                results[idx] = result
            else:
                print(
                    f"Generated text is empty or too short: {generated_text.strip()}, replacing with original scene graph"
                )
                results[idx] = scene_graph

        return results

    def _insert_iter(self, desc, graph_to_fix, batch_size):
        results = {}
        for i in range(0, len(desc), batch_size):
            batch_desc = desc[i : i + batch_size]
            for _, d in enumerate(batch_desc):
                if graph_to_fix and d in graph_to_fix:
                    scene_graph = graph_to_fix[d]
                    results[d] = self._insert_triples(
                        scene_graph=scene_graph, caption=d
                    )
                else:
                    results[d] = ""
        return results

    def _delete_iter(self, desc, graph_to_fix, batch_size):
        results = {}
        for i in range(0, len(desc), batch_size):
            batch_desc = desc[i : i + batch_size]
            for _, d in enumerate(batch_desc):
                if graph_to_fix and d in graph_to_fix:
                    scene_graph = graph_to_fix[d]
                    results[d] = self._batch_delete_triples(
                        scene_graph=scene_graph, caption=d
                    )
                else:
                    results[d] = ""
        return results

    def parse(
        self,
        descriptions: List[str],
        graph_to_fix=None,
        batch_size: int = 32,
        task: str = "insert_delete",
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_triples_num: int = 128,
        skip_toolong=False,
        skip_len=-1,
        num_beams: int = 1,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:

        results = {}

        if task == "delete":
            results = {}
            for i in range(0, len(descriptions), batch_size):
                batch_descriptions = descriptions[
                    i : i + batch_size
                ]  # TODO: need batchify
                for j, desc in enumerate(batch_descriptions):
                    if graph_to_fix and desc in graph_to_fix:
                        scene_graph = graph_to_fix[desc]
                        results[desc] = self._batch_delete_triples(
                            scene_graph=scene_graph,
                            caption=desc,
                            max_input_len=max_input_len * 4,
                            max_triples_num=max_triples_num,
                        )
                    else:
                        results[desc] = ""
            return results

        elif task == "insert":  # TODO: check here
            results = {}
            for i in range(0, len(descriptions), batch_size):
                batch_descriptions = descriptions[i : i + batch_size]
                for j, desc in enumerate(batch_descriptions):
                    if graph_to_fix and desc in graph_to_fix:
                        scene_graph = graph_to_fix[desc]
                        results[desc] = self._batch_insert_triples_skip_len(
                            scene_graph=scene_graph,
                            caption=desc,
                            skip_toolong=skip_toolong,
                            skip_len=skip_len,
                            max_input_len=max_input_len,
                            max_output_len=max_output_len,
                            num_beams=num_beams if not do_sample else 1,
                            do_sample=do_sample,
                            top_k=top_k if do_sample else 50,
                            top_p=top_p if do_sample else 1.0,
                            temperature=temperature if do_sample else 1.0,
                        )
                    else:
                        results[desc] = ""
            return results

        elif task == "insert_delete":
            delete_results = {}
            for i in tqdm(range(0, len(descriptions), batch_size), ncols=88):
                batch_descriptions = descriptions[i : i + batch_size]
                batch_scene_graphs = []
                batch_captions = []

                for desc in batch_descriptions:
                    if graph_to_fix and desc in graph_to_fix:
                        batch_scene_graphs.append(graph_to_fix[desc])
                        batch_captions.append(desc)
                    else:
                        batch_scene_graphs.append("")
                        batch_captions.append(desc)  # TODO: check here

                batch_delete_results = self._batch_delete_triples(
                    scene_graphs=batch_scene_graphs,
                    captions=batch_captions,
                    max_input_len=max_input_len,
                    max_triples_num=max_triples_num,
                )
                for batch_idx, desc in enumerate(batch_captions):
                    delete_results[desc] = batch_delete_results[batch_idx]

            insert_results = {}
            for i in tqdm(range(0, len(descriptions), batch_size), ncols=88):
                batch_descriptions = descriptions[i : i + batch_size]
                batch_scene_graphs = []
                batch_captions = []
                for desc in batch_descriptions:
                    if graph_to_fix and desc in graph_to_fix:
                        batch_scene_graphs.append(graph_to_fix[desc])
                        batch_captions.append(desc)
                    else:
                        batch_scene_graphs.append("")
                        batch_captions.append(desc)
                batch_insert_results = self._batch_insert_triples_skip_len(
                    scene_graphs=batch_scene_graphs,
                    captions=batch_captions,
                    skip_toolong=skip_toolong,
                    skip_len=skip_len,
                    max_input_len=max_input_len,
                    max_output_len=max_output_len,
                    num_beams=num_beams if not do_sample else 1,
                    do_sample=do_sample,
                    top_k=top_k if do_sample else 50,
                    top_p=top_p if do_sample else 1.0,
                    temperature=temperature if do_sample else 1.0,
                )
                for batch_idx, desc in enumerate(batch_captions):
                    insert_results[desc] = batch_insert_results[batch_idx]

            return {"delete": delete_results, "insert": insert_results}

        elif task == "delete_before_insert":
            # insert based on delete results
            delete_results = {}
            for i in tqdm(range(0, len(descriptions), batch_size), ncols=88):
                batch_descriptions = descriptions[i : i + batch_size]
                batch_scene_graphs = []
                batch_captions = []
                for desc in batch_descriptions:
                    if graph_to_fix and desc in graph_to_fix:
                        batch_scene_graphs.append(graph_to_fix[desc])
                        batch_captions.append(desc)
                    else:
                        batch_scene_graphs.append("")
                        batch_captions.append(desc)
                batch_delete_results = self._batch_delete_triples(
                    scene_graphs=batch_scene_graphs,
                    captions=batch_captions,
                    max_input_len=max_input_len,
                    max_triples_num=max_triples_num,
                )
                for batch_idx, desc in enumerate(batch_captions):
                    delete_results[desc] = batch_delete_results[batch_idx]

            insert_results = {}

            for i in tqdm(range(0, len(descriptions), batch_size), ncols=88):
                batch_descriptions = descriptions[i : i + batch_size]
                batch_scene_graphs = []
                batch_captions = []
                for desc in batch_descriptions:
                    if delete_results and desc in delete_results:
                        batch_scene_graphs.append(delete_results[desc])
                        batch_captions.append(desc)
                    else:
                        batch_scene_graphs.append("")
                        batch_captions.append(desc)
                batch_insert_results = self._batch_insert_triples_skip_len(
                    scene_graphs=batch_scene_graphs,
                    captions=batch_captions,
                    skip_toolong=skip_toolong,
                    skip_len=skip_len,
                    max_input_len=max_input_len,
                    max_output_len=max_output_len,
                    num_beams=num_beams if not do_sample else 1,
                    do_sample=do_sample,
                    top_k=top_k if do_sample else 50,
                    top_p=top_p if do_sample else 1.0,
                    temperature=temperature if do_sample else 1.0,
                )
                for batch_idx, desc in enumerate(batch_captions):
                    insert_results[desc] = batch_insert_results[batch_idx]
            return {"delete": delete_results, "insert": insert_results}

        else:
            raise ValueError(f"Unsupported task: {task}")
