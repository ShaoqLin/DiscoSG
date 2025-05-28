import torch
from nltk import WordNetLemmatizer
from tqdm import tqdm
from itertools import islice
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel

from src.factual_scene_graph.utils import (
    space_out_symbols_in_graph,
    clean_graph_string,
    remove_factual_chars,
)


class SceneGraphParser:
    def __init__(
        self,
        checkpoint_path,
        device="cuda:0",
        lemmatize=False,
        lowercase=False,
        decoder_only=False,
    ):
        self.device = device
        if decoder_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, padding_side="left"
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device).eval()
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            self.model = (
                AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device).eval()
            )
        self.lemmatize = lemmatize
        self.lowercase = lowercase

        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def _process_text(self, text):
        """
        Perform text processing: lemmatization and optionally converting to lowercase.

        :param text: A string containing the text to be processed.
        :return: Processed text as a string.
        """

        if self.lemmatize:
            # Lemmatize each word in the text
            tokens = text.split(" ")
            text = " ".join([self.lemmatizer.lemmatize(token) for token in tokens])

        if self.lowercase:
            text = text.lower()

        return text

    def chunk_list(self, data, batch_size):
        it = iter(data)
        for chunk in iter(lambda: list(islice(it, batch_size)), []):
            yield chunk

    def parse(
        self,
        descriptions,
        max_input_len=64,
        max_output_len=128,
        num_beams=5,
        return_text=False,
        filter_factual_chars=False,
        batch_size=32,
        sub_sentence=False,
        repetition_penalty=1.0,
        num_return_sequences=1,
        logits_processor=None,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        do_sample=False,
    ):
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # Apply text processing (lemmatization and lowercase) to each description
        processed_descriptions = [self._process_text(desc) for desc in descriptions]

        # Initialize results list
        all_formatted_texts = []

        # Process descriptions in batches
        if not sub_sentence:
            for i in tqdm(range(0, len(processed_descriptions), batch_size)):
                batch_descriptions = processed_descriptions[i : i + batch_size]
                prompt_texts = [
                    "Generate Scene Graph: " + desc.strip()
                    for desc in batch_descriptions
                ]
                with torch.no_grad():
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        max_length=max_input_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    tokens = encoded_inputs["input_ids"].to(self.device)
                    attention_masks = encoded_inputs["attention_mask"].to(self.device)

                    early_stopping = num_beams > 1

                    generated_ids = self.model.generate(
                        tokens,
                        attention_mask=attention_masks,
                        use_cache=True,
                        decoder_start_token_id=self.tokenizer.pad_token_id,
                        num_beams=num_beams,
                        max_length=max_output_len,
                        early_stopping=early_stopping,
                        logits_processor=logits_processor,
                        # num_return_sequences=1,
                        # repetition_penalty=repetition_penalty,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )

                    # Decoding the output
                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                    if filter_factual_chars:
                        generated_texts = [
                            remove_factual_chars(text) for text in generated_texts
                        ]

                    formatted_texts = [
                        clean_graph_string(
                            space_out_symbols_in_graph(
                                text.replace("Generate Scene Graph:", "").strip()
                            )
                        )
                        for text in generated_texts
                    ]

                    # formatted_texts = generated_texts

                    all_formatted_texts.extend(formatted_texts)
        else:
            for i in tqdm(range(0, len(processed_descriptions), batch_size)):
                batch_descriptions = processed_descriptions[i : i + batch_size]

                all_sentences = []
                for description in batch_descriptions:
                    sentences = [
                        s.strip() + "." if not s.strip().endswith(".") else s.strip()
                        for s in description.split(". ")
                        if s
                    ]
                    all_sentences.extend(sentences)

                prompt_texts = [
                    "Generate Scene Graph: " + desc.strip() for desc in all_sentences
                ]

                with torch.no_grad():
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        max_length=max_input_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    tokens = encoded_inputs["input_ids"].to(self.device)
                    attention_masks = encoded_inputs["attention_mask"].to(self.device)

                    early_stopping = num_beams > 1

                    generated_ids = self.model.generate(
                        tokens,
                        attention_mask=attention_masks,
                        use_cache=True,
                        decoder_start_token_id=self.tokenizer.pad_token_id,
                        num_beams=num_beams,
                        max_length=max_output_len,
                        early_stopping=early_stopping,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                    )

                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                    if filter_factual_chars:
                        generated_texts = [
                            remove_factual_chars(text) for text in generated_texts
                        ]

                    formatted_texts = [
                        clean_graph_string(
                            space_out_symbols_in_graph(
                                text.replace("Generate Scene Graph:", "").strip()
                            )
                        )
                        for text in generated_texts
                    ]

                    # formatted_texts = generated_texts

                    start_idx = 0
                    for description in batch_descriptions:
                        # num_sentences = len(description.split(". "))
                        num_sentences = len([
                            s.strip() + "." if not s.strip().endswith(".") else s.strip()
                            for s in description.split(". ")
                            if s
                        ])
                        assert start_idx + num_sentences <= len(formatted_texts), "Index out of range"
                        batch_result = " , ".join(
                            formatted_texts[start_idx : start_idx + num_sentences]
                        )
                        all_formatted_texts.append(batch_result)
                        start_idx += num_sentences

        if return_text:
            return all_formatted_texts
        else:
            return [self.graph_string_to_object(text) for text in all_formatted_texts]

    def graph_string_to_object(self, graph_text):
        graph = {"entities": [], "relations": []}
        entity_map = {}  # Entity name to index mapping

        # Process each relation in the description
        relation_strs = graph_text.strip().split(") ,")
        for relation_str in relation_strs:
            relation_str = relation_str.strip().strip("()")
            parts = [part.strip() for part in relation_str.split(",")]

            if len(parts) != 3 and len(relation_strs) > 1:
                continue  # Skip malformed relations
            elif len(parts) != 3 and len(relation_strs) == 1:
                self._get_or_create_entity_index(parts[0], graph, entity_map)
            else:
                subject, relationship, object_ = parts

                subject_index = self._get_or_create_entity_index(
                    subject, graph, entity_map
                )

                if relationship == "is":
                    if object_.isdigit():  # Quantity
                        graph["entities"][subject_index]["quantity"] = object_
                    else:  # Attribute
                        graph["entities"][subject_index]["attributes"].add(object_)
                else:
                    object_index = self._get_or_create_entity_index(
                        object_, graph, entity_map
                    )
                    # Add relation
                    graph["relations"].append(
                        {
                            "subject": subject_index,
                            "relation": relationship,
                            "object": object_index,
                        }
                    )

        return graph

    def _get_or_create_entity_index(self, entity_name, graph, entity_map):
        if entity_name not in entity_map:
            new_index = len(graph["entities"])
            graph["entities"].append(
                {"head": entity_name, "quantity": "", "attributes": set()}
            )
            entity_map[entity_name] = new_index
        else:
            new_index = entity_map[entity_name]

        return new_index


class SceneGraphParser_LoRA(SceneGraphParser):
    def __init__(
        self,
        base_model_path_or_name: str,
        peft_model_path: str,
        device: str = "cuda:0",
        lemmatize=False,
        lowercase=False,
        decoder_only=False,
        max_length: int = 256,
    ):
        self.device = device
        print(self.device)
        self.max_length = max_length
        if decoder_only:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path_or_name, padding_side="left"
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(base_model_path_or_name)
                .to(device)
                .eval()
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path_or_name)
            self.model = (
                AutoModelForSeq2SeqLM.from_pretrained(base_model_path_or_name)
                .to(device)
                .eval()
            )
        print(f"Loading PeftModel from {peft_model_path}")
        self.model = PeftModel.from_pretrained(self.model, peft_model_path).to(device)
        self.model.eval()

        self.lemmatize = lemmatize
        self.lowercase = lowercase

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()


    def parse(
        self,
        descriptions,
        max_input_len=64,
        max_output_len=128,
        num_beams=5,
        return_text=False,
        filter_factual_chars=False,
        batch_size=32,
        sub_sentence=False,
        repetition_penalty=1.0,
        num_return_sequences=1,
        logits_processor=None,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        do_sample=False,
    ):
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # Apply text processing (lemmatization and lowercase) to each description
        processed_descriptions = [self._process_text(desc) for desc in descriptions]

        # Initialize results list
        all_formatted_texts = []

        # Process descriptions in batches
        if not sub_sentence:
            for i in tqdm(range(0, len(processed_descriptions), batch_size)):
                batch_descriptions = processed_descriptions[i : i + batch_size]
                prompt_texts = [
                    "Generate Scene Graph: " + desc.strip()
                    for desc in batch_descriptions
                ]
                with torch.no_grad():
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        max_length=max_input_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    generated_ids = self.model.generate(
                        **encoded_inputs,
                        use_cache=True,
                        decoder_start_token_id=self.tokenizer.pad_token_id,
                        # num_beams=num_beams,
                        max_length=max_output_len,
                        # num_return_sequences=1,
                        repetition_penalty=repetition_penalty,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )

                    # Decoding the output
                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                    if filter_factual_chars:
                        generated_texts = [
                            remove_factual_chars(text) for text in generated_texts
                        ]

                    formatted_texts = [
                        clean_graph_string(
                            space_out_symbols_in_graph(
                                text.replace("Generate Scene Graph:", "").strip()
                            )
                        )
                        for text in generated_texts
                    ]

                    # formatted_texts = generated_texts

                    all_formatted_texts.extend(formatted_texts)
        else:
            for i in tqdm(range(0, len(processed_descriptions), batch_size)):
                batch_descriptions = processed_descriptions[i : i + batch_size]

                all_sentences = []
                for description in batch_descriptions:
                    sentences = [
                        s.strip() + "." if not s.strip().endswith(".") else s.strip()
                        for s in description.split(". ")
                        if s
                    ]
                    all_sentences.extend(sentences)

                prompt_texts = [
                    "Generate Scene Graph: " + desc.strip() for desc in all_sentences
                ]

                with torch.no_grad():
                    encoded_inputs = self.tokenizer(
                        prompt_texts,
                        max_length=max_input_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    generated_ids = self.model.generate(
                        **encoded_inputs,
                        use_cache=True,
                        decoder_start_token_id=self.tokenizer.pad_token_id,
                        # num_beams=num_beams,
                        max_length=max_output_len,
                        # num_return_sequences=1,
                        repetition_penalty=repetition_penalty,
                        # no_repeat_ngram_size=no_repeat_ngram_size,
                    )

                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )

                    if filter_factual_chars:
                        generated_texts = [
                            remove_factual_chars(text) for text in generated_texts
                        ]

                    formatted_texts = [
                        clean_graph_string(
                            space_out_symbols_in_graph(
                                text.replace("Generate Scene Graph:", "").strip()
                            )
                        )
                        for text in generated_texts
                    ]

                    # formatted_texts = generated_texts

                    start_idx = 0
                    for description in batch_descriptions:
                        # num_sentences = len(description.split(". "))
                        num_sentences = len([
                            s.strip() + "." if not s.strip().endswith(".") else s.strip()
                            for s in description.split(". ")
                            if s
                        ])
                        assert start_idx + num_sentences <= len(formatted_texts), "Index out of range"
                        batch_result = " , ".join(
                            formatted_texts[start_idx : start_idx + num_sentences]
                        )
                        all_formatted_texts.append(batch_result)
                        start_idx += num_sentences

        if return_text:
            return all_formatted_texts
        else:
            return [self.graph_string_to_object(text) for text in all_formatted_texts]
