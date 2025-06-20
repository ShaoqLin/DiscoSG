# DiscoSG
Official repo for DiscoSG: Towards Discourse-Level Text Scene Graph Parsing through Iterative Graph Refinement

## Dataset Statistics and Analysis

The **DiscoSG** dataset is composed of in DiscoSG_dataset folder:

- **Human-annotated instances**: 400 total (300 train, 100 test)  
- **Synthesized instances**: 8,430 train  
- **Total training examples**: 8,730 (300 manual + 8,430 synthetic)  

Each instance in DiscoSG-DS is substantially richer than existing benchmarks, as shown below:

| Dataset     | # Inst.    | Avg Len | Avg Trp | Avg Obj | Avg Rel | Total Trp  |
|-------------|-----------:|--------:|--------:|--------:|--------:|-----------:|
| **VG**      | 2,966,195  |   5.34   |   1.53   |    –     |    –     | 4,533,271  |
| **FACTUAL** |   40,369   |   6.08   |   1.76   |    –     |    –     |    71,124  | 
|**DiscoSG**&nbsp;&nbsp;Human     |      400    | 181.15   |  20.49  |  10.11  |   6.54   |   8,195    |
|**DiscoSG**  &nbsp;&nbsp;Synthetic |    8,430    | 163.07   |  19.41  |  10.06  |   6.39   | 163,640    |

- **Avg Len**: average number of tokens per instance  
- **Avg Trp/Obj/Rel**: average triples, objects, and relations per graph  
- **Total Trp**: sum of all triples across the dataset

## Quick Start Guide

### File Configuration

1. **Dataset Path Configuration**
   - In `detailcap_discosg_mr.py` at line 64, replace the path with files from the `DiscoSG_datasets` directory
   - This ensures proper dataset loading for inference
   - In `dataset_utils.py` at line 136 and 167, replace the path with files from the `DiscoSG_datasets` directory.
   - Use the same dataset setting with [Detailcaps](https://github.com/foundation-multimodal-models/CAPTURE) and [Caparena](https://github.com/njucckevin/CapArena)

2. **Fast Inference with Reusable Graphs**
   
   For quick inference, you can replace the following parameters in `detailcap_discosg_mr.py` with JSON files from the `reusable_graph` directory:

   ```python
   parser.add_argument("--original_parse_dict", type=str, default=None, 
                      help="Path to the original parse dict file")
   parser.add_argument("--sub_sentence_parse_dict", type=str, default=None, 
                      help="Path to the sub sentence parse dict file")
   parser.add_argument("--combined_parse_dict", type=str, default=None, 
                      help="Path to the combined parse dict file")
   ```

   **Usage Example:**
   ```bash
   python detailcap_discosg_mr.py \
     --original_parse_dict reusable_graph/original_parse.json \
     --sub_sentence_parse_dict reusable_graph/sub_sentence_parse.json \
     --combined_parse_dict reusable_graph/combined_parse.json
   ```

3. **Reproduction Materials**
   
   We have included the following materials to help with reproduction:
   - **Log files**: Complete inference logs from our experiments
   - **Intermediate graphs**: Generated graph structures during the inference process
   
   **These materials allow researchers to verify our inference results**

4. **CAPTURE Metric**

   Get and replace capture.py from [CAPTURE](https://github.com/foundation-multimodal-models/CAPTURE) with capture.py in our repo.

### Directory Structure
```
├── detailcap_discosg_mr.py
├── caparena_mr.py
├── discourse_foil_acc_mr.py
├── DiscoSG_datasets/
│   └── [dataset files]
├── reusable_graph/
│   ├── Disco_large_subsent_100.json
    └── ...
└── logs/
    └── [inference logs]
```
