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