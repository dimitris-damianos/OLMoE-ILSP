# SFT for Domain Experts

Below are the domain-specific datasets used for supervised fine-tuning (SFT):

| Domain                   | Dataset Name                     | Hugging Face Path                                                                 | Description |
|--------------------------|----------------------------------|------------------------------------------------------------------------------------|-------------|
| **Math**                | MathInstruct                     | [TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)  | ~262K math instruction–response pairs featuring chain-of-thought and program-of-thought reasoning across 13 diverse math sources. |
| **Code**                | Tested Python Code               | [Vezora/Tested-143k-Python-Alpaca](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca) | ~143K high-quality Python programming tasks with unit-tested correct solutions for reliable code generation. |
| **Multilingual**        | Aya Dataset                      | [CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | ~204K human‑annotated instruction pairs in 65+ languages, curated via the Aya platform. |
| **General Instruction** | Alpaca                           | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)              | ~52K English instruction–response examples, popular baseline for instruction tuning. |
| **Social Reasoning**    | Social IQA                       | [allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)        | ~37K QA pairs probing emotional/social inference in everyday scenarios. |
| **Medical QA**          | PubMedQA                         | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)              | 1K expert-annotated and 61K unlabeled + 211K synthetic biomedical QA based on PubMed abstracts. |
| **Physical Commonsense**| PIQA                             | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)                          | ~20K multiple-choice QA targeting physical commonsense reasoning. |
| **Causal Reasoning**    | e-CARE                           | [12ml/e-CARE](https://huggingface.co/datasets/12ml/e-CARE)                         | ~21K human-annotated causal reasoning QAs with explanations. |
| **Biomedical Chemistry**| MoleculeQA                       | [hcaoaf/MoleculeQA](https://huggingface.co/datasets/hcaoaf/MoleculeQA)            | ~62K question-answer pairs evaluating molecular factual accuracy. |
| **Legal Reasoning**     | CaseHOLD                         | [casehold/casehold](https://huggingface.co/datasets/casehold/casehold)            | ~53K multiple choice questions asking for judicial holdings from case citations. |
| **Financial Reasoning** | FinQA                            | [ibm-research/finqa](https://huggingface.co/datasets/ibm-research/finqa)          | ~8K expert-written multi-hop finance QA instances over reports with annotated reasoning. |