# **Project Setup and Usage Guide**

## 📖 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Instructions](#installation-instructions)
4. [Configuration for Different Methods](#configuration-for-different-methods)
   - [ViT-Only Pruning](#vit-only-pruning)
   - [LLM-Only Pruning](#llm-only-pruning)
   - [Hybrid Pruning](#hybrid-pruning)
5. [Running the Code](#running-the-code)
6. [Command Parameters](#command-parameters)

---

## Overview

Our method, **SAINT**, is designed to optimize Vision-Language Models (VLMs) by pruning them in three distinct ways. This enables a balance between inference speed and model performance. The three approaches are:

1. **ViT-Only Pruning**  
   - This method drops similar visual tokens in the Vision Transformer (ViT) without considering text information.  
   - It is **task-agnostic** and offers **fast inference**, but **performance degrades significantly** when too many tokens are removed.

2. **LLM-Only Pruning**  
   - In this approach, vision tokens within the language model (LLM) are pruned.  
   - Unlike ViT-only pruning, it maintains **better performance even with aggressive token removal**, but **lacks the speed improvements** of the first method.

3. **Hybrid Pruning (Best of Both Worlds)**  
   - This approach **combines the strengths** of both methods.  
   - Initially, around **30% of vision tokens are pruned** in ViT for a speed boost.  
   - Additional pruning occurs within the LLM, allowing it to selectively remove redundant vision tokens **while considering textual context**.  
   - This achieves **both speed improvement and minimal performance loss**.

---

## Prerequisites
- Python 3.10

---

## Installation Instructions

1. **Create a Python Virtual Environment**
   ```bash
   python3.10 -m venv env
   source env/bin/activate
   ```

2. **Navigate to the `VLM` Directory**  
   From the project's root directory, enter the `VLM` folder:

   ```bash
   cd VLM
   ```

3. **Install Project Dependencies**  
   Execute the following commands sequentially:

   ```bash
   cd lmms_eval
   pip install -e .

   cd ../LLaVA
   pip install -e .

   cd ../transformers
   pip install -e .
   ```

4. **Configure Python Path**  
   Set the `PYTHONPATH` environment variable to the `VLM` directory:

   ```bash
   export PYTHONPATH=$(pwd)
   ```

   > **Note:** Ensure you execute this command while inside the `VLM` directory.

---

## Configuration for Different Methods

All pruning methods are controlled using the following file:
**`VLM/lmms-eval/lmms_eval/models/llava.py`**

### ViT-Only Pruning
- The **`saint.patch.clip`** function is called to handle token dropping.
- Any combination of layers and dropping methods can be specified.
- For layers where no pruning is required, set `prune_mode = "None"`.
- For layers where pruning is needed, set `prune_mode = "iterative_drop_full_graph"`.
- Parameters **`keep_num`** and **`r`** should be adjusted in **`VLM/saint/patch/clip.py`**.
- Ensure **`use_saint`** in the LLM-only section is set to **`False`** when running in ViT-only mode.

### LLM-Only Pruning
- Ensure the ViT-only section is **commented out**.
- Set **`use_saint = True`** in the LLM-only section.
- Specify the parameters:
  - **`saint_start`**
  - **`saint_end`**
  - **`saint_threshold`**

### Hybrid Pruning
- Set **`use_saint = True`**.
- Configure settings in both the **ViT-only** and **LLM-only** sections.
- Adjust ViT-only parameters in **`VLM/saint/patch/clip.py`**.
- Set **`saint_image_token_length`** in the LLM-only part to match the number of remaining tokens after ViT pruning.

---

## Running the Code

To execute the code for the **MME** dataset, use the following command:

```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path logs
```

## Command Parameters
- `--model_args`: Provides arguments for the model, like the model checkpoint path.
- `--tasks mme`: Specifies the dataset/task.
- `--batch_size 1`: Sets the batch size for processing.

---
