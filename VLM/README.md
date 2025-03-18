# **Project Setup and Usage Guide**

## 📝 Overview

Our method, **SAINT**, is designed to optimize Vision-Language Models (VLMs) by pruning them in three distinct ways. This enables a balance between inference speed and model performance. The three approaches are:

1. **ViT-Only Pruning**  
   - This method drops similar visual tokens in the Vision Transformer (ViT) without considering text information.  
   - It is **text-agnostic** and offers **fast inference**, but **performance degrades significantly** when too many tokens are removed.

2. **LLM-Only Pruning**  
   - In this approach, vision tokens within the language model (LLM) are pruned.  
   - Unlike ViT-only pruning, it maintains **better performance even with aggressive token removal**, but **gains less speed improvement** compared to the first method.

3. **Hybrid Pruning (Best of Both Worlds)**  
   - This approach **combines the strengths** of both methods.  
   - Initially, around **30% of vision tokens are pruned** in ViT for a speed boost.  
   - Additional pruning occurs within the LLM, allowing it to selectively remove redundant vision tokens **while considering textual context**.  
   - This achieves **both speed improvement and minimal performance loss**.

---

## 📚 Prerequisites
- Python 3.10
- `pip` package manager

---

## ⚙️ Installation Instructions

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

## 🚀 Running the Code

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

### 🔍 Important Command Parameters
- `--model_args`: Provides arguments for the model, like the model checkpoint path.
- `--tasks mme`: Specifies the dataset/task.
- `--batch_size 1`: Sets the batch size for processing.

---
