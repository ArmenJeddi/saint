
# **Project Setup and Usage Guide**

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

3. **Install Project Dependencies in Editable Mode**  
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

### 🔍 Command Parameters
- `--num_processes=8`: Defines the number of parallel processes.
- `--model llava`: Specifies the model to use.
- `--model_args`: Provides arguments for the model.
- `--tasks mme`: Specifies the dataset/task.
- `--batch_size 1`: Sets the batch size for processing.
- `--log_samples`: Enables detailed sample logging.
- `--log_samples_suffix`: Appends a custom suffix to the log files.
- `--output_path logs`: Defines the directory for log storage.

---
