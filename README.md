# Gardiner Code Translation Approaches

This project explores and implements three different approaches to translating Gardiner codes into their English meanings. The approaches include:

- **Regular Fine-Tuning**
- **Fine-Tuning with Retrieval-Augmented Generation (RAG)**
- **Prompt Engineering Without Fine-Tuning**

## Project Structure

### Fine-Tuning Approaches

- **Base Model**: Utilized LLaMA 3.2 models with parameter-efficient tuning using QLoRA.
- **Dataset**: Gardiner codes and their English translations.
- **Evaluation Metrics**: BLEU, ROUGE-L, F1, accuracy, precision, and recall.

### Prompt Engineering Approach

- Eliminates fine-tuning by leveraging pre-trained models.
- Provides dynamic and context-aware queries using custom-designed prompts.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install required dependencies:

   ```bash
   pip install -U transformers datasets accelerate peft trl bitsandbytes
   ```

3. Ensure a GPU-enabled environment for optimal performance.

## Running the Code

### Fine-Tuning Approaches

1. **Prepare the Dataset:**
   - Ensure the dataset is in a structured format.
   - Include Gardiner codes with their English translations.

2. **Train the Model:**
   - Run the training script for regular fine-tuning or fine-tuning with RAG.

3. **Evaluate:**
   - Use provided metrics (BLEU, ROUGE-L, F1, accuracy, etc.) to evaluate performance.

### Prompt Engineering Approach

1. **Setup Prompting Pipeline:**
   - Use the pre-trained LLaMA 3.2 3B Instruct model.
   - Load dynamic prompts containing Gardiner codes and instructions.

2. **Run Queries:**
   - Use the provided script to query the model for translations and generate coherent sentences.

## Key Features

### Regular Fine-Tuning

- Trained with PEFT QLoRA, 4-bit quantization, and LoRA for efficient memory and computational usage.
- Fine-tuned specific layers: `q_proj` and `v_proj`.

### Retrieval-Augmented Generation (RAG)

- Integrated vector embeddings and semantic retrieval using LangChain and Chroma vector database.

### Prompt Engineering Without Fine-Tuning

- Utilized structured prompts to generate accurate translations without modifying model weights.
- Supports dynamic context retrieval and robust handling of unknown codes.

## Example: Generating Sentences

### Input Gardiner Codes:

- `A1`: "man"
- `A2`: "woman"
- `G1`: "goose"
- `F12`: "basket"

### Generated Sentence:

> A man and a woman offered a goose in a basket.

## Challenges and Resolutions

1. **Memory Constraints:**
   - Addressed with QLoRA and 4-bit quantization.

2. **Tokenizer Issues:**
   - Added a padding token to maintain compatibility.

3. **Dataset Formatting:**
   - Developed preprocessing scripts to create structured conversational templates.

4. **Evaluation Metrics:**
   - Implemented comprehensive metrics for robust evaluation.

---

Contributions and feedback are welcome! Please submit issues or pull requests to improve the project.
