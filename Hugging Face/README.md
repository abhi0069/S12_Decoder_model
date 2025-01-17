# Shakespeare-style Text Generator

A GPT-2 based model trained on Shakespeare's works to generate Shakespeare-style text.

## Model Details

- Architecture: Custom GPT model (6 layers, 8 attention heads)
- Vocabulary: GPT-2 tokenizer (50304 tokens)
- Training Data: Shakespeare's works
- Model Size: ~45M parameters

## Usage

Enter a prompt and the model will generate Shakespeare-style text continuation. You can adjust:

- **Max Tokens**: Control the length of generated text (10-500)
- **Temperature**: Control randomness (0.1-2.0)
  - Lower values (e.g., 0.3) = more focused and deterministic
  - Higher values (e.g., 1.5) = more creative and random
- **Top-K**: Control diversity of word choices (1-100)

## Example Prompts

- "First Citizen:"
- "MENENIUS:"
- "The king"

## Files

- `app.py`: Gradio interface
- `model.py`: Model architecture
- `model_best.pt`: Trained model weights
- `requirements.txt`: Dependencies 