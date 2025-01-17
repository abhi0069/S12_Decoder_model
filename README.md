 # Shakespeare-Style Text Generator

A GPT-2 based model trained on Shakespeare's works to generate Shakespeare-style text. This project includes both the training code and a web interface for text generation.

## Project Structure

```
project/
├── Assignment_12_Colab.py    # Training code
├── app.py                    # Gradio web interface
├── model.py                  # Model architecture
├── requirements.txt          # Dependencies
├── README.md                # This file
└── .gitignore              # Git ignore rules
```

## Features

- Custom GPT model with 6 layers and 8 attention heads
- Trained on Shakespeare's complete works
- Interactive web interface using Gradio
- Support for multiple text variations
- Adjustable generation length
- Example prompts included

## Model Architecture

- Layers: 6
- Attention Heads: 8
- Embedding Dimension: 384
- Vocabulary: GPT-2 tokenizer (50304 tokens)
- Parameters: ~45M

## Setup and Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model:
- Download `model_best.pt` from the releases page
- Place it in the project root directory

## Usage

### Local Development

Run the Gradio interface locally:
```bash
python app.py
```

### Training

To train the model:
```bash
python Assignment_12_Colab.py
```

### Hugging Face Spaces

The model is also deployed on Hugging Face Spaces at [your-space-url].

## Files Not Included in Repository

The following files are not included in the repository due to size or privacy:
- `model_best.pt` (trained model checkpoint)
- `model_final.pt` (final model state)
- `input.txt` (training data)
- `training_log.json` (training statistics)
- `training_metrics.txt` (training metrics)

## Example Usage

```python
from model import GPT, GPTConfig
import torch

# Load model
config = GPTConfig()
model = GPT(config)
checkpoint = torch.load('model_best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate text
prompt = "First Citizen:"
# ... generation code ...
```

## Training Results

- Best Loss: [Your best loss]
- Training Steps: [Total steps]
- Training Time: [Total time]
- Hardware Used: [Your hardware details]

## License

[Your chosen license]

## Acknowledgments

- Based on the GPT architecture
- Training data from Shakespeare's complete works
- Part of The School of AI coursework