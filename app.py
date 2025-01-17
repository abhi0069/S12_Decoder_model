import torch
import torch.nn.functional as F
import gradio as gr
import tiktoken
from model import GPT, GPTConfig

# Load the model
def load_model():
    config = GPTConfig()
    model = GPT(config)
    checkpoint = torch.load('model_best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Initialize tokenizer
enc = tiktoken.get_encoding('gpt2')

def generate_text(prompt, max_words=50, num_variations=1):
    model = load_model()
    
    # Convert max_words to tokens (approximately)
    max_tokens = max_words * 2  # Rough approximation: 1 word ≈ 2 tokens
    
    # Generate multiple variations
    variations = []
    for _ in range(num_variations):
        input_ids = enc.encode(prompt)
        x = torch.tensor(input_ids).unsqueeze(0)
        
        with torch.no_grad():
            # Generate text
            for _ in range(max_tokens):
                logits = model(x)[0]
                logits = logits[:, -1, :] / 0.8  # Fixed temperature of 0.8
                v, _ = torch.topk(logits, min(50, logits.size(-1)))  # Fixed top_k of 50
                logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, next_token), dim=1)
        
        generated_ids = x[0].tolist()
        generated_text = enc.decode(generated_ids)
        variations.append(generated_text)
    
    # Format output
    if num_variations == 1:
        return variations[0]
    else:
        formatted_variations = []
        for i, text in enumerate(variations):
            formatted_variations.append(f"\n=== Variation {i+1} ===\n{text}")
        return "\n".join(formatted_variations)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            placeholder="Example: 'First Citizen:', 'The king', 'My lord,'",
            lines=3
        ),
        gr.Slider(
            minimum=10,
            maximum=200,
            value=50,
            step=10,
            label="Number of words to generate"
        ),
        gr.Slider(
            minimum=1,
            maximum=5,
            value=1,
            step=1,
            label="Number of variations"
        )
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Shakespeare-style Text Generator",
    description="""Enter a prompt and generate Shakespeare-style text continuations.
    
    - **Number of words**: Control how long the generated text should be (10-200 words)
    - **Number of variations**: Generate multiple different continuations (1-5 variations)
    
    Example prompts:
    - "First Citizen:"
    - "MENENIUS:"
    - "The king"
    - "My lord,"
    """,
    examples=[
        ["First Citizen:", 50, 1],
        ["The king", 100, 2],
        ["My lord,", 30, 3]
    ]
)

iface.launch() 