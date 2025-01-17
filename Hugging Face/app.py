import torch
import torch.nn.functional as F
import gradio as gr
import tiktoken
from model import GPT, GPTConfig  # You'll need to create model.py with your architecture

# Load the model
def load_model():
    config = GPTConfig()  # Make sure this matches your training config
    model = GPT(config)
    checkpoint = torch.load('model_best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Text generation function
def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50):
    model = load_model()
    enc = tiktoken.get_encoding('gpt2')
    
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids).unsqueeze(0)
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)[0]
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            x = torch.cat((x, next_token), dim=1)
    
    generated_text = enc.decode(generated_tokens)
    return prompt + generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=3),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Shakespeare-style Text Generator",
    description="Enter a prompt and generate Shakespeare-style text using the trained GPT model.",
    examples=[
        ["First Citizen:", 100, 0.8, 50],
        ["MENENIUS:", 100, 0.8, 50],
        ["The king", 100, 0.8, 50],
    ]
)

iface.launch() 