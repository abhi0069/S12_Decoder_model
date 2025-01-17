# Cell 1: Install dependencies
 
!pip install tiktoken tqdm
 

# Cell 2: Import libraries and setup
 
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from google.colab import files
import os
import json
import datetime

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Check and setup device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Using device: {device}")

# Set seeds for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
 

# Cell 3: Model Architecture
 
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

@dataclass
class GPTConfig:
    block_size: int = 512  # Reduced for Colab
    vocab_size: int = 50304
    n_layer: int = 6      # Reduced for Colab
    n_head: int = 8       # Reduced for Colab
    n_embd: int = 384     # Reduced for Colab
    dropout: float = 0.1
    bias: bool = True

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)
        return optimizer
 

# Cell 4: Data Loading
'''
# Upload input.txt file
print("Please upload your input.txt file...")
uploaded = files.upload()

if 'input.txt' not in uploaded:
    raise Exception("input.txt was not uploaded. Please try again.")
'''

# Data loader
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding('gpt2')
        with open('/content/input.txt', 'r') as f:
            text = f.read()
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
 

# Cell 5: Training Setup and Loop
 
# Model initialization
torch.set_float32_matmul_precision('high')
config = GPTConfig()  # Create config instance
model = GPT(config)
model.to(device)

# Training settings - Adjusted for better convergence
max_lr = 5e-4  # Reduced learning rate for more stable training
min_lr = max_lr * 0.1
warmup_steps = 100  # Increased warmup for better stability
max_steps = 30000  # More steps since we're learning well

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Initialize data loader and optimizer
train_loader = DataLoaderLite(B=32, T=128)  # Changed to favor longer sequences
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device)

# Initialize logging
def initialize_training_log(config, train_loader, max_lr, min_lr, warmup_steps, max_steps, weight_decay, device, use_mixed_precision):
    return {
        "training_summary": {
            "model_architecture": {
                "n_layers": config.n_layer,
                "n_heads": config.n_head,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
                "block_size": config.block_size
            },
            "training_config": {
                "batch_size": train_loader.B,
                "context_length": train_loader.T,
                "learning_rate": {
                    "max": max_lr,
                    "min": min_lr
                },
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "weight_decay": weight_decay
            },
            "hardware_info": {
                "device": device,
                "precision": "bfloat16 mixed precision" if use_mixed_precision else "float32"
            },
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "training_logs": []
    }

# Initialize EMA tracking
ema_alpha = 0.99
ema_loss = None

def update_ema_loss(current_loss):
    global ema_loss
    if ema_loss is None:
        ema_loss = current_loss
    else:
        ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss
    return ema_loss

def update_training_log(log_data, step_metrics):
    log_data["training_logs"].append(step_metrics)
    
    # Update metrics.txt file
    with open('training_metrics.txt', 'w') as f:
        f.write("=== Training Configuration ===\n")
        f.write("Model Architecture:\n")
        for k, v in log_data["training_summary"]["model_architecture"].items():
            f.write(f"- {k}: {v}\n")
        
        f.write("\nTraining Settings:\n")
        for k, v in log_data["training_summary"]["training_config"].items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    f.write(f"- {k} {sub_k}: {sub_v}\n")
            else:
                f.write(f"- {k}: {v}\n")
        
        f.write("\n=== Latest Performance Metrics ===\n")
        metrics = step_metrics
        f.write(f"Current Loss: {metrics['loss']:.4f}\n")
        f.write(f"Best Loss: {best_loss:.4f}\n")
        f.write(f"Average Loss: {metrics['avg_loss']:.4f}\n")
        f.write(f"Learning Rate: {metrics['lr']:.2e}\n")
        f.write(f"Tokens/sec: {metrics['tokens_per_sec']:.2f}\n")
        f.write(f"Gradient Norm: {metrics['grad_norm']:.2f}\n")
        f.write(f"Batch Time: {metrics['dt_ms']:.2f}ms\n")
        f.write(f"Total Tokens: {metrics['tokens_processed']:,}\n")
        
        f.write("\n=== Hardware Configuration ===\n")
        for k, v in log_data["training_summary"]["hardware_info"].items():
            f.write(f"{k}: {v}\n")
        
        f.write(f"\nTraining Start Time: {log_data['training_summary']['start_time']}\n")
        f.write(f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Training loop setup
from tqdm.notebook import tqdm
import datetime
log_interval = 1  # Log more frequently
training_log = []
best_loss = float('inf')
start_time = time.time()
running_loss = 0.0
running_tokens = 0
ema_loss = None  # Will be initialized with first loss

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def log_training_stats(step, loss, lr, dt, tokens_per_sec, norm, elapsed_time, ema=None):
    print(f"\n{'='*80}")
    print(f"Step {step:5d} | Time: {format_time(elapsed_time)}")
    print(f"{'='*80}")
    print(f"Current Loss:    {loss:.4f}")
    if ema is not None:
        print(f"EMA Loss:        {ema:.4f}")
    print(f"Learning Rate:   {lr:.2e}")
    print(f"Tokens/sec:      {tokens_per_sec:.2f}")
    print(f"Grad Norm:       {norm:.2f}")
    print(f"Batch Time:      {dt:.2f}ms")
    if step > 0:
        print(f"Avg Loss:        {running_loss/step:.4f}")
        print(f"Tokens Seen:     {running_tokens:,}")
    print(f"{'='*80}\n")

# Enable mixed precision training if CUDA is available
use_mixed_precision = device == 'cuda'
scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

# Training loop
training_log_data = initialize_training_log(
    config=config,
    train_loader=train_loader,
    max_lr=max_lr,
    min_lr=min_lr,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    weight_decay=0.1,
    device=device,
    use_mixed_precision=use_mixed_precision
)

for step in tqdm(range(max_steps), desc="Training"):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    
    if use_mixed_precision:
        with torch.amp.autocast('cuda'):  # Fixed deprecated warning
            logits, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Update stats
    current_loss = loss.item()
    running_loss += current_loss
    running_tokens += train_loader.B * train_loader.T
    
    # Update EMA loss
    if ema_loss is None:
        ema_loss = current_loss
    else:
        ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    elapsed = t1 - start_time
    
    if step % log_interval == 0:
        step_metrics = {
            'step': step,
            'loss': current_loss,
            'avg_loss': running_loss/(step+1),
            'ema_loss': ema_loss,
            'lr': lr,
            'dt_ms': dt,
            'tokens_per_sec': tokens_per_sec,
            'grad_norm': norm.item(),
            'tokens_processed': running_tokens,
            'elapsed_time': elapsed,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update logs
        update_training_log(training_log_data, step_metrics)
        
        # Save JSON log
        with open('training_log.json', 'w') as f:
            json.dump(training_log_data, f, indent=4)
        
        # Print detailed stats
        log_training_stats(step, current_loss, lr, dt, tokens_per_sec, norm.item(), elapsed, ema=ema_loss)
        
        if ema_loss < best_loss:
            best_loss = ema_loss
            print(f"\n🌟 New best EMA loss: {best_loss:.4f}")
            
            # Save checkpoint with logs
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'ema_loss': ema_loss,
                'avg_loss': running_loss/(step+1),
                'tokens_processed': running_tokens,
                'elapsed_time': elapsed,
                'training_log': training_log_data
            }
            torch.save(checkpoint, 'model_best.pt')

# At the end of training, save final stats
final_stats = {
    'total_steps': step + 1,
    'best_loss': best_loss,
    'final_loss': loss.item(),
    'avg_loss': running_loss/(step+1),
    'ema_loss': ema_loss,
    'total_tokens': running_tokens,
    'training_time': elapsed,
    'tokens_per_second': running_tokens/elapsed
}

training_log_data["final_stats"] = final_stats

# Save final logs
with open('training_log.json', 'w') as f:
    json.dump(training_log_data, f, indent=4)

print("\n📊 Final Training Statistics:")
for k, v in final_stats.items():
    print(f"{k:20s}: {v}")

# Save final model with complete stats
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'training_stats': final_stats,
    'training_log': training_log_data
}, 'model_final.pt')

# Plot training progress
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
losses = [entry['loss'] for entry in training_log]
steps = [entry['step'] for entry in training_log]
plt.plot(steps, losses)
plt.title('Training Loss Over Time')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
plt.show()
 

# Cell 6: Text Generation
 
def generate_text(model, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    model.eval()
    input_ids = train_loader.enc.encode(prompt)
    x = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)[0]
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
    
    generated_ids = x[0].tolist()
    return train_loader.enc.decode(generated_ids)

# Generate sample text
if loss.item() < 0.1:
    print("\nGenerating sample text...")
    prompts = [
        "First Citizen:",
        "MENENIUS:",
        "All:",
        "The king",
        "My lord,"
    ]
    
    for prompt in prompts:
        generated = generate_text(model, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}\n")
        print("-" * 50)
  