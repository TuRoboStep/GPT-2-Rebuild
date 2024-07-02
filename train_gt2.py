from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embeding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights) # calls all submodules and applies the _init_weights function to it


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5 # 2* comes from each redisual plcok get's contributios from mlp ad attention layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', "gpt2-medium", "gpt2-large", "gpt-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024), # 3504M params
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GP model checkpoints
        config_args['block_size'] = 1024 # always 50257 for GP model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        # init a hugginfface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1" module, but we only want to use a vanilla versio
        # this means, that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self, idx, target = None) -> tuple[torch.tensor, torch.tensor]:
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1)) # (B*T, vocabsize), (B*T)
        return logits, loss


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output_projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularizatio
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following die OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values, for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of chanels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs = 6, so nh*hs = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # transform tensor so pytorch treats (B, nh) as "batch dimension" and applies all operations
        # to those dimensions in parallel (all the batches and all the heads)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # attention (materializes the larte (T,T) matric for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # # mask out future attentions
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # # normalize the attention so it sums to 1
        # att = F.softmax(att, dim=-1)
        # # weighted sum of all the values that the net finds interesting
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        

        # re-assemble all head outputs side by side, in essence concatenation
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class DataLoaderLite:
    def __init__(self, B, T):
        self.B  = B
        self.T = T

        # at init load tokens from disk and store them in memty

        enc = tiktoken.get_encoding('gpt2')
        with open('../datasets/tiny_shakespear/input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)// (B*T)}")

        # state
        self.current_position = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        #if loading the next batch would be out o fbounds, reset
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmuo for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# ------------------------------------------------------------------------
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

#torch.set_float32_matmul_precision('high') # doesn't spee dup at all on my GPU GFORCE 1070

num_return_sequences = 5
max_length = 30

model =  GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)

# not using compile because it's not improving the training time
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# model = torch.compile(model) # cannot be used by GTX 1070 because is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability 6.1

# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
    
        #with torch.autocast(device_type=device, dtype=torch.bfloat16): # would increase speed for A100 but not for GF 1070
        logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # length of the grad is clipped to 1 to prevent large gradients
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(f"step {step}, loss: {loss_accum.item()} | lr: {lr:.4f} | gradient_norm: {norm:.4f} | dt: {dt}, tok/sec: {tokens_per_sec}")


print(loss)


# inference
# model =  GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
# x = tokens.to('cuda')

# # generate! right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         # forward the model to get the logits
#         logits = model(x)  #(B, T, vocab_size)
#         #take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probablilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k- sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indeices
#         xcol = torch.gather(topk_indices, -1, ix)  #(B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)

# # print the generated text        
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


