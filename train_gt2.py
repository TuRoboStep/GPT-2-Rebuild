import torch
import math
from config.GPTConfig import GPTConfig
from config.GPTConfig import DatasetConfig
from dataloader.data_loader_lite import DataLoaderLite
from model_architectures.GPT import GPT


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


config = GPTConfig(vocab_size=50304)
train_config = config.train_config
assert train_config.total_batch_size % (train_config.B * train_config.T) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = train_config.total_batch_size // (train_config.B * train_config.T)
print(f"total desired batch size: {train_config.total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# TODO we can include dataset config into config so we only pass one object here
train_loader = DataLoaderLite(B=train_config.B, T=train_config.T, dataset_config=DatasetConfig())

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


