import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

batch_size = 64
block_size = 256
epochs = 5000
step_at = 500
learning_rate = 3e-4
n_embed = 384
n_heads = 6
n_layers = 6
dropout = 0.2

train_ratio = 0.9

text = open('dataset.txt').read()
characters = sorted(list(set(text)))
vocabulary_size = len(characters)

stoi = {char: i for i, char in enumerate(characters)}
itos = {i: char for i, char in enumerate(characters)}

train_size = int(len(text)*train_ratio)
train_data = text[: train_size]
val_data = text[train_size * len(text): ]

encode = lambda s : [stoi[c] for c in s]
decode = lambda ls : "".join([itos[i] for i in ls])

train_t = torch.tensor(encode(train_data))
val_t = torch.tensor(encode(val_data))

def get_batches(dataset_type) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset_type == 'train':
        data = train_t
    else:
        data = val_t

    start_points = torch.randint(len(data) - block_size, (batch_size,))

    x = [data[i: i + block_size] for i in start_points]
    y = [data[i + 1: i + block_size + 1] for i in start_points]

    return torch.stack(x).to(device=device), torch.stack(y).to(device=device)

xb, yb = get_batches('train')

class AttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.query(x)   # B x T x head_size
        k = self.key(x)     # B x T x head_size
        v = self.value(x)   # B x T x head_size

        B, T, C = x.shape

        wei = q @ k.transpose(-2, -1)
        wei = wei / (C ** 0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # decoder as masking un-necessary
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out
    

class MultiAttentionHead(nn.Module):
    def __init__(self, head_size, parallels) -> None:
        super().__init__()
        self.mult_heads = nn.ModuleList(
            [AttentionHead(head_size=head_size) for _ in range(parallels)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # along C
        op = torch.cat([head(x) for head in self.mult_heads], dim=-1)
        return self.dropout(self.proj(op))
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_heads) -> None:
        super().__init__()
        self.att_heads = MultiAttentionHead(n_embed // n_heads, n_heads)
        self.ffrwd = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = self.ln1(x)
        x = x + self.att_heads(x)
        x = self.ln2(x)
        x = x + self.ffrwd(x)

        return x



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocabulary_size, n_embed) # 65 * 32
        self.position_emb_table = nn.Embedding(block_size, n_embed)
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocabulary_size)
        self.blocks = nn.Sequential(
            *[Block(n_embed=n_embed, n_heads=n_heads) for i in range(n_layers)],
        )

    def forward(self, x, y):
        '''
            x -> 32 x 8
            y -> 32 x 8
        '''
        B, T = x.shape
        emb = self.embedding_table(x) # 32 * 8 * 65
        # this means - Each batch has 8 characters (tokens). For each token (emb[batch][token]), emb[batch][token][i] has probability of presence of that token for token at emb[batch][token]
        pos_emb = self.position_emb_table(torch.arange(T, device=device))
        x = emb + pos_emb
        bx = self.blocks(x)
        bx = self.ln(bx)
        logits = self.lm_head(bx)

        if y is None:
            return logits
        
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.view(B * T)
        
        loss = F.cross_entropy(logits, y)

        return logits, loss
    
    def generate(self, max_tokens=50):
        # max_tokens = 50
        print(max_tokens)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        while True:
            trimmed_context = context[:, -block_size:] # cropping only last x chars where x is block size
            logits = self(trimmed_context, None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

            if context.shape[1] >= max_tokens:
                break

        return decode(context[0].tolist())

def train_model():
    model = Model()
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        xb, yb = get_batches('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 0 or epoch % step_at == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    print(loss.item())
    # print(model.state_dict())

    torch.save(model.state_dict(), 'model_weights.pth')


    # loaded_model = torch.load('model_weights.pth')

    # model.load_state_dict(loaded_model)

    print(model.generate(100))
    
# loss = 1.8405121564865112
