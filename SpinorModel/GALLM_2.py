import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

# --- Constants for Clifford Algebra Cl(2,0) ---
# A multivector in Cl(2,0) has 4 components: [scalar, e1, e2, e12]
MV_DIM = 4 # Dimension of a single multivector in this chosen GA

# --- GA Operations (Cl(2,0) Specific) ---

def geometric_product_cl2_0(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes the geometric product of two multivectors in Cl(2,0).
    A, B are tensors of shape [..., MV_DIM], where MV_DIM=4.
    The components are assumed to be [scalar, e1_comp, e2_comp, e12_comp].
    """
    # Extract components
    s_A, e1_A, e2_A, e12_A = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    s_B, e1_B, e2_B, e12_B = B[..., 0], B[..., 1], B[..., 2], B[..., 3]

    # Initialize output components to zero
    s_C = torch.zeros_like(s_A)
    e1_C = torch.zeros_like(s_A)
    e2_C = torch.zeros_like(s_A)
    e12_C = torch.zeros_like(s_A)

    # --- Scalar part of C (s_C) ---
    s_C += s_A * s_B
    s_C += e1_A * e1_B  # e1*e1 = 1
    s_C += e2_A * e2_B  # e2*e2 = 1
    s_C -= e12_A * e12_B # e12*e12 = -1

    # --- e1 part of C (e1_C) ---
    e1_C += s_A * e1_B
    e1_C += e1_A * s_B
    e1_C -= e2_A * e12_B # e2*e12 = -e1
    e1_C += e12_A * e2_B # e12*e2 = e1

    # --- e2 part of C (e2_C) ---
    e2_C += s_A * e2_B
    e2_C += e2_A * s_B
    e2_C += e1_A * e12_B # e1*e12 = e2
    e2_C -= e12_A * e1_B # e12*e1 = -e2

    # --- e12 part of C (e12_C) ---
    e12_C += s_A * e12_B
    e12_C += e12_A * s_B
    e12_C += e1_A * e2_B  # e1*e2 = e12
    e12_C -= e2_A * e1_B  # e2*e1 = -e12

    return torch.stack([s_C, e1_C, e2_C, e12_C], dim=-1)

# --- GA-specific Activation Functions ---

class GAMultiplicationActivation(nn.Module):
    """
    This is a placeholder for a more advanced activation.
    Perhaps multiply the multivector by a learnable multivector.
    For this example, we'll keep it simple and just use ReLU on scalar and components.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple element-wise ReLU for MV components.
        # This is a pragmatic choice, true GA activations are complex research.
        return torch.relu(x)


# --- GALinear with true GA products ---
class GALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Ensure feature dimensions are multiples of MV_DIM
        assert in_features % MV_DIM == 0
        assert out_features % MV_DIM == 0

        self.in_multivectors = in_features // MV_DIM
        self.out_multivectors = out_features // MV_DIM

        # Each output multivector is a sum of geometric products of input multivectors
        # with learnable operator multivectors.
        # Shape: [in_num_mvs, out_num_mvs, MV_DIM]
        self.operator_weights = nn.Parameter(torch.randn(self.in_multivectors, self.out_multivectors, MV_DIM))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape [B, S, in_features]
        B, S, _ = x.shape
        
        # Reshape input to [B*S, self.in_multivectors, MV_DIM]
        x_reshaped = x.view(-1, self.in_multivectors, MV_DIM)
        
        output_components = []
        for out_mv_idx in range(self.out_multivectors):
            current_output_mv = torch.zeros_like(x_reshaped[:, 0, :]) # [B*S, MV_DIM]
            for in_mv_idx in range(self.in_multivectors):
                operator = self.operator_weights[in_mv_idx, out_mv_idx] # [MV_DIM]
                input_mv = x_reshaped[:, in_mv_idx, :] # [B*S, MV_DIM]
                
                # Compute the geometric product: operator * input_mv
                # operator needs to be expanded to [B*S, MV_DIM] for batching
                expanded_operator = operator.unsqueeze(0).expand_as(input_mv)
                prod = geometric_product_cl2_0(expanded_operator, input_mv)
                current_output_mv = current_output_mv + prod # Summing multivectors
            output_components.append(current_output_mv)
        
        # Concatenate all output multivectors
        output = torch.cat(output_components, dim=-1) # [B*S, out_features]
        output = output.view(B, S, self.out_multivectors * MV_DIM) + self.bias
        return output


# ---------- GA Positional Encoding ----------
class GAPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # The assert is good, keep it.
        assert d_model % MV_DIM == 0, "d_model must be a multiple of MV_DIM for proper GA representation."
        
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Standard sinusoidal positional encoding across the entire d_model.
        # This is the most common and robust approach in Transformers.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        encoding = encoding.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('encoding', encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)].to(x.device)


# ---------- GA Multi-head Attention ----------
class GAMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 32, heads: int = 4):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        assert d_model % MV_DIM == 0, "d_model must be a multiple of MV_DIM for proper GA representation."

        self.heads = heads
        self.d_head = d_model // heads
        
        # Ensure d_head is also a multiple of MV_DIM
        assert self.d_head % MV_DIM == 0, "d_head must be a multiple of MV_DIM."

        # Use GALinear for Q, K, V projections
        self.q_lin = nn.ModuleList([GALinear(d_model, self.d_head) for _ in range(heads)])
        self.k_lin = nn.ModuleList([GALinear(d_model, self.d_head) for _ in range(heads)])
        self.v_lin = nn.ModuleList([GALinear(d_model, self.d_head) for _ in range(heads)])
        
        self.out_lin = GALinear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, S, _ = x.size()
        all_heads = []

        for i in range(self.heads):
            Q = self.q_lin[i](x)  # [B, S, d_head] (collection of MVs)
            K = self.k_lin[i](x)  # [B, S, d_head] (collection of MVs)
            V = self.v_lin[i](x)  # [B, S, d_head] (collection of MVs)

            # Reshape Q, K to [B, S_query, 1, d_head // MV_DIM, MV_DIM] and [B, 1, S_key, d_head // MV_DIM, MV_DIM]
            Q_mv = Q.view(B, S, self.d_head // MV_DIM, MV_DIM)
            K_mv = K.view(B, S, self.d_head // MV_DIM, MV_DIM)
            
            expanded_Q_mv = Q_mv.unsqueeze(2) # [B, S_query, 1, d_head//MV_DIM, MV_DIM]
            expanded_K_mv = K_mv.unsqueeze(1) # [B, 1, S_key, d_head//MV_DIM, MV_DIM]

            # Compute geometric product for each multivector pair
            prod_QK = geometric_product_cl2_0(expanded_Q_mv, expanded_K_mv)
            
            # Extract scalar part: [B, S_query, S_key, d_head // MV_DIM]
            scalar_parts = prod_QK[..., 0]
            
            # Sum scalar parts over the multivector dimension to get a single score
            # Normalize by sqrt of the number of multivectors per head
            scores = scalar_parts.sum(dim=-1) / math.sqrt(self.d_head // MV_DIM) 

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn = torch.softmax(scores, dim=-1)  # [B, S, S]
            
            # Value aggregation: weighted sum of multivectors
            # attn: [B, S_query, S_key]
            # V_mv: [B, S_key, d_head // MV_DIM, MV_DIM]
            
            # Unsqueeze attn: [B, S_query, S_key, 1, 1] for broadcasting
            attn_expanded = attn.unsqueeze(-1).unsqueeze(-1)
            
            # V needs to be expanded for query dim: [B, 1, S_key, d_head // MV_DIM, MV_DIM]
            V_mv_expanded = V.view(B, S, self.d_head // MV_DIM, MV_DIM).unsqueeze(1) 
            
            # Weighted sum over S_key dimension
            # (attn_expanded * V_mv_expanded) would be [B, S_query, S_key, d_head//MV_DIM, MV_DIM]
            # Sum over S_key (dimension 2)
            out_mv = (attn_expanded * V_mv_expanded).sum(dim=2) # [B, S_query, d_head // MV_DIM, MV_DIM]

            out = out_mv.view(B, S, self.d_head) # Flatten back to [B, S, d_head]
            all_heads.append(out)

        concat = torch.cat(all_heads, dim=-1)  # [B, S, d_model]
        return self.out_lin(concat)


# ---------- GA Transformer Layer ----------
class GALayer(nn.Module):
    def __init__(self, d_model: int = 32):
        super().__init__()
        assert d_model % MV_DIM == 0, "d_model must be a multiple of MV_DIM."
        self.attn = GAMultiHeadAttention(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            GALinear(d_model, d_model * 4), # Intermediate features also collections of MVs
            GAMultiplicationActivation(), # GA-specific activation
            GALinear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# ---------- GA Transformer Model ----------
class GATransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, layers: int = 4, max_len: int = 512):
        super().__init__()
        assert d_model % MV_DIM == 0, "d_model must be a multiple of MV_DIM for proper GA representation."
        self.d_model = d_model
        
        # Token embeddings are still standard vectors, but we interpret them as flattened multivectors.
        # Initialize them to have components for each blade.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        nn.init.xavier_uniform_(self.token_embed.weight) # Better initialization

        self.pos_embed = GAPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([GALayer(d_model) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

        # Output layer: weight tied to token_embed weights
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.token_embed.weight # Weight tying

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, S] token indices
        attention_mask: [B, S] with 1 for tokens, 0 for padding
        """
        B, S = x.shape

        # Embed tokens + positions
        x = self.token_embed(x) * math.sqrt(self.d_model)  # scale embedding
        x = self.pos_embed(x)

        # Build causal mask (prevent attention to future tokens)
        causal_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)

        # Combine with attention_mask for padding
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).expand(-1, S, -1)  # [B,S,S]
            combined_mask = causal_mask & (padding_mask.bool())
        else:
            combined_mask = causal_mask

        # Convert to int mask (1 or 0)
        combined_mask = combined_mask.int()

        for layer in self.layers:
            x = layer(x, combined_mask)

        x = self.norm(x)
        logits = self.out(x)  # [B, S, vocab_size]

        return logits


# ---------- Tokenizer Setup ----------
def train_tokenizer_from_text(corpus_file: str, vocab_size: int = 256) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train([corpus_file], trainer)
    return tokenizer


# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, corpus: list[str], tokenizer: Tokenizer, pad_id: int):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int) -> list[int]:
        # Add BOS and EOS tokens around each sequence
        encoded = self.tokenizer.encode(self.corpus[idx])
        ids = [self.tokenizer.token_to_id("<BOS>")] + encoded.ids + [self.tokenizer.token_to_id("<EOS>")]
        return ids


def collate_fn(batch: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in batch)
    input_ids = []
    attention_mask = []
    for ids in batch:
        padded = ids + [pad_id] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        input_ids.append(padded)
        attention_mask.append(mask)
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def train_model(model: nn.Module, tokenizer: Tokenizer, corpus: list[str], epochs: int = 50, batch_size: int = 2, lr: float = 0.01):
    pad_id = tokenizer.token_to_id("<PAD>")
    if pad_id is None or pad_id < 0:
        raise ValueError("PAD token not found or invalid in tokenizer vocab!")

    dataset = TextDataset(corpus, tokenizer, pad_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            # Shift targets for causal LM (predict next token)
            targets = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :].contiguous()

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            loss = loss_fn(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.6f}")


# ---------- Decode Output ----------
def decode_output(tokenizer: Tokenizer, logits: torch.Tensor, stop_token: str = "<EOS>") -> str:
    """
    Greedy decode from logits, stopping at stop_token if found.
    logits: [1, S, V]
    """
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    decoded_tokens = []
    stop_id = tokenizer.token_to_id(stop_token)
    for i in ids:
        if i == stop_id:
            break
        decoded_tokens.append(i)
    return tokenizer.decode(decoded_tokens)


# ---------- Run ----------
if __name__ == "__main__":
    corpus_file = Path(__file__).with_name("corpus.txt")
    
    # Generate a dummy corpus if it doesn't exist or is empty
    if not os.path.exists(corpus_file) or os.stat(corpus_file).st_size == 0:
        corpus = [
            "This is an example.",
            "Geometric algebra meets attention.",
            "Spinors and tensors combined.",
            "Add more diverse sentences here for better tokenizer training.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
            "This script uses geometric algebra and transformers.",
            "Mathematical objects like vectors and bivectors are key.",
            "The geometric product unifies operations.",
            "Linear algebra is extended by geometric algebra."
        ]
        with open(corpus_file, "a") as f: # Use 'w' to overwrite/create
            f.write("\n".join(corpus))
        print("Corpus created.")
    else:
        with open(corpus_file, "r") as f:
            corpus = [line.strip() for line in f if line.strip()]
        print(f"Loaded corpus from '{corpus_file}'.")

    print("Training BPE tokenizer...")
    # Increase vocab size and add more diverse data for better tokenizer
    tokenizer = train_tokenizer_from_text(str(corpus_file), vocab_size=512)
    print("Tokenizer vocab size:", tokenizer.get_vocab_size())

    # Ensure d_model is a multiple of MV_DIM (4 for Cl(2,0))
    # We choose d_model=32, so it's 8 multivectors concatenated.
    d_model_val = 32 
    if d_model_val % MV_DIM != 0:
        raise ValueError(f"d_model ({d_model_val}) must be a multiple of MV_DIM ({MV_DIM}).")

    model = GATransformer(vocab_size=tokenizer.get_vocab_size(), d_model=d_model_val, layers=2) # Reduced layers for faster demo
    
    print("\nStarting model training...")
    # Increased epochs for potentially better learning, though small corpus limits.
    train_model(model, tokenizer, corpus, epochs=1000, batch_size=8, lr=0.005)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to {device}.")

    # Example generation
    test_input_text = "Hello, i am "
    # Ensure test input has BOS token
    test_input_ids = [tokenizer.token_to_id("<BOS>")] + tokenizer.encode(test_input_text).ids
    input_ids = torch.tensor([test_input_ids]).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = input_ids[0].tolist()
        max_generated_len = 56 # Max length for generation
        
        print(f"\nGenerating from: '{test_input_text}'")
        for _ in range(max_generated_len):
            current_input = torch.tensor([generated_ids]).to(device)
            
            output_logits = model(current_input) # [1, current_len, vocab_size]
            next_token_logits = output_logits[:, -1, :] # Logits for the last token
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id == tokenizer.token_to_id("<EOS>"):
                break
            
            generated_ids.append(next_token_id)
            
            # Simple print during generation
            print(tokenizer.decode(generated_ids[1:])) # Exclude BOS for print

    final_decoded_output = tokenizer.decode(generated_ids[1:]) # Exclude BOS
    print(f"\nFinal Generated output: {final_decoded_output}")
