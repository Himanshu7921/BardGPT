import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "decoder-only-design", text: "Decoder-Only Design", level: 2 },
  { id: "embedding-layer", text: "Embedding Layer", level: 2 },
  { id: "multi-head-attention", text: "Multi-Head Attention", level: 2 },
  { id: "causal-masking", text: "Causal Masking", level: 3 },
  { id: "attention-computation", text: "Attention Computation", level: 3 },
  { id: "feedforward-network", text: "Feedforward Network", level: 2 },
  { id: "layer-normalization", text: "Layer Normalization", level: 2 },
  { id: "residual-connections", text: "Residual Connections", level: 2 },
  { id: "output-projection", text: "Output Projection", level: 2 },
  { id: "weight-tying", text: "Weight Tying", level: 2 },
];

export default function Architecture() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Architecture" }]} />
      
      <div className="doc-prose">
        <h1>Model Architecture</h1>
        
        <p className="text-lg text-muted-foreground">
          A detailed examination of BardGPT's decoder-only transformer architecture 
          and its components.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          BardGPT implements the decoder-only transformer architecture introduced in 
          "Attention Is All You Need" (Vaswani et al., 2017) and refined in GPT 
          (Radford et al., 2018). The architecture consists of stacked decoder blocks, 
          each containing multi-head self-attention and position-wise feedforward networks.
        </p>

        <div className="bg-card border border-border rounded-lg p-6 my-6">
          <h4 className="text-sm font-medium text-muted-foreground mb-4">Architecture Diagram</h4>
          <div className="font-mono text-sm space-y-2">
            <p className="text-muted-foreground">Input Tokens</p>
            <p className="pl-4">↓</p>
            <p className="pl-4 text-primary">Token Embeddings + Positional Embeddings</p>
            <p className="pl-4">↓</p>
            <p className="pl-4 text-primary">Dropout</p>
            <p className="pl-4">↓</p>
            <p className="pl-4 border-l-2 border-primary pl-6">
              <span className="text-muted-foreground">× N layers:</span>
              <br />
              → LayerNorm → Multi-Head Attention → Residual
              <br />
              → LayerNorm → FeedForward → Residual
            </p>
            <p className="pl-4">↓</p>
            <p className="pl-4 text-primary">Final LayerNorm</p>
            <p className="pl-4">↓</p>
            <p className="pl-4 text-primary">Linear Projection (vocab_size)</p>
            <p className="pl-4">↓</p>
            <p className="text-muted-foreground">Output Logits</p>
          </div>
        </div>

        <h2 id="decoder-only-design">Decoder-Only Design</h2>
        <p>
          Unlike encoder-decoder transformers used in machine translation, BardGPT uses 
          only the decoder stack. This design is optimal for autoregressive language 
          modeling where each token is predicted based solely on preceding tokens.
        </p>

        <p>Key characteristics:</p>
        <ul>
          <li>Unidirectional attention (tokens can only attend to previous positions)</li>
          <li>Causal masking prevents information leakage from future tokens</li>
          <li>Same architecture used during training and inference</li>
        </ul>

        <h2 id="embedding-layer">Embedding Layer</h2>
        <p>
          The embedding layer combines token embeddings with positional information:
        </p>

        <CodeBlock
          language="python"
          code={`class EmbeddingBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embed = TokenEmbeddings(config.vocab_size, config.d_model)
        
        if config.use_fixed_positional_embeddings:
            self.pos_embed = FixedPositionalEmbeddings(
                config.d_model, 
                config.max_seq_length
            )
        else:
            self.pos_embed = LearnedPositionalEmbeddings(
                config.d_model, 
                config.max_seq_length
            )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_length)
        tok_emb = self.token_embed(x)  # (B, S, d_model)
        pos_emb = self.pos_embed(x)    # (S, d_model)
        return self.dropout(tok_emb + pos_emb)`}
        />

        <Callout type="info" title="Positional Embeddings">
          BardGPT supports both learned and fixed (sinusoidal) positional embeddings. 
          Learned embeddings are used by default as they typically perform better for 
          shorter sequences.
        </Callout>

        <h2 id="multi-head-attention">Multi-Head Attention</h2>
        <p>
          Multi-head attention allows the model to jointly attend to information from 
          different representation subspaces at different positions.
        </p>

        <h3 id="causal-masking">Causal Masking</h3>
        <p>
          A causal mask ensures that position <code>i</code> can only attend to positions 
          <code>≤ i</code>. This prevents the model from "seeing the future" during training:
        </p>

        <CodeBlock
          language="python"
          code={`def causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Creates an upper triangular mask for causal attention.
    
    Returns:
        Tensor of shape (seq_length, seq_length) where:
        - mask[i, j] = 0 if j <= i (allowed attention)
        - mask[i, j] = -inf if j > i (blocked attention)
    """
    mask = torch.triu(
        torch.ones(seq_length, seq_length, device=device), 
        diagonal=1
    )
    return mask.masked_fill(mask == 1, float('-inf'))`}
        />

        <h3 id="attention-computation">Attention Computation</h3>
        <p>
          The scaled dot-product attention is computed as:
        </p>

        <div className="bg-muted/50 rounded-lg p-4 my-4 text-center font-mono">
          Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub> + mask) · V
        </div>

        <CodeBlock
          language="python"
          code={`class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def get_attention_scores(self, query, key, mask=None):
        # query, key: (B, h, S, d_k)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
        
        return F.softmax(scores, dim=-1)
    
    def forward(self, q, k, v, mask=None):
        B, S, _ = q.size()
        
        # Linear projections and reshape to (B, h, S, d_k)
        Q = self.W_Q(q).view(B, S, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(B, S, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(B, S, self.h, self.d_k).transpose(1, 2)
        
        # Attention
        attn = self.get_attention_scores(Q, K, mask)
        attn = self.dropout(attn)
        
        # Combine heads
        out = torch.matmul(attn, V)  # (B, h, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        
        return self.W_O(out)`}
        />

        <h2 id="feedforward-network">Feedforward Network</h2>
        <p>
          Each decoder block contains a position-wise feedforward network with an 
          expansion factor of 4x:
        </p>

        <CodeBlock
          language="python"
          code={`class TransformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, S, d_model)
        x = self.linear1(x)        # (B, S, d_ff)
        x = F.gelu(x)              # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)        # (B, S, d_model)
        return x`}
        />

        <Callout type="tip" title="GELU Activation">
          BardGPT uses the Gaussian Error Linear Unit (GELU) activation function, 
          which provides smoother gradients compared to ReLU and is standard in 
          modern transformers.
        </Callout>

        <h2 id="layer-normalization">Layer Normalization</h2>
        <p>
          Layer normalization is applied before each sub-layer (Pre-LN architecture):
        </p>

        <CodeBlock
          language="python"
          code={`class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, S, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta`}
        />

        <h2 id="residual-connections">Residual Connections</h2>
        <p>
          Pre-LN residual connections wrap each sub-layer. The Pre-LN variant applies 
          normalization before the sub-layer rather than after, which improves training 
          stability:
        </p>

        <CodeBlock
          language="python"
          code={`class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Pre-LN: normalize, apply sublayer, add residual
        return x + self.dropout(sublayer(self.norm(x)))`}
        />

        <h2 id="output-projection">Output Projection</h2>
        <p>
          The final layer projects the transformer output back to vocabulary size:
        </p>

        <CodeBlock
          language="python"
          code={`class LinearProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # x: (B, S, d_model) -> (B, S, vocab_size)
        return self.linear(x)`}
        />

        <h2 id="weight-tying">Weight Tying</h2>
        <p>
          BardGPT supports weight tying between the token embedding matrix and the 
          output projection layer. This reduces parameters and often improves 
          generalization:
        </p>

        <CodeBlock
          language="python"
          code={`# In DecoderOnlyTransformerModel.__init__:
self.embedding = EmbeddingBlock(config)
self.linear_head = LinearProjection(config.d_model, config.vocab_size)

# Tie weights
self.linear_head.linear.weight = self.embedding.token_embed.embedding.weight`}
        />

        <Callout type="info" title="Parameter Reduction">
          Weight tying reduces the total parameter count by <code>vocab_size × d_model</code> 
          parameters. For BardGPT's default configuration, this saves approximately 25K parameters.
        </Callout>
      </div>
    </DocLayout>
  );
}
