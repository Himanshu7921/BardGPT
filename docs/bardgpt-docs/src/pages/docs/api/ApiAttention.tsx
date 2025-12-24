import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "multiheadattention", text: "MultiHeadAttention", level: 2 },
  { id: "constructor", text: "Constructor", level: 3 },
  { id: "get-attention-scores", text: "get_attention_scores()", level: 3 },
  { id: "forward", text: "forward()", level: 3 },
  { id: "tensor-shapes", text: "Tensor Shapes", level: 2 },
  { id: "masking-behavior", text: "Masking Behavior", level: 2 },
  { id: "dropout-behavior", text: "Dropout Behavior", level: 2 },
];

const constructorParams = [
  { name: "d_model", type: "int", description: "Dimension of the model's hidden states." },
  { name: "h", type: "int", description: "Number of attention heads. Must divide d_model evenly." },
  { name: "dropout", type: "float", description: "Dropout probability applied to attention weights." },
];

const attentionScoreParams = [
  { name: "query", type: "torch.Tensor", description: "Query tensor of shape (B, h, S, d_k)." },
  { name: "key", type: "torch.Tensor", description: "Key tensor of shape (B, h, S, d_k)." },
  { name: "mask", type: "torch.Tensor | None", description: "Optional mask tensor of shape (S, S) or (B, 1, S, S).", default: "None" },
];

const forwardParams = [
  { name: "q", type: "torch.Tensor", description: "Query input of shape (B, S, d_model)." },
  { name: "k", type: "torch.Tensor", description: "Key input of shape (B, S, d_model)." },
  { name: "v", type: "torch.Tensor", description: "Value input of shape (B, S, d_model)." },
  { name: "mask", type: "torch.Tensor | None", description: "Optional attention mask.", default: "None" },
];

export default function ApiAttention() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.attention" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.attention</h1>
        
        <p className="text-lg text-muted-foreground">
          Implementation of scaled dot-product multi-head attention.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module implements multi-head self-attention, the core mechanism that allows 
          the transformer to weigh the importance of different positions in the input 
          sequence. The attention computation is parallelized across multiple heads, 
          each operating on a different subspace of the input.
        </p>

        <h2 id="multiheadattention">MultiHeadAttention</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> MultiHeadAttention(nn.Module)
        </div>

        <p>
          Multi-head attention mechanism with scaled dot-product attention, as described 
          in "Attention Is All You Need" (Vaswani et al., 2017).
        </p>

        <h3 id="constructor">Constructor</h3>

        <CodeBlock
          language="python"
          code={`def __init__(self, d_model: int, h: int, dropout: float)`}
        />

        <ApiTable parameters={constructorParams} title="Parameters" />

        <p><strong>Attributes:</strong></p>
        <ul>
          <li><code>W_Q</code> — Linear projection for queries <code>(d_model → d_model)</code></li>
          <li><code>W_K</code> — Linear projection for keys <code>(d_model → d_model)</code></li>
          <li><code>W_V</code> — Linear projection for values <code>(d_model → d_model)</code></li>
          <li><code>W_O</code> — Output projection <code>(d_model → d_model)</code></li>
          <li><code>dropout</code> — Dropout layer for attention weights</li>
          <li><code>d_k</code> — Dimension per head <code>(d_model // h)</code></li>
        </ul>

        <CodeBlock
          filename="transformer/attention.py"
          language="python"
          code={`class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=True)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)`}
        />

        <Callout type="warning" title="Constraint">
          <code>d_model</code> must be evenly divisible by <code>h</code>. If 
          <code>d_model=384</code> and <code>h=6</code>, each head operates on 
          64-dimensional subspaces.
        </Callout>

        <h3 id="get-attention-scores">get_attention_scores()</h3>

        <CodeBlock
          language="python"
          code={`def get_attention_scores(
    self, 
    query: torch.Tensor, 
    key: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor`}
        />

        <ApiTable parameters={attentionScoreParams} title="Parameters" />

        <p><strong>Returns:</strong></p>
        <ul>
          <li><code>torch.Tensor</code> — Attention weights of shape <code>(B, h, S, S)</code></li>
        </ul>

        <CodeBlock
          language="python"
          code={`def get_attention_scores(self, query, key, mask=None):
    """
    Compute scaled dot-product attention scores.
    
    Attention(Q, K) = softmax(QK^T / sqrt(d_k) + mask)
    """
    d_k = query.size(-1)
    
    # (B, h, S, d_k) @ (B, h, d_k, S) -> (B, h, S, S)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask  # mask contains -inf for blocked positions
    
    attention_weights = F.softmax(scores, dim=-1)
    return attention_weights`}
        />

        <h3 id="forward">forward()</h3>

        <CodeBlock
          language="python"
          code={`def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor`}
        />

        <ApiTable parameters={forwardParams} title="Parameters" />

        <p><strong>Returns:</strong></p>
        <ul>
          <li><code>torch.Tensor</code> — Output of shape <code>(B, S, d_model)</code></li>
        </ul>

        <CodeBlock
          language="python"
          code={`def forward(self, q, k, v, mask=None):
    """
    Multi-head attention forward pass.
    
    1. Project inputs through W_Q, W_K, W_V
    2. Split into multiple heads
    3. Compute attention for each head
    4. Concatenate heads and project through W_O
    """
    B, S, _ = q.size()
    
    # Project and reshape: (B, S, d_model) -> (B, h, S, d_k)
    Q = self.W_Q(q).view(B, S, self.h, self.d_k).transpose(1, 2)
    K = self.W_K(k).view(B, S, self.h, self.d_k).transpose(1, 2)
    V = self.W_V(v).view(B, S, self.h, self.d_k).transpose(1, 2)
    
    # Compute attention scores
    attn_weights = self.get_attention_scores(Q, K, mask)
    attn_weights = self.dropout(attn_weights)
    
    # Apply attention to values: (B, h, S, S) @ (B, h, S, d_k) -> (B, h, S, d_k)
    context = torch.matmul(attn_weights, V)
    
    # Concatenate heads: (B, h, S, d_k) -> (B, S, d_model)
    context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
    
    # Final projection
    output = self.W_O(context)
    
    return output`}
        />

        <h2 id="tensor-shapes">Tensor Shapes</h2>

        <div className="overflow-x-auto">
          <table className="api-table">
            <thead>
              <tr>
                <th>Variable</th>
                <th>Shape</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>q, k, v</code> (input)</td>
                <td><code>(B, S, d_model)</code></td>
                <td>Query, key, value inputs</td>
              </tr>
              <tr>
                <td><code>Q, K, V</code> (projected)</td>
                <td><code>(B, h, S, d_k)</code></td>
                <td>After projection and head split</td>
              </tr>
              <tr>
                <td><code>scores</code></td>
                <td><code>(B, h, S, S)</code></td>
                <td>Raw attention scores</td>
              </tr>
              <tr>
                <td><code>attn_weights</code></td>
                <td><code>(B, h, S, S)</code></td>
                <td>Softmax attention weights</td>
              </tr>
              <tr>
                <td><code>context</code></td>
                <td><code>(B, S, d_model)</code></td>
                <td>Attended values (concatenated)</td>
              </tr>
              <tr>
                <td><code>output</code></td>
                <td><code>(B, S, d_model)</code></td>
                <td>Final output after W_O</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h2 id="masking-behavior">Masking Behavior</h2>
        <p>
          The mask tensor prevents attention to certain positions. For causal (autoregressive) 
          attention, the mask is an upper triangular matrix with <code>-inf</code> above the 
          diagonal:
        </p>

        <CodeBlock
          language="python"
          code={`# Example causal mask for sequence length 4:
# [  0, -inf, -inf, -inf ]
# [  0,    0, -inf, -inf ]
# [  0,    0,    0, -inf ]
# [  0,    0,    0,    0 ]

# After adding to scores and softmax:
# Position 0 can only attend to position 0
# Position 1 can attend to positions 0, 1
# Position 2 can attend to positions 0, 1, 2
# Position 3 can attend to all positions`}
        />

        <h2 id="dropout-behavior">Dropout Behavior</h2>
        <p>
          Dropout is applied to attention weights after softmax but before multiplying 
          with values. This randomly zeros out attention connections during training, 
          acting as a regularizer.
        </p>

        <Callout type="info" title="Training vs Inference">
          Dropout is automatically disabled during <code>model.eval()</code> mode. 
          Always call <code>model.eval()</code> before inference to ensure deterministic 
          attention patterns.
        </Callout>
      </div>
    </DocLayout>
  );
}
