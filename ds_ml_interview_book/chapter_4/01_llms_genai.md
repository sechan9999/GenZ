# Chapter 4: LLMs & Generative AI

## Transformer Architecture

### Q1: Explain the Transformer architecture. Why did it replace RNNs?

**A:** Transformers use **self-attention** to process sequences in parallel, unlike RNNs which process sequentially.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Attention(Q, K, V) = softmax(QK^T / √d_k)V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        return output, attention
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        return self.W_o(x)
```

**Why Transformers won:**

| RNN/LSTM | Transformer |
|---|---|
| Sequential (slow) | Parallel (fast) |
| Gradient vanishing | Direct connections via attention |
| Fixed memory | Attends to entire context |
| O(n) time complexity | O(n²) but parallelizable |

**Interview insight:** "The key innovation is self-attention lets every token directly attend to every other token in one step, whereas RNNs need O(n) steps to propagate information across the sequence. This parallelization enabled training on massive datasets with GPUs."

---

### Q2: How does prompt engineering work? Give a framework for designing prompts.

**A:**

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

## Framework: CRAFT
# Context, Role, Action, Format, Tone

def create_prompt(context, role, action, format, tone="professional"):
    prompt = f"""
Context: {context}

Role: You are {role}.

Task: {action}

Output format: {format}

Tone: {tone}

Let's think step by step.
"""
    return prompt

# Example 1: Data extraction
prompt = create_prompt(
    context="We have customer reviews that need sentiment analysis",
    role="an expert data scientist",
    action="Extract the sentiment (positive/negative/neutral) and key entities (product, feature) from this review",
    format="JSON with fields: sentiment, entities, confidence",
    tone="technical"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

# Example 2: Few-shot prompting (best for structured tasks)
few_shot_prompt = """
Extract product name and price from text. Format: JSON

Examples:
Input: "Buy iPhone 14 for $799 today!"
Output: {"product": "iPhone 14", "price": 799}

Input: "MacBook Pro M3 on sale - $1999"
Output: {"product": "MacBook Pro M3", "price": 1999}

Now extract from this text:
Input: "Get the Samsung Galaxy S24 for only $899!"
Output:"""

# Example 3: Chain-of-thought (best for reasoning)
cot_prompt = """
Question: A store has 15 apples. They sell 40% and then buy 12 more. How many do they have?

Let's solve this step by step:
1. First, calculate 40% of 15
2. Subtract from original amount
3. Add the 12 new apples
4. Provide final answer
"""

# Example 4: Role-based prompting
role_prompt = """
<role>You are a senior ML engineer conducting a code review.</role>

<task>Review this code for bugs, performance issues, and best practices:</task>

<code>
def train_model(data):
    model = RandomForest()
    model.fit(data)
    return model
</code>

<instructions>
1. Identify issues
2. Suggest fixes
3. Rate severity (high/medium/low)
</instructions>
"""
```

**Best practices:**
- ✅ Be specific (not "summarize this", but "summarize in 3 bullet points under 50 words each")
- ✅ Give examples (few-shot) for structured outputs
- ✅ Use delimiters (```, XML tags) to separate instructions from data
- ✅ Request step-by-step reasoning for complex tasks
- ❌ Don't assume the model has access to real-time data or external tools (unless using tool use API)

---

### Q3: Explain RAG (Retrieval-Augmented Generation). When would you use it vs fine-tuning?

**A:**

```python
from anthropic import Anthropic
import chromadb
from sentence_transformers import SentenceTransformer

## RAG Pipeline

# Step 1: Build vector database
client = chromadb.Client()
collection = client.create_collection("company_docs")

# Documents (e.g., company policies, documentation)
documents = [
    "Our return policy allows returns within 30 days.",
    "Shipping is free for orders over $50.",
    "We offer 24/7 customer support.",
]

# Embed and store
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents)

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Step 2: Retrieval function
def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results['documents'][0]

# Step 3: Generation with context
def rag_answer(question):
    # Retrieve relevant documents
    context_docs = retrieve_context(question)
    context = "\n".join(context_docs)
    
    # Generate answer
    anthropic_client = Anthropic(api_key="your-key")
    
    prompt = f"""
Context from company documentation:
{context}

Question: {question}

Answer the question using ONLY the provided context. If the answer is not in the context, say "I don't have that information."
"""
    
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Usage
answer = rag_answer("What is your return policy?")
print(answer)
```

**RAG vs Fine-tuning:**

| Aspect | RAG | Fine-tuning |
|---|---|---|
| **When to use** | External knowledge, frequently updated | Task-specific behavior, style |
| **Data needed** | Document corpus | Labeled examples (1k-100k+) |
| **Cost** | Embedding + retrieval + inference | Training compute + inference |
| **Update frequency** | Real-time (just add docs) | Requires retraining |
| **Transparency** | Can cite sources | Black box |
| **Latency** | Higher (retrieval + generation) | Lower (just generation) |
| **Example use case** | Customer support, Q&A over docs | Code generation, style transfer |

**My recommendation:**
- **RAG first** for most applications (company knowledge bases, documentation, customer support)
- **Fine-tuning** when you need specific behavior that can't be prompted (medical report writing, specific code style)
- **Both together** for production systems (RAG for knowledge, fine-tuning for format/style)

---

### Q4: How do you evaluate LLM outputs? What metrics would you use?

**A:**

```python
## Metric 1: Perplexity (for language models)
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss
        
        nlls.append(nll)
    
    perplexity = torch.exp(torch.stack(nlls).mean())
    return perplexity.item()

# Lower perplexity = better

## Metric 2: BLEU/ROUGE (for generation quality)
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps over the lazy dog"
candidate = "A fast brown fox leaps over a sleeping dog"

# BLEU (for translation, summarization)
bleu = sentence_bleu([reference.split()], candidate.split())

# ROUGE (for summarization)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, candidate)

## Metric 3: Embedding similarity (semantic similarity)
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

embedding1 = model.encode(reference, convert_to_tensor=True)
embedding2 = model.encode(candidate, convert_to_tensor=True)

cosine_sim = util.cos_sim(embedding1, embedding2).item()
print(f"Semantic similarity: {cosine_sim:.3f}")

## Metric 4: LLM-as-Judge (best for open-ended tasks)
from anthropic import Anthropic

def llm_judge(question, answer, criteria):
    client = Anthropic(api_key="your-key")
    
    prompt = f"""
You are evaluating an AI assistant's response.

Question: {question}
Answer: {answer}

Evaluation criteria:
{criteria}

Rate the response on a scale of 1-10 and provide reasoning.

Format:
Score: [1-10]
Reasoning: [explain]
"""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Usage
criteria = """
- Accuracy: Is the answer factually correct?
- Completeness: Does it address all parts of the question?
- Clarity: Is it easy to understand?
"""

judgment = llm_judge(
    question="What is the capital of France?",
    answer="The capital of France is Paris, one of the most visited cities in the world.",
    criteria=criteria
)

## Metric 5: Human evaluation (gold standard)
# Implement A/B testing, user ratings, expert review
```

**Production evaluation stack:**
1. **Offline metrics** (before deployment):
   - Perplexity, BLEU/ROUGE for baseline
   - Embedding similarity for semantic correctness
   - LLM-as-judge for complex criteria
   
2. **Online metrics** (in production):
   - User thumbs up/down rates
   - Task completion rate
   - Time to resolution
   - Human spot-checks (random sampling)

---

### Q5: What is the difference between GPT, BERT, and Claude architectures?

**A:**

| Model | Architecture | Training | Use Case |
|---|---|---|---|
| **GPT (Generative)** | Decoder-only | Causal LM (predict next token) | Text generation, chat |
| **BERT (Bidirectional)** | Encoder-only | Masked LM (predict masked tokens) | Classification, NER, Q&A |
| **Claude / Llama** | Decoder-only + RLHF | Causal LM + human feedback | Conversational AI, reasoning |

```python
# GPT-style (autoregressive generation)
# Input:  "The cat sat on the"
# Output: "mat" (predicts NEXT token)
# Attention: Can only see LEFT context (causal mask)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode("The cat sat on the", return_tensors='pt')
output = model.generate(input_ids, max_length=20)
print(tokenizer.decode(output[0]))  # "The cat sat on the mat and looked around."

# BERT-style (masked language modeling)
# Input:  "The cat [MASK] on the mat"
# Output: "sat" (fills in MASKED token)
# Attention: Can see BOTH left and right context (bidirectional)

from transformers import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The cat [MASK] on the mat."
input_ids = tokenizer.encode(text, return_tensors='pt')
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(predicted_token)  # "sat"
```

**When to use each:**
- **GPT-style (Claude, GPT-4, Llama)**: Text generation, chatbots, creative writing, code generation
- **BERT-style**: Classification, named entity recognition, sentiment analysis, question answering (extractive)
- **Hybrid (T5, BART)**: Summarization, translation (encoder for input, decoder for output)

---

### Q6: Explain fine-tuning vs prompt tuning vs in-context learning

**A:**

```python
## 1. In-context learning (no training)
# Just provide examples in the prompt

prompt = """
Classify sentiment as positive, negative, or neutral.

Review: "This product is amazing!"
Sentiment: positive

Review: "Terrible quality, waste of money."
Sentiment: negative

Review: "It works as expected."
Sentiment: neutral

Review: "I love this so much!"
Sentiment:"""

# No gradient updates, just clever prompting

## 2. Prompt tuning (train soft prompts)
# Freeze model weights, only train the prompt embeddings

import torch
import torch.nn as nn

class PromptTuning(nn.Module):
    def __init__(self, model, n_prompt_tokens=20, embed_dim=768):
        super().__init__()
        self.model = model
        
        # Learnable prompt embeddings
        self.soft_prompt = nn.Parameter(torch.randn(n_prompt_tokens, embed_dim))
        
        # Freeze model weights
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_embeds):
        # Prepend soft prompt to input
        batch_size = input_embeds.size(0)
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        
        inputs_with_prompt = torch.cat([soft_prompt_batch, input_embeds], dim=1)
        return self.model(inputs_embeds=inputs_with_prompt)

# Train only self.soft_prompt (~1KB of parameters vs billions)

## 3. Fine-tuning (train model weights)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()  # Updates ALL model weights
```

**Comparison:**

| Method | Parameters trained | Data needed | Cost | Performance |
|---|---|---|---|---|
| In-context learning | 0 | 0-10 examples | $0.001/request | Good |
| Prompt tuning | ~1KB | 100-1k examples | $10-100 | Better |
| LoRA (efficient FT) | ~1MB | 1k-10k examples | $100-1k | Best |
| Full fine-tuning | Billions | 10k-1M examples | $10k-100k | Best (but diminishing returns) |

**My production approach:**
1. Try in-context learning first (cheapest, fastest)
2. If not good enough, collect 1k examples and try prompt tuning
3. If still not enough, use LoRA fine-tuning (99% of performance, 1% of cost)
4. Full fine-tuning only for critical applications with massive budgets

---

## Summary: LLM Interview Checklist

| Concept | Key Insight | Interview Signal |
|---|---|---|
| Transformers | Self-attention enables parallelization | "O(n²) attention but parallelizable across sequence" |
| Prompt engineering | CRAFT framework (Context, Role, Action, Format, Tone) | "Few-shot for structured, CoT for reasoning" |
| RAG | Retrieval + generation for external knowledge | "Use RAG for knowledge, fine-tuning for behavior" |
| Evaluation | Mix of automated metrics + human judgment | "LLM-as-judge for offline, user feedback online" |
| GPT vs BERT | Decoder vs encoder architecture | "GPT for generation, BERT for classification" |
| Fine-tuning tiers | In-context → prompt tuning → LoRA → full FT | "Start cheap (prompting), scale up if needed (LoRA)" |
