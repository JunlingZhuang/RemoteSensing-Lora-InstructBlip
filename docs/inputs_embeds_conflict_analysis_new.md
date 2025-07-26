# InstructBLIP inputs_embeds Conflict Analysis

## The Problem

When trying to fine-tune InstructBLIP with LoRA, I kept hitting this frustrating parameter conflict with `inputs_embeds`. Turns out it's a fundamental incompatibility between how HuggingFace's InstructBLIP works and what LoRA expects.

## The Error Messages

Depending on your transformers version, you'll see different errors:

**transformers 4.53.0.dev0:**
```
TypeError: T5ForConditionalGeneration got multiple values for keyword argument 'inputs_embeds'
```

**transformers 4.44.0:**
```
TypeError: InstructBlipForConditionalGeneration.forward() got an unexpected keyword argument 'inputs_embeds'
```

**transformers 4.37.0:**
```
Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper
```

## Why This Happens

### The InstructBLIP Pipeline

InstructBLIP has this internal flow:

1. **Vision encoder** processes the image
2. **Q-Former** combines image features with text instruction
3. **Language model** (T5) generates the response

The problem is in step 3. InstructBLIP's forward method generates `inputs_embeds` from `input_ids` internally, but when you use LoRA, the training loop also tries to pass `inputs_embeds` directly.

### The Conflict

```python
# What InstructBLIP does internally:
inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

# What LoRA training tries to do:
model.forward(..., inputs_embeds=some_embeddings)

# Result: T5 gets both input_ids AND inputs_embeds = conflict
```

## The Solution: Custom Forward Pass

Since HuggingFace's implementation is fundamentally incompatible with LoRA, I had to implement a custom forward pass that manually chains the components:

```python
def custom_instructblip_forward(model, pixel_values, qformer_input_ids, input_ids, labels):
    # Step 1: Vision encoding
    vision_outputs = model.vision_model(pixel_values)
    image_embeds = vision_outputs.last_hidden_state
    
    # Step 2: Q-Former processing
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=torch.ones(image_embeds.shape[:-1]),
        input_ids=qformer_input_ids,
        return_dict=True
    )
    
    # Step 3: Language projection
    language_model_inputs = model.language_projection(query_outputs.last_hidden_state)
    
    # Step 4: Get text embeddings
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    
    # Step 5: Concatenate and generate
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
    
    # Step 6: Call T5 with ONLY inputs_embeds (no input_ids)
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        return_dict=True
    )
    
    return outputs
```

## Key Implementation Details

### 1. Bypass HuggingFace's Forward

Never call `model.forward()` directly. Always use the custom implementation.

### 2. Manual Component Chaining

Each step is done manually to avoid parameter conflicts:
- Vision model → image features
- Q-Former → query embeddings
- Language projection → projected features
- Text embedding → text features
- Concatenation → final input
- T5 generation → output

### 3. LoRA Integration

The custom forward works with LoRA because:
- LoRA adapters are applied to Q-Former components automatically
- T5 only receives `inputs_embeds` (no `input_ids` conflict)
- All gradients flow correctly through LoRA parameters

### 4. Numerical Validation

Added extensive checks throughout:

```python
# Check for NaN/Inf at each step
if torch.any(torch.isnan(query_outputs.last_hidden_state)):
    print("Warning: NaN detected in Q-Former outputs")
    return None

if torch.any(torch.isinf(inputs_embeds)):
    print("Warning: Inf detected in inputs_embeds")
    return None
```

## Performance Impact

The custom forward pass has minimal overhead:
- Same computational complexity as original
- Slightly more memory usage due to intermediate tensors
- No significant speed difference in practice

## Alternative Approaches (That Didn't Work)

### 1. Monkey Patching
Tried modifying HuggingFace's forward method directly. Too fragile and version-dependent.

### 2. Parameter Filtering
Attempted to filter out conflicting parameters before calling forward. Didn't work because the conflict is internal.

### 3. Different LoRA Targets
Tried applying LoRA only to specific modules. Still had the same parameter passing issue.

## Lessons Learned

1. **HuggingFace compatibility isn't guaranteed** - Sometimes you need custom implementations
2. **LoRA integration can be tricky** - Parameter passing conflicts are common
3. **Manual component chaining works** - When the high-level API fails, go low-level
4. **Extensive validation is crucial** - Numerical issues are hard to debug without checks

## Current Status

The custom forward implementation is stable and works reliably with:
- LoRA fine-tuning
- Multiple batch sizes
- Different LoRA configurations
- Both training and inference

It's now the standard approach used in all training scripts.

---

*This analysis is based on extensive debugging across transformers versions 4.37.0, 4.44.0, and 4.53.0.dev0*
