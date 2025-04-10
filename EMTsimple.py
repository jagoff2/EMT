import os
import math
import time
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Fixed imports for Hugging Face models
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class EnhancedMemoryConfig:
    """Configuration for enhanced memory module"""
    base_model_name: str = "gpt2-medium"  # Using GPT2-medium for balance of capability/size
    memory_dim: int = 256  # Memory representation dimension
    num_memory_slots: int = 64  # Increased number of memory slots
    memory_update_rate: float = 0.5  # Increased update rate for better retention
    num_memory_layers: int = 2  # Number of memory enhancement layers
    attention_heads: int = 4  # Number of attention heads
    dropout_rate: float = 0.1  # Dropout rate for regularization
    use_gru_controller: bool = True  # Use GRU for memory updates
    feedback_strength: float = 0.7  # Increased feedback strength
    use_layernorm: bool = True  # Use layer normalization
    blend_ratio: float = 0.6  # Increased memory influence in output


class MemoryAttentionGate(nn.Module):
    """Attention-based memory gate that controls info flow to/from memory"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        assert memory_dim % num_heads == 0, "Memory dimension must be divisible by number of heads"
        
        # Multi-head attention components
        self.query = nn.Linear(input_dim, memory_dim)
        self.key = nn.Linear(memory_dim, memory_dim)
        self.value = nn.Linear(memory_dim, memory_dim)
        self.output = nn.Linear(memory_dim, input_dim)
        
        # Normalization and dropout
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feedforward network
        self.ff_network = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * input_dim, input_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, memory):
        """
        Process input through memory attention mechanism
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            memory: Memory tensor [batch_size, num_slots, memory_dim]
            
        Returns:
            output: Memory-enhanced representation [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute residual attention
        residual = x
        x = self.layer_norm1(x)
        
        # Project to query, key, value spaces
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(memory).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.value(memory).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, num_slots, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, num_slots, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention-weighted values
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.memory_dim)
        
        # Output projection and dropout
        attn_output = self.output(attn_output)
        attn_output = self.dropout(attn_output)
        
        # First residual connection
        x = residual + attn_output
        
        # Feedforward network with residual connection and layer norm
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.ff_network(x)
        
        return x


class MemoryController(nn.Module):
    """Optimized memory controller for storing and retrieving information"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_slots: int = 64,
        update_rate: float = 0.5,
        use_gru: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.update_rate = update_rate
        self.use_gru = use_gru
        
        # Memory content projections
        self.input_transform = nn.Linear(input_dim, memory_dim)
        self.key_transform = nn.Linear(memory_dim, memory_dim)
        self.value_transform = nn.Linear(input_dim, memory_dim)
        self.query_transform = nn.Linear(input_dim, memory_dim)
        
        # Memory state controller
        if use_gru:
            self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_update = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_reset = nn.Linear(memory_dim * 2, memory_dim)
        else:
            self.memory_update = nn.Linear(memory_dim * 2, memory_dim)
            self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)
        
        # For memory access
        self.content_score = nn.Linear(memory_dim, 1)
        self.age_factor = 0.98  # Decay factor for memory age
        
        # Memory slot management
        self.register_buffer('memory', None, persistent=False)
        self.register_buffer('usage', None, persistent=False)
        self.register_buffer('age', None, persistent=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_memory(self):
        """Reset memory and usage counters"""
        self.memory = None
        self.usage = None
        self.age = None
    
    def _initialize_memory(self, batch_size, device):
        """Initialize memory if not already set"""
        if self.memory is None:
            # Initialize with small random values
            memory = torch.randn(batch_size, self.num_slots, self.memory_dim, device=device) * 0.01
            memory = F.normalize(memory, p=2, dim=-1)  # Initialize on unit hypersphere
            
            # Initialize usage as all zeros (no usage)
            usage = torch.zeros(batch_size, self.num_slots, device=device)
            
            # Initialize age tracking
            age = torch.zeros(batch_size, self.num_slots, device=device)
            
            # Register as buffers
            self.memory = memory
            self.usage = usage
            self.age = age
    
    def _gru_update(self, x, h):
        """Improved GRU update function with better weight control"""
        # Ensure inputs have correct dimensions
        # Concatenate input and hidden state
        concat = torch.cat([x, h], dim=1)
        
        # GRU gates
        reset_gate = torch.sigmoid(self.memory_reset(concat))
        update_gate = torch.sigmoid(self.memory_gate(concat))
        
        # Compute candidate hidden state
        concat_reset = torch.cat([x, reset_gate * h], dim=1)
        candidate = torch.tanh(self.memory_update(concat_reset))
        
        # Update hidden state
        new_h = (1 - update_gate) * h + update_gate * candidate
        
        return new_h
    
    def forward(self, hidden_states):
        """Process hidden states through memory controller with improved information capture"""
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize memory if needed
        self._initialize_memory(batch_size, device)
        
        # Process each sequence step to update memory
        for t in range(seq_len):
            # Current hidden state [batch_size, input_dim]
            current_state = hidden_states[:, t]
            
            # Transform to memory space
            memory_input = self.input_transform(current_state)  # [batch_size, memory_dim]
            
            # Calculate similarity with existing memory
            similarity = torch.bmm(
                memory_input.unsqueeze(1),  # [batch_size, 1, memory_dim]
                self.memory.transpose(1, 2)  # [batch_size, memory_dim, num_slots]
            ).squeeze(1)  # [batch_size, num_slots]
            
            # Adjust similarity by age factor - older memories less likely to be updated
            adjusted_scores = similarity - (self.age * 0.1)
            
            # Add usage bias to prevent overwriting frequently used slots
            usage_bias = self.usage * 0.2
            update_scores = adjusted_scores - usage_bias
            
            # Compute key strength for current input
            values = self.value_transform(current_state)  # [batch_size, memory_dim]
            key_strength = torch.norm(values, dim=1, keepdim=True)
            
            # Apply softmax to get read and write weights
            read_weights = F.softmax(similarity, dim=1)
            write_weights = F.softmax(-update_scores, dim=1)  # Lower score -> higher write weight
            
            # Read from memory
            read_vector = torch.bmm(
                read_weights.unsqueeze(1),  # [batch_size, 1, num_slots]
                self.memory  # [batch_size, num_slots, memory_dim]
            ).squeeze(1)  # [batch_size, memory_dim]
            
            # Update all memory slots (with varying degrees) - truly parallel update
            for b in range(batch_size):
                for i in range(self.num_slots):
                    # Weight for this memory slot
                    write_weight = write_weights[b, i].item()
                    
                    # Only update if the write weight is significant
                    if write_weight > 0.01:  # Threshold for efficiency
                        # Current slot content
                        current_memory = self.memory[b, i]
                        
                        if self.use_gru:
                            # Apply GRU update mechanism
                            new_content = self._gru_update(
                                values[b].unsqueeze(0),  # [1, memory_dim]
                                current_memory.unsqueeze(0)  # [1, memory_dim]
                            )
                            # Weighted update based on write weight
                            self.memory[b, i] = self.memory[b, i] * (1 - write_weight * self.update_rate) + \
                                            new_content.squeeze(0) * (write_weight * self.update_rate)
                        else:
                            # Simple weighted linear update
                            new_content = torch.tanh(self.memory_update(
                                torch.cat([current_memory, values[b]]).unsqueeze(0)
                            )).squeeze(0)
                            self.memory[b, i] = self.memory[b, i] * (1 - write_weight * self.update_rate) + \
                                            new_content * (write_weight * self.update_rate)
                        
                        # Update usage statistics
                        self.usage[b, i] += write_weight
                
                # Normalize memory vectors to prevent explosion
                self.memory[b] = F.normalize(self.memory[b], p=2, dim=1)
            
            # Age all memories
            self.age = self.age * self.age_factor + 1.0
            
            # Decay usage statistics
            self.usage = self.usage * 0.99
        
        # Return full memory state
        return self.memory


class EnhancedMemoryLayer(nn.Module):
    """Improved memory layer that processes all inputs without thresholding"""
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_slots: int = 64,
        num_heads: int = 4,
        update_rate: float = 0.5,
        feedback_strength: float = 0.7,
        use_gru: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.feedback_strength = feedback_strength
        
        # Memory controller for storage
        self.memory_controller = MemoryController(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_slots=num_slots,
            update_rate=update_rate,
            use_gru=use_gru
        )
        
        # Memory attention gate for retrieval and integration
        self.memory_gate = MemoryAttentionGate(
            input_dim=input_dim,
            memory_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Input salience network (for analytics, not filtering)
        self.salience_network = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate distributions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_memory(self):
        """Reset memory state"""
        self.memory_controller.reset_memory()
    
    def forward(self, hidden_states):
        """Process all hidden states through memory layer without thresholding"""
        # Calculate salience of input (for monitoring only)
        salience = self.salience_network(hidden_states)
        
        # ALWAYS process input through memory (removing thresholding)
        # Update memory with current hidden states
        memory = self.memory_controller(hidden_states)
        
        # Retrieve and integrate memory with hidden states
        enhanced_states = self.memory_gate(hidden_states, memory)
        
        # Apply stronger feedback for improved memory integration
        output = hidden_states + self.feedback_strength * (enhanced_states - hidden_states)
        
        return output


class GPT2WithEnhancedMemory(nn.Module):
    """Improved GPT-2 model with better memory integration"""
    def __init__(self, config: EnhancedMemoryConfig):
        super().__init__()
        self.config = config
        
        # Load base GPT-2 model
        logger.info(f"Loading base model: {config.base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        
        # Extract model dimensions
        self.hidden_size = self.model.config.n_embd
        logger.info(f"Model hidden size: {self.hidden_size}")
        
        # Create memory layers
        logger.info(f"Creating {config.num_memory_layers} memory enhancement layers")
        self.memory_layers = nn.ModuleList([
            EnhancedMemoryLayer(
                input_dim=self.hidden_size,
                memory_dim=config.memory_dim,
                num_slots=config.num_memory_slots,
                num_heads=config.attention_heads,
                update_rate=config.memory_update_rate,
                feedback_strength=config.feedback_strength,
                use_gru=config.use_gru_controller,
                dropout=config.dropout_rate
            ) for _ in range(config.num_memory_layers)
        ])
        
        # Hook for capturing hidden states
        self.hidden_states = None
        self._register_hooks()
        
        # Store blend ratio for memory integration
        self.blend_ratio = config.blend_ratio
        
        # Initialize weights
        self._init_weights()
        
        logger.info("Memory-enhanced GPT-2 model initialized successfully")
    
    def _init_weights(self):
        """Initialize new weights with appropriate distributions"""
        for name, module in self.named_children():
            if name == 'memory_layers':
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def _register_hooks(self):
        """Register forward hooks to capture hidden states"""
        def hook_fn(module, input, output):
            # This hook captures the hidden states from the transformer
            if hasattr(output, 'last_hidden_state'):
                self.hidden_states = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                self.hidden_states = output[0]
            else:
                self.hidden_states = output
        
        # Hook the transformer output
        if hasattr(self.model, 'transformer'):
            self.model.transformer.register_forward_hook(hook_fn)
            logger.info("Hook registered on model.transformer")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            self.model.model.decoder.register_forward_hook(hook_fn)
            logger.info("Hook registered on model.model.decoder")
        elif hasattr(self.model, 'gpt2') and hasattr(self.model.gpt2, 'h'):
            # Hook the last transformer block
            self.model.gpt2.h[-1].register_forward_hook(hook_fn)
            logger.info("Hook registered on last transformer block")
        else:
            # Try to find the last layer dynamically
            for name, module in reversed(list(self.model.named_modules())):
                if 'h.' in name and '.mlp' in name:
                    module.register_forward_hook(hook_fn)
                    logger.info(f"Hook registered on {name}")
                    break
    
    def reset_memory(self):
        """Reset all memory layers"""
        for layer in self.memory_layers:
            layer.reset_memory()
        logger.info("Memory state reset")
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, 
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass with improved memory augmentation"""
        # Forward pass through base GPT-2 model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always get hidden states
            return_dict=True,  # Always use return dict for consistency
            **kwargs
        )
        
        # Process hidden states through memory layers
        if self.hidden_states is not None:
            memory_enhanced = self.hidden_states
            logger.debug(f"Hidden states shape: {memory_enhanced.shape}")
            
            # Apply each memory layer in sequence
            for i, layer in enumerate(self.memory_layers):
                memory_enhanced = layer(memory_enhanced)
                logger.debug(f"After memory layer {i+1}: {memory_enhanced.shape}")
            
            # Find appropriate lm_head
            if hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
            elif hasattr(self.model, 'gpt2') and hasattr(self.model.gpt2, 'lm_head'):
                lm_head = self.model.gpt2.lm_head
            else:
                # Find lm_head dynamically
                lm_head = None
                for name, module in self.model.named_modules():
                    if 'lm_head' in name or (isinstance(module, nn.Linear) and 
                                           module.out_features == self.model.config.vocab_size):
                        lm_head = module
                        break
                
                if lm_head is None:
                    logger.warning("Could not find lm_head, using original logits")
                    return outputs
            
            # Generate new logits from enhanced hidden states
            try:
                # Use memory enhanced hidden states to generate logits
                new_logits = lm_head(memory_enhanced)
                
                # Apply increased blend ratio for stronger memory influence
                blended_logits = self.blend_ratio * new_logits + (1.0 - self.blend_ratio) * outputs.logits
                outputs.logits = blended_logits
                logger.debug("Successfully blended logits with memory enhancements")
            except Exception as e:
                logger.error(f"Error generating new logits: {e}")
                # Return original outputs if there's an error
                pass
        
        return outputs


def create_memory_benchmark(
    tokenizer, 
    memory_key="IMPORTANT-MEMORY-CUE-alpha", 
    memory_value="42",
    context_length=200  # Increased context length for more challenge
):
    """Create a more challenging memory benchmark with longer context"""
    # Create context with varying sentence structures
    context_parts = [
        "This is part of the padding context.",
        "We are testing the model's memory capabilities.",
        "Language models often struggle with long-term retention.",
        "The context window can be quite challenging.",
        "Information presented early tends to be forgotten.",
        "Our memory mechanism aims to address this limitation.",
        "Neural networks process information sequentially.",
        "Attention mechanisms help, but aren't perfect.",
        "Let's see if our model can recall important details.",
        "The enhanced architecture should improve performance.",
        "Without explicit memory, retrieval is difficult.",
        "The transformer architecture relies on attention.",
        "Longer contexts present particular challenges.",
        "Key information can be lost in lengthy sequences.",
        "Current models have limited context windows.",
        "Sequential processing affects information retention.",
        "Tokens from earlier in the sequence may be forgotten.",
        "Memory mechanisms provide a solution to this issue.",
        "Retrieval-based approaches can help with recall.",
        "The ability to remember is crucial for language models."
    ]
    
    # Create varied context by sampling from parts
    np.random.shuffle(context_parts)
    context = " ".join(context_parts * ((context_length // len(context_parts)) + 1))[:context_length * 10]
    
    # Create memory cue at the beginning
    memory_cue = f"{memory_key}: The value is {memory_value}."
    
    # Create query at the end
    query = f"What was the value of {memory_key}?"
    
    # Full test input
    test_input = f"{memory_cue} {context} {query}"
    
    # Tokenize
    tokens = tokenizer(
        test_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    
    return test_input, tokens


def evaluate_memory_retrieval(model, tokenizer, test_input, tokens, device, target_value="42"):
    """Evaluate model's memory retrieval capability with improved detection logic"""
    model.eval()
    
    # Move tokens to device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Get target token ID (with space prefix for proper tokenization)
    target_token_id = tokenizer.encode(" " + target_value)[0]
    
    # Get logits for the last position
    logits = outputs.logits[:, -1, :]
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probability for target token
    target_prob = probs[0, target_token_id].item() if target_token_id < probs.size(1) else 0
    
    # Get top predictions
    top_k = 10  # Check for target in more positions
    top_probs, top_indices = torch.topk(probs[0], min(top_k, probs.size(-1)))
    
    top_tokens = []
    for i in range(min(top_k, len(top_indices))):
        token_id = top_indices[i].item()
        token = tokenizer.decode([token_id])
        prob = top_probs[i].item()
        top_tokens.append((token, prob))
    
    # Check if target is in top predictions
    target_in_top = any(target_value in token for token, _ in top_tokens)
    target_position = next((i for i, (token, _) in enumerate(top_tokens) if target_value in token), -1)
    
    # Generate text completion
    gen_tokens = tokenizer(test_input, return_tensors="pt").to(device)
    
    # Custom generation without resetting memory
    def generate_with_memory(model, input_ids, max_new_tokens=30, temperature=0.7):
        model.eval()
        
        # Starting sequence
        generated = input_ids.clone()
        
        try:
            # Generate tokens autoregressively
            for _ in range(max_new_tokens):
                # Forward pass with current sequence
                with torch.no_grad():
                    outputs = model(input_ids=generated)
                
                # Get logits for next token prediction
                next_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p (nucleus) sampling
                next_probs = F.softmax(next_logits, dim=-1)
                
                # Sample from filtered distribution
                next_token = torch.multinomial(next_probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
        
        return generated
    
    # Generate text
    try:
        generated_ids = generate_with_memory(
            model, 
            gen_tokens.input_ids, 
            max_new_tokens=30,
            temperature=0.7
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        generated_text = "Error during text generation"
    
    # IMPROVED DETECTION: Check for target value in different formats
    # 1. Look for exact value
    exact_value_present = target_value in generated_text.split()
    
    # 2. Look for common patterns like "value is X" or "value was X"
    response_start_idx = generated_text.find("What was the value")
    response_text = ""
    if response_start_idx != -1:
        response_text = generated_text[response_start_idx:]
        
    value_pattern_present = any([
        f"value is {target_value}" in response_text,
        f"value was {target_value}" in response_text,
        f"value: {target_value}" in response_text,
        f"answer is {target_value}" in response_text,
        f"answer was {target_value}" in response_text
    ])
    
    # Combined detection
    value_in_generation = exact_value_present or value_pattern_present
    
    # Calculate response quality with more nuance
    if response_start_idx != -1:
        if value_pattern_present:
            response_quality = 5  # Direct answer with pattern
        elif exact_value_present and response_text.find(target_value) > 0:
            response_quality = 4  # Value appears in response but not in pattern
        else:
            response_quality = 1  # Response exists but no value
    else:
        response_quality = 0  # No response detected
    
    # Additional verification - extract and save the actual answer
    actual_answer = "Not found"
    if response_start_idx != -1:
        # Try to extract the answer after the question
        answer_segment = response_text[:100]  # Take first 100 chars after question
        logger.info(f"Answer segment: {answer_segment}")
        actual_answer = answer_segment
    
    # Compile detailed results
    results = {
        'target_prob': target_prob,
        'top_k_tokens': top_tokens,
        'target_in_top': target_in_top,
        'target_position': target_position,
        'generated_text': generated_text,
        'value_in_generation': value_in_generation,
        'response_quality': response_quality,
        'actual_answer': actual_answer
    }
    
    return results


def demo_enhanced_memory():
    """Demonstrate enhanced memory capabilities with GPT-2"""
    print("\n=== Enhanced Memory GPT-2 Demonstration ===\n")
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize configuration
        config = EnhancedMemoryConfig(
            base_model_name="gpt2-medium",  # Start with medium for balance
            memory_dim=256,
            num_memory_slots=64,  # Increased slots
            memory_update_rate=0.5,  # Increased update rate
            num_memory_layers=2,
            attention_heads=4,
            dropout_rate=0.1,
            use_gru_controller=True,
            feedback_strength=0.7,  # Increased feedback
            blend_ratio=0.6  # Increased memory influence
        )
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create enhanced memory model
        print(f"Creating enhanced memory model with {config.base_model_name}...")
        enhanced_model = GPT2WithEnhancedMemory(config).to(device)
        
        # Create baseline model for comparison
        print("Creating baseline model for comparison...")
        baseline_model = AutoModelForCausalLM.from_pretrained(config.base_model_name).to(device)
        
        # Create memory benchmark
        print("\nCreating memory benchmark...")
        test_input, tokens = create_memory_benchmark(
            tokenizer,
            memory_key="IMPORTANT-MEMORY-CUE-alpha",
            memory_value="42",
            context_length=200  # Increased context length
        )
        
        print(f"Memory benchmark created with context length: 200")
        
        # Reset memory once at the beginning, not during evaluation
        if hasattr(enhanced_model, 'reset_memory'):
            enhanced_model.reset_memory()
            print("Memory state reset once at the beginning")
        
        # Evaluate baseline model
        print("Evaluating baseline model...")
        baseline_results = evaluate_memory_retrieval(
            baseline_model,
            tokenizer,
            test_input,
            tokens,
            device,
            target_value="42"
        )
        
        # Evaluate enhanced model
        print("Evaluating enhanced memory model...")
        enhanced_results = evaluate_memory_retrieval(
            enhanced_model,
            tokenizer,
            test_input,
            tokens,
            device,
            target_value="42"
        )
        
        # Print comprehensive results
        print("\n=== Memory Retrieval Results ===")
        
        print("\nBaseline Model:")
        print(f"Probability of '42': {baseline_results['target_prob']:.6f}")
        print(f"Target appears in top tokens: {'Yes' if baseline_results['target_in_top'] else 'No'}")
        if baseline_results['target_in_top']:
            print(f"Target position in predictions: {baseline_results['target_position'] + 1}")
        print("\nTop 5 token predictions:")
        for i, (token, prob) in enumerate(baseline_results['top_k_tokens'][:5]):
            print(f"  {i+1}. {token!r}: {prob:.6f}")
        
        print("\nEnhanced Memory Model:")
        print(f"Probability of '42': {enhanced_results['target_prob']:.6f}")
        print(f"Target appears in top tokens: {'Yes' if enhanced_results['target_in_top'] else 'No'}")
        if enhanced_results['target_in_top']:
            print(f"Target position in predictions: {enhanced_results['target_position'] + 1}")
        print("\nTop 5 token predictions:")
        for i, (token, prob) in enumerate(enhanced_results['top_k_tokens'][:5]):
            print(f"  {i+1}. {token!r}: {prob:.6f}")
        
        # Compare performance
        print("\nPerformance Comparison:")
        prob_diff = enhanced_results['target_prob'] - baseline_results['target_prob']
        prob_ratio = enhanced_results['target_prob'] / max(baseline_results['target_prob'], 1e-10)
        
        print(f"Absolute improvement: {prob_diff:.6f}")
        print(f"Relative improvement: {(prob_ratio - 1) * 100:.2f}%")
        
        if enhanced_results['target_in_top'] and not baseline_results['target_in_top']:
            print("✅ Enhanced model has target in top predictions while baseline does not")
        elif baseline_results['target_in_top'] and not enhanced_results['target_in_top']:
            print("❌ Baseline model has target in top predictions while enhanced model does not")
        elif enhanced_results['target_in_top'] and baseline_results['target_in_top']:
            position_diff = baseline_results['target_position'] - enhanced_results['target_position']
            print(f"Both models have target in top predictions (position improvement: {position_diff})")
        else:
            print("Neither model has target in top predictions")
        
        # Compare generated text
        print("\n=== Text Generation Results ===")
        
        print("\nBaseline Model Output:")
        print(baseline_results['generated_text'])
        
        print("\nEnhanced Memory Model Output:")
        print(enhanced_results['generated_text'])
        
        # Check if target value appears in generated text
        if enhanced_results['value_in_generation'] and not baseline_results['value_in_generation']:
            print("\n✅ Enhanced model successfully generated the target value while baseline did not")
        elif baseline_results['value_in_generation'] and not enhanced_results['value_in_generation']:
            print("\n❌ Baseline model generated the target value while enhanced model did not")
        elif enhanced_results['value_in_generation'] and baseline_results['value_in_generation']:
            print("\n✓ Both models generated the target value")
        else:
            print("\n❌ Neither model generated the target value explicitly")
        

        # Check if target value appears in generated text - with improved detection
        print("\nDetailed Answer Analysis:")

        print("\nBaseline Model:")
        print(f"Actual answer segment: {baseline_results['actual_answer']}")
        print(f"Contains target value: {'Yes' if baseline_results['value_in_generation'] else 'No'}")

        print("\nEnhanced Memory Model:")
        print(f"Actual answer segment: {enhanced_results['actual_answer']}")
        print(f"Contains target value: {'Yes' if enhanced_results['value_in_generation'] else 'No'}")

        # Improved check for value in generation
        if enhanced_results['value_in_generation'] and not baseline_results['value_in_generation']:
            print("\n✅ Enhanced model successfully retrieved the target value while baseline did not")
        elif baseline_results['value_in_generation'] and not enhanced_results['value_in_generation']:
            print("\n❌ Baseline model retrieved the target value while enhanced model did not")
        elif enhanced_results['value_in_generation'] and baseline_results['value_in_generation']:
            # Compare response quality
            if enhanced_results['response_quality'] > baseline_results['response_quality']:
                print("\n✅ Both models retrieved the value, but enhanced model had higher quality response")
            elif baseline_results['response_quality'] > enhanced_results['response_quality']:
                print("\n✓ Both models retrieved the value, but baseline model had higher quality response")
            else:
                print("\n✓ Both models retrieved the target value with similar quality")
        else:
            print("\n❌ Neither model retrieved the target value")

        print("\n=== Demonstration completed successfully ===")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting enhanced memory demonstration...")
    demo_enhanced_memory()