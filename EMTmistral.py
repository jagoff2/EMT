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

# Imports for Hugging Face models
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
    base_model_name: str = "mistralai/Mistral-7B-v0.1"  # Using Mistral 7B
    hf_token: str = None  # Hugging Face token for authentication
    memory_dim: int = 256  # Memory representation dimension
    num_memory_slots: int = 64  # Number of memory slots
    memory_update_rate: float = 0.5  # Rate at which memory is updated
    num_memory_layers: int = 2  # Number of memory enhancement layers
    attention_heads: int = 4  # Number of attention heads
    dropout_rate: float = 0.1  # Dropout rate for regularization
    use_gru_controller: bool = True  # Use GRU for memory updates
    feedback_strength: float = 0.7  # Strength of memory feedback
    blend_ratio: float = 0.6  # Memory influence in output
    use_8bit_quantization: bool = False  # Disable 8-bit quantization for CPU
    fallback_model: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Fallback model if primary fails
    force_cpu: bool = True  # Force CPU execution


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
        # Store original dtype for conversion back at the end
        orig_dtype = x.dtype
        
        # Handle input dtype if necessary
        if x.dtype != torch.float32:
            x = x.float()
            memory = memory.float()
        
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
        
        # Convert back to original dtype
        if x.dtype != orig_dtype:
            x = x.to(dtype=orig_dtype)
        
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
    
    def _initialize_memory(self, batch_size, device, dtype=torch.float32):
        """Initialize memory if not already set"""
        if self.memory is None:
            # Initialize with small random values
            memory = torch.randn(batch_size, self.num_slots, self.memory_dim, device=device, dtype=dtype) * 0.01
            memory = F.normalize(memory, p=2, dim=-1)  # Initialize on unit hypersphere
            
            # Initialize usage as all zeros (no usage)
            usage = torch.zeros(batch_size, self.num_slots, device=device, dtype=dtype)
            
            # Initialize age tracking
            age = torch.zeros(batch_size, self.num_slots, device=device, dtype=dtype)
            
            # Register as buffers
            self.memory = memory
            self.usage = usage
            self.age = age
    
    def _gru_update(self, x, h):
        """Improved GRU update function with better weight control"""
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
        # Store original dtype for conversion back at the end
        orig_dtype = hidden_states.dtype
        
        # Handle input dtype if necessary
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize memory with the correct dtype
        self._initialize_memory(batch_size, device, dtype=hidden_states.dtype)
        
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
        
        # Convert back to original dtype if needed
        if self.memory.dtype != orig_dtype:
            memory_output = self.memory.to(dtype=orig_dtype)
            return memory_output
        
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
        # Store original dtype
        orig_dtype = hidden_states.dtype
        
        # Handle input dtype if necessary
        if hidden_states.dtype != torch.float32:
            hidden_states_float = hidden_states.float()
            # Calculate salience of input (for monitoring only)
            salience = self.salience_network(hidden_states_float)
            
            # Update memory with current hidden states
            memory = self.memory_controller(hidden_states_float)
            
            # Retrieve and integrate memory with hidden states
            enhanced_states = self.memory_gate(hidden_states_float, memory)
            
            # Apply stronger feedback for improved memory integration
            output = hidden_states_float + self.feedback_strength * (enhanced_states - hidden_states_float)
            
            # Convert back to original precision
            return output.to(dtype=orig_dtype)
        else:
            # Process in native precision
            salience = self.salience_network(hidden_states)
            memory = self.memory_controller(hidden_states)
            enhanced_states = self.memory_gate(hidden_states, memory)
            output = hidden_states + self.feedback_strength * (enhanced_states - hidden_states)
            return output


class MistralWithEnhancedMemory(nn.Module):
    """Mistral model with enhanced memory mechanisms"""
    def __init__(self, config: EnhancedMemoryConfig):
        super().__init__()
        self.config = config
        
        # Load base Mistral model with authentication and CPU
        logger.info(f"Loading base model: {config.base_model_name}")
        
        try:
            # Prepare model loading kwargs
            model_kwargs = {}
            
            # For CPU-only execution
            if config.force_cpu:
                model_kwargs["device_map"] = "cpu"
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Forcing CPU execution with float32 precision")
            
            # Add authentication token if provided
            if config.hf_token:
                model_kwargs["token"] = config.hf_token
                logger.info("Using provided Hugging Face token for authentication")
            
            # Load the model with lower memory usage
            model_kwargs["low_cpu_mem_usage"] = True
            
            # Load in half precision first to reduce memory, then convert back if needed
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name, 
                **model_kwargs
            )
            logger.info(f"Successfully loaded model: {config.base_model_name}")
        
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            
            # Try fallback model if provided
            if config.fallback_model:
                logger.info(f"Attempting to load fallback model: {config.fallback_model}")
                try:
                    model_kwargs["torch_dtype"] = torch.float32  # Ensure float32 for CPU
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.fallback_model,
                        **model_kwargs
                    )
                    logger.info(f"Successfully loaded fallback model: {config.fallback_model}")
                except Exception as fallback_err:
                    logger.error(f"Failed to load fallback model: {fallback_err}")
                    raise RuntimeError("Failed to load both primary and fallback models")
            else:
                # Re-raise the original exception if no fallback
                raise
        
        # Extract model dimensions
        self.hidden_size = self.model.config.hidden_size
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
        
        # Move memory layers to CPU
        if config.force_cpu:
            self.memory_layers = self.memory_layers.to("cpu")
        
        # Hook for capturing hidden states
        self.hidden_states = None
        self._register_hooks()
        
        # Store blend ratio for memory integration
        self.blend_ratio = config.blend_ratio
        
        # Initialize weights
        self._init_weights()
        
        logger.info("Memory-enhanced Mistral model initialized successfully")
    
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
            # This hook captures the hidden states
            if hasattr(output, 'last_hidden_state'):
                self.hidden_states = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                self.hidden_states = output[0]
            else:
                self.hidden_states = output
        
        # Try different hook placements to handle model architecture variations
        try:
            # First, try hooking onto the norm layer which is typically at the end
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                self.model.model.norm.register_forward_hook(hook_fn)
                logger.info("Hook registered on model.model.norm")
            # Next, try hooking onto the last transformer layer
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                last_layer_idx = len(self.model.model.layers) - 1
                self.model.model.layers[last_layer_idx].register_forward_hook(hook_fn)
                logger.info(f"Hook registered on last transformer layer {last_layer_idx}")
            else:
                # If the above methods fail, search more thoroughly
                found_hook = False
                
                # Look for the final layer norm (often used in transformer architectures)
                for name, module in reversed(list(self.model.named_modules())):
                    if ('norm' in name or 'ln_f' in name) and isinstance(module, nn.LayerNorm):
                        module.register_forward_hook(hook_fn)
                        logger.info(f"Hook registered on {name}")
                        found_hook = True
                        break
                
                # If still not found, try to find the last layer of any transformer blocks
                if not found_hook:
                    for name, module in reversed(list(self.model.named_modules())):
                        if ('layers' in name and '.output' in name) or ('layers' in name and '.feed_forward' in name):
                            module.register_forward_hook(hook_fn)
                            logger.info(f"Hook registered on {name}")
                            found_hook = True
                            break
                
                # Last resort: hook onto the model itself
                if not found_hook:
                    logger.warning("Could not find appropriate layer for hook, falling back to model")
                    self.model.register_forward_hook(hook_fn)
                    logger.info("Fallback hook registered directly on model")
        
        except Exception as e:
            logger.error(f"Failed to register hook: {e}")
            # Fallback: hook directly onto the model
            self.model.register_forward_hook(hook_fn)
            logger.info("Fallback hook registered due to error")
    
    def reset_memory(self):
        """Reset all memory layers"""
        for layer in self.memory_layers:
            layer.reset_memory()
        logger.info("Memory state reset")
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None, 
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        """Forward pass with memory augmentation"""
        # Forward pass through base Mistral model
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,  # Always get hidden states
                return_dict=True,  # Always use return dict for consistency
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error during model forward pass: {e}")
            raise
        
        # Process hidden states through memory layers
        if self.hidden_states is not None:
            memory_enhanced = self.hidden_states
            orig_dtype = memory_enhanced.dtype
            logger.debug(f"Hidden states shape: {memory_enhanced.shape}, dtype: {memory_enhanced.dtype}")
            
            # Apply each memory layer in sequence
            for i, layer in enumerate(self.memory_layers):
                memory_enhanced = layer(memory_enhanced)
                logger.debug(f"After memory layer {i+1}: {memory_enhanced.shape}")
            
            # Find appropriate lm_head for output projection
            try:
                if hasattr(self.model, 'lm_head'):
                    lm_head = self.model.lm_head
                else:
                    # Search for lm_head if not found directly
                    lm_head = None
                    for name, module in self.model.named_modules():
                        if 'lm_head' in name or 'embed_out' in name:
                            lm_head = module
                            logger.info(f"Found lm_head: {name}")
                            break
                    
                    if lm_head is None:
                        logger.warning("Could not find lm_head, using original logits")
                        return outputs
                
                # Generate new logits from enhanced hidden states
                # Ensure types match before blending
                if memory_enhanced.dtype != outputs.logits.dtype:
                    memory_enhanced = memory_enhanced.to(dtype=outputs.logits.dtype)
                
                # Use memory enhanced hidden states to generate logits
                new_logits = lm_head(memory_enhanced)
                
                # Apply blend ratio for memory influence
                blended_logits = self.blend_ratio * new_logits + (1.0 - self.blend_ratio) * outputs.logits
                outputs.logits = blended_logits
                logger.debug("Successfully blended logits with memory enhancements")
            
            except Exception as e:
                logger.error(f"Error generating enhanced logits: {e}")
                logger.error(f"Error details: {str(e)}")
                # Return original outputs if there's an error
        
        return outputs


def load_tokenizer_with_auth(model_name, hf_token=None, fallback_model=None):
    """Load the tokenizer with authentication"""
    logger.info(f"Loading tokenizer for model: {model_name}")
    
    try:
        # Try with token if provided
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            logger.info("Tokenizer loaded successfully with auth token")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Tokenizer loaded successfully")
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
            
        return tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load tokenizer for primary model: {e}")
        
        # Try fallback model if provided
        if fallback_model:
            logger.info(f"Attempting to load tokenizer for fallback model: {fallback_model}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_model, token=hf_token if hf_token else None)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Fallback tokenizer loaded successfully")
                return tokenizer
            except Exception as fallback_err:
                logger.error(f"Failed to load fallback tokenizer: {fallback_err}")
        
        # Try a public alternative if all else fails
        logger.info("Attempting to load public tokenizer as last resort")
        try:
            public_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
            if public_tokenizer.pad_token is None:
                public_tokenizer.pad_token = public_tokenizer.eos_token
            logger.warning("Using public tokenizer as fallback")
            return public_tokenizer
        except:
            raise RuntimeError("Failed to load any tokenizer")


def create_memory_benchmark(
    tokenizer, 
    memory_key="IMPORTANT-MEMORY-CUE-alpha", 
    memory_value="42",
    context_length=200  # Same context length as in original
):
    """Create a memory benchmark test with identical params to the original"""
    # Create context with varying sentence structures - identical to original
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
    """Evaluate model's memory retrieval capability"""
    model.eval()
    
    # Move tokens to device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Forward pass
    with torch.no_grad():
        try:
            outputs = model(**tokens)
        except Exception as e:
            logger.error(f"Error during evaluation forward pass: {e}")
            raise
    
    # Get target token ID (with space prefix for proper tokenization)
    try:
        target_token_id = tokenizer.encode(" " + target_value)[0]
    except:
        # Fallback if space prefix doesn't work with this tokenizer
        try:
            target_token_id = tokenizer.encode(target_value)[0]
        except Exception as e:
            logger.error(f"Error encoding target value: {e}")
            raise
    
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
    
    # Improved detection of target value in response
    # Check if target value appears in generated text
    exact_value_present = target_value in generated_text.split()
    
    # Look for common patterns like "value is X" or "value was X"
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
    
    # Calculate output quality metrics
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


def demo_enhanced_memory_mistral(hf_token=None):
    """Demonstrate enhanced memory capabilities with Mistral (CPU-only)"""
    print("\n=== Enhanced Memory Mistral Demonstration (CPU-only) ===\n")
    
    # Force CPU only
    device = torch.device("cpu")
    print(f"Using device: {device} (forced)")
    
    try:
        # Initialize configuration with authentication
        config = EnhancedMemoryConfig(
            base_model_name="mistralai/Mistral-7B-v0.1",  # Primary model
            fallback_model="mistralai/Mistral-7B-Instruct-v0.2",  # Fallback model
            hf_token=hf_token,  # Pass authentication token
            memory_dim=256,  # Same as original
            num_memory_slots=64,  # Same as original
            memory_update_rate=0.5,  # Same as original
            num_memory_layers=2,  # Same as original
            attention_heads=4,  # Same as original
            dropout_rate=0.1,  # Same as original
            use_gru_controller=True,  # Same as original
            feedback_strength=0.7,  # Same as original
            blend_ratio=0.6,  # Same as original
            use_8bit_quantization=False,  # Disable 8-bit quantization for CPU
            force_cpu=True  # Force CPU execution
        )
        
        # Initialize tokenizer with authentication
        print("Initializing tokenizer...")
        tokenizer = load_tokenizer_with_auth(
            config.base_model_name,
            hf_token=config.hf_token,
            fallback_model=config.fallback_model
        )
        
        # Create enhanced memory model
        print(f"Creating enhanced memory model with {config.base_model_name}...")
        enhanced_model = MistralWithEnhancedMemory(config).to(device)
        
        # Create baseline model for comparison with same authentication
        print("Creating baseline model for comparison...")
        try:
            model_kwargs = {
                "device_map": "cpu",  # Force CPU
                "torch_dtype": torch.float32,  # Use float32 for CPU
                "low_cpu_mem_usage": True  # Optimize memory usage
            }
                
            if config.hf_token:
                model_kwargs["token"] = config.hf_token
                
            baseline_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                **model_kwargs
            )
            logger.info(f"Successfully loaded baseline model: {config.base_model_name}")
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            # Try fallback model
            if config.fallback_model:
                logger.info(f"Attempting to load fallback baseline model: {config.fallback_model}")
                baseline_model = AutoModelForCausalLM.from_pretrained(
                    config.fallback_model,
                    **model_kwargs
                )
                logger.info(f"Successfully loaded fallback baseline model")
        
        # Create memory benchmark
        print("\nCreating memory benchmark...")
        test_input, tokens = create_memory_benchmark(
            tokenizer,
            memory_key="IMPORTANT-MEMORY-CUE-alpha",
            memory_value="42",
            context_length=200  # Same as original
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
        
        # Improved output analysis
        print("\nDetailed Answer Analysis:")

        print("\nBaseline Model:")
        print(f"Actual answer segment: {baseline_results['actual_answer']}")
        print(f"Contains target value: {'Yes' if baseline_results['value_in_generation'] else 'No'}")
        
        print("\nEnhanced Memory Model:")
        print(f"Actual answer segment: {enhanced_results['actual_answer']}")
        print(f"Contains target value: {'Yes' if enhanced_results['value_in_generation'] else 'No'}")
        
        # Check if target value appears in generated text with improved detection
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
    # Get Hugging Face token if available in environment
    hf_token = os.environ.get("HF_TOKEN", None)
    
    # Run the demonstration with authentication
    demo_enhanced_memory_mistral(hf_token=hf_token)