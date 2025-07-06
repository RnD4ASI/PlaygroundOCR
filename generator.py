import os
# Set environment variables to suppress HuggingFace interactive prompts
# os.environ['HF_HUB_DISABLE_INTERACTIVE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import os
import io
import base64
import requests
from pdf2image import convert_from_path
from dotenv import load_dotenv
import json
import math
import numpy as np
import pandas as pd
import re
import shlex
import string
import time
from typing import List, Optional, Dict, Any, Union, Tuple
import uuid
from pathlib import Path
import torch
import psutil
import gc
import threading
import tempfile


# Monkey patch to automatically approve trust_remote_code prompts
# import builtins
# original_input = builtins.input
# def auto_approve_input(prompt=""):
#     if "trust_remote_code" in prompt.lower() or "remote code" in prompt.lower():
#         print(f"Auto-approving: {prompt} -> yes")
#         return "yes"
#     return original_input(prompt)
# builtins.input = auto_approve_input

# # Additional monkey patch for huggingface_hub if available
# try:
#     from huggingface_hub import hf_hub_download
#     from huggingface_hub.utils import HfHubHTTPError
#     # Patch the trust_remote_code confirmation function if it exists
#     import warnings
#     warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
# except ImportError:
#     pass
import torch.nn.functional as F
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationConfig
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder
from openai import AzureOpenAI
from azure.identity import ClientSecretCredential
import google.generativeai as genai
from google.generativeai import types
from anthropic import Anthropic
from mistralai import Mistral
# Optional MLX support for Apple-Silicon / CPU inference
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
except ImportError:
    mlx_load, mlx_generate = None, None
    logger.debug("mlx_lm not found, MLX support disabled.")

# Google Gemini imports (ensure they are here)
import google.generativeai as genai
from google.generativeai import types as genai_types

# Imports for specific OCR models like Nanonets
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.pipeline.shared.logging import get_logger
from src.pipeline.shared.utility import DataUtility, StatisticsUtility, AIUtility, MemoryUtility

# --- Type Definitions for Standardized Agentic Responses ---
from typing import TypedDict, List, Union, Optional, Any

# No longer using StandardizedToolCall directly in StandardizedMessage,
# tool call info is part of StandardizedMessageContentPart with type="tool_use"

class StandardizedMessageContentPart(TypedDict, total=False): # total=False allows for optional keys
    type: str # "text", "tool_use" (assistant requests tool), "tool_result" (user/system provides tool output)

    # Common fields
    text: Optional[str] # For type="text"

    # For type="tool_use" (assistant's request to call a tool)
    id: Optional[str] # Unique ID for this specific tool call suggestion
    name: Optional[str] # Name of the tool to be called
    input: Optional[Dict[str, Any]] # Parsed arguments for the tool

    # For type="tool_result" (response from a tool execution)
    tool_use_id: Optional[str] # ID of the tool_use request this result corresponds to
    # content for tool_result can be a string (e.g. simple output or error message) or structured JSON
    content: Optional[Union[str, Dict[str, Any], List[Any]]]
    is_error: Optional[bool] # True if the tool execution resulted in an error

class StandardizedMessage(TypedDict):
    role: str  # "user", "assistant", "tool"
    content: Union[str, List[StandardizedMessageContentPart]] # str for simple text messages, list of content parts for complex messages
    name: Optional[str] # Optional: For "tool" role, name of the tool. For "assistant", model's name if it has one.

class StandardizedAgenticResponse(TypedDict):
    prompt_id: Optional[int]
    prompt: str  # The initial user prompt text
    response: str  # The final textual response from the LLM
    full_conversation_history: List[StandardizedMessage]
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    model: str
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    tool_calls_made: int
# --- End Type Definitions ---


# --- MCP Client Imports ---
import asyncio
import nest_asyncio
from contextlib import AsyncExitStack
import json # Added json import
from typing import List, Optional, Dict, Any, Union, Tuple # Ensure these are available for MCPClientManager
from pathlib import Path # Ensure Path is available if server_config_path could be Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# --- End MCP Client Imports ---

logger = get_logger(__name__)

# Apply nest_asyncio to allow running asyncio code from synchronous generator methods
nest_asyncio.apply()


class MCPClientManager:
    """
    Manages connections to MCP servers and discovery of available tools.
    This is used by Generator when interacting with LLMs that support tool calling.
    """
    def __init__(self, server_config_path="server_config.json"):
        self.server_config_path = server_config_path
        self.exit_stack = None
        self.sessions: Dict[str, ClientSession] = {} # Maps server name to session
        self.tool_to_session_map: Dict[str, ClientSession] = {} # Maps tool name to its session
        self.available_tools: List[Dict[str, Any]] = [] # Tool schemas for LLM
        self._initialized = False
        self._initializing = False # Lock to prevent re-entrant initialization

    async def _initialize_stack(self):
        if self.exit_stack is None:
            self.exit_stack = AsyncExitStack()
            await self.exit_stack.__aenter__()

    async def connect_to_server(self, server_name: str, server_config: Dict[str, Any]):
        """Connects to a single MCP server and populates tools."""
        if server_name in self.sessions:
            logger.info(f"Already connected to MCP server: {server_name}")
            return

        try:
            logger.info(f"Connecting to MCP server: {server_name} with config {server_config}")
            server_params = StdioServerParameters(**server_config)

            await self._initialize_stack() # Ensure stack is ready

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions[server_name] = session
            logger.info(f"Successfully connected to MCP server: {server_name}")

            response = await session.list_tools()
            if response and response.tools:
                for tool in response.tools:
                    if tool.name in self.tool_to_session_map:
                        logger.warning(f"Tool '{tool.name}' from server '{server_name}' conflicts with an existing tool. Check MCP server configurations.")
                    self.tool_to_session_map[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                logger.info(f"Discovered {len(response.tools)} tools from {server_name}.")
            else:
                logger.info(f"No tools discovered from {server_name}.")


        except Exception as e:
            logger.error(f"Failed to connect or discover tools from MCP server {server_name}: {e}")
            # Optionally, remove server from sessions if connection failed mid-way
            if server_name in self.sessions:
                del self.sessions[server_name]


    async def connect_to_all_servers(self):
        if self._initialized or self._initializing:
            return

        self._initializing = True
        logger.info("MCPClientManager: Starting connection to all servers...")
        await self._initialize_stack()

        try:
            # Ensure current working directory is predictable or use absolute path for server_config.json
            config_file = Path(self.server_config_path)
            if not config_file.is_absolute():
                # Assuming generator.py is in the root of the project for this example
                config_file = Path(__file__).parent / self.server_config_path

            logger.info(f"Attempting to load MCP server configuration from: {config_file}")
            with open(config_file, "r") as f:
                config_data = json.load(f)

            mcp_servers_config = config_data.get("mcpServers", {})
            if not mcp_servers_config:
                logger.warning(f"No MCP servers found in {config_file}.")
                self._initialized = True
                self._initializing = False
                return

            # Filter out generator_config_server to prevent self-connection issues
            # and research_server if its tools are now in common_tools
            servers_to_connect = {
                name: cfg for name, cfg in mcp_servers_config.items()
                # if name not in ["generator_config", "research_server"] # Adjust if research_server is fully merged
            }
            if not servers_to_connect:
                logger.info("No applicable MCP servers to connect to after filtering.")


            for server_name, server_cfg in servers_to_connect.items():
                await self.connect_to_server(server_name, server_cfg)

            self._initialized = True
            logger.info(f"MCPClientManager initialized. Found {len(self.available_tools)} tools across configured servers.")

        except FileNotFoundError:
            logger.error(f"MCP server configuration file not found: {config_file}")
        except Exception as e:
            logger.error(f"Error initializing MCPClientManager during connect_to_all_servers: {e}", exc_info=True)
        finally:
            self._initializing = False


    async def get_tool_session(self, tool_name: str) -> Optional[ClientSession]:
        if not self._initialized:
            logger.warning("MCPClientManager not initialized when get_tool_session was called. Attempting lazy initialization.")
            await self.connect_to_all_servers() # Or self.ensure_initialized() if preferred sync wrapper

        session = self.tool_to_session_map.get(tool_name)
        if not session:
            logger.error(f"Tool '{tool_name}' not found in any connected MCP server session.")
        return session

    async def close_connections(self):
        if self.exit_stack:
            logger.info("MCPClientManager: Closing connections...")
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error during MCPClientManager exit_stack.aclose(): {e}")
            finally:
                self.exit_stack = None
                self.sessions.clear()
                self.tool_to_session_map.clear()
                self.available_tools.clear()
                self._initialized = False
                self._initializing = False
                logger.info("MCPClientManager connections closed and reset.")
        else:
            logger.info("MCPClientManager: No active connections to close.")


    def ensure_initialized(self):
        """Synchronous wrapper to ensure async initialization is run if needed."""
        if self._initialized or self._initializing:
            return

        logger.info("MCPClientManager not initialized. Initializing synchronously...")
        try:
            # Check if an event loop is already running.
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                logger.warning("Asyncio loop already running. Submitting connect_to_all_servers to the running loop.")
                # This is tricky. If called from a sync context with a running loop,
                # we can't just `asyncio.run()`. We might need a more complex setup
                # or rely on `nest_asyncio` heavily.
                # For now, let's assume nest_asyncio handles this if it's a nested call.
                # If this is the *outermost* sync call starting things, asyncio.run() is fine.
                asyncio.run(self.connect_to_all_servers()) # nest_asyncio should allow this
            else:
                asyncio.run(self.connect_to_all_servers())
        except RuntimeError as e:
            if "cannot be called when another asyncio event loop is running" in str(e) or \
               "Nesting asyncio.run() is not supported" in str(e):
                logger.warning(f"Asyncio context issue during ensure_initialized: {e}. nest_asyncio might be needed or initialization strategy revised.")
            else:
                logger.error(f"RuntimeError during MCPClientManager synchronous initialization: {e}", exc_info=True)
                # raise # Optionally re-raise
        except Exception as e:
            logger.error(f"Critical error during MCPClientManager synchronous initialization: {e}", exc_info=True)
            # raise # Optionally re-raise


class Generator:
    """Handles model interactions and text generation.
    Provides token management, Azure OpenAI API and HuggingFace model integrations.
    
    Singleton pattern ensures only one instance exists to prevent memory duplication.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not Generator._initialized:
            with Generator._lock:
                if not Generator._initialized:
                    self._real_init()
                    Generator._initialized = True
    
    def _real_init(self):
        """Actual initialization logic (called only once)."""
        logger.info("Initializing SINGLETON Generator instance")

        # Initialize MCP Client Manager
        # Using the existing mcp_tools.json configuration file
        self.mcp_client_manager = MCPClientManager(server_config_path="config/mcp_tools.json")
        self.mcp_client_manager.ensure_initialized()
        
        # Initialize utilities
        self.datautility = DataUtility()
        self.statsutility = StatisticsUtility()
        self.memoryutility = MemoryUtility()
        
        # Memory cleanup throttling
        self._last_cleanup_time = 0
        self._cleanup_throttle_seconds = 2  # Minimum 2 seconds between cleanups
        
        # Load configurations
        self.hf_model_dir = Path.cwd() / "model"
        self.config_dir = Path.cwd() / "config"
        
        # Default embedding dimensions for different model providers
        # These will be used as fallbacks if dimensions cannot be determined dynamically
        self.default_embedding_dimensions = {
            "azure_openai": 1536,
            "anthropic": 1024,
            "vertex": 768,
            "huggingface": 384
        }
        
        # Cache for model embedding dimensions
        self.embedding_dimensions_cache = {}
        
        # Load Main Configurations
        try:
            self.main_config_path  = self.config_dir / "config_main.json"
            self.main_config  = self.datautility.text_operation('load', self.main_config_path, file_type='json')
        except Exception as e:
            logger.error(f"Failed to load main configuration: {e}")
            raise

        # Load Model Configurations
        try:
            self.model_config_path = self.config_dir / "config_model.json"
            self.model_config = self.datautility.text_operation('load', self.model_config_path, file_type='json')
            
            # Load embedding dimensions from config if available
            # if "embedding_dimensions" in self.model_config:
            #     self.default_embedding_dimensions.update(self.model_config["embedding_dimensions"])
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            raise
        
        # Default Model Settings
        self.default_embedding_model = self.model_config["validation_rules"]["models"]["embedding"]["azure_openai"][0]
        self.default_completion_model = self.model_config["validation_rules"]["models"]["completion"]["azure_openai"][0]
        self.default_reasoning_model = self.model_config["validation_rules"]["models"]["reasoning"]["azure_openai"][0]
        
        self.default_hf_embedding_model = self.model_config["validation_rules"]["models"]["embedding"]["huggingface"][0]
        self.default_hf_completion_model = self.model_config["validation_rules"]["models"]["completion"]["huggingface"][0]
        self.default_hf_reasoning_model = self.model_config["validation_rules"]["models"]["reasoning"]["huggingface"][0]
        self.default_hf_reranker_model = self.model_config["validation_rules"]["models"]["reranker"]["huggingface"][0]
        self.default_hf_ocr_model = self.model_config["validation_rules"]["models"]["ocr"]["huggingface"][0]
        self.default_hf_tokeniser_model = self.model_config["validation_rules"]["models"]["tokeniser"]["huggingface"][0]
        self.default_max_attempts = self.main_config["generator"]["api_configuration"]["max_retry_attempts"]
        self.default_wait_time = self.main_config["generator"]["api_configuration"]["retry_wait_time_seconds"]
        
        # Azure OpenAI settings
        load_dotenv()
        self.scope = os.getenv("SCOPE")
        self.tenant_id = os.getenv("TENANT_ID")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.subscription_key = os.getenv("SUBSCRIPTION_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Google Gemini settings
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            
        # Anthropic Claude settings
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        
        # Mistral settings
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")

        # Model caches for reuse
        self.embedding_model_cache = {}
        self.completion_model_cache = {}
        
        # HuggingFace models - loaded once at initialization
        self.hf_embedding_model = None
        self.hf_completion_model = None
        self.hf_completion_tokenizer = None
        
        # Memory management settings
        self.max_cache_size = 2  # Maximum number of models to keep in memory
        self.memory_cleanup_interval = 4  # Clear memory every N batches
        self.current_model = None  # Track currently loaded model for cleanup
        
        # Configure environment to prevent virtual memory explosion
        self._configure_memory_environment()
        
        logger.debug(f"Generator initialized with default models:\n- Embedding Model: {self.default_embedding_model}\n- Completion Model: {self.default_completion_model}\n- Reasoning Model: {self.default_reasoning_model}\n- HuggingFace Embedding Model: {self.default_hf_embedding_model}\n- HuggingFace Completion Model: {self.default_hf_completion_model}\n- HuggingFace Reasoning Model: {self.default_hf_reasoning_model}\n- HuggingFace Reranker Model: {self.default_hf_reranker_model}\n- HuggingFace OCR Model: {self.default_hf_ocr_model}")

    def _configure_memory_environment(self):
        """Configure environment variables to prevent virtual memory explosion."""
        try:
            import os
            # Set environment variables to prevent MPS backend VMS explosion
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable high watermark
            os.environ['MPS_DISABLE_LARGE_MEMORY_ALLOCATION'] = '1'  # Disable large allocations
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism to reduce memory
            os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
            os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads
            os.environ['MKL_NUM_THREADS'] = '1'  # Limit Intel MKL threads
            # Set PyTorch specific settings
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Allow fallback to CPU
                        
        except Exception as e:
            logger.warning(f"Failed to configure memory environment: {e}")

    def load_hf_models(self, embedding_model: Optional[str] = None, completion_model: Optional[str] = None):
        """Load HuggingFace models once at initialization to prevent VMS explosion.
        
        Args:
            embedding_model: Optional embedding model name, defaults to configured model
            completion_model: Optional completion model name, defaults to configured model
        """
        logger.info("Loading HuggingFace models once at initialization")
        
        # Log baseline memory before any model operations
        self._log_memory_baseline("BEFORE_MODEL_LOADING")
        
        # Load embedding model
        embedding_model_name = embedding_model or self.default_hf_embedding_model
        try:
            local_model_path = self.hf_model_dir / embedding_model_name
            if local_model_path.exists():
                logger.info(f"Loading HF embedding model from: {local_model_path}")
                
                # Log memory before cleanup
                self._log_memory_baseline("BEFORE_EMBEDDING_CLEANUP")
                
                self._cleanup_memory()
                
                # Log memory after cleanup, before SentenceTransformer creation
                self._log_memory_baseline("AFTER_CLEANUP_BEFORE_SENTENCE_TRANSFORMER")
                
                # Set extremely restrictive environment before SentenceTransformer creation
                old_env = {}
                try:
                    # Save current environment
                    env_vars_to_control = [
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO',
                        'MPS_DISABLE_LARGE_MEMORY_ALLOCATION', 
                        'TOKENIZERS_PARALLELISM',
                        'TRANSFORMERS_CACHE',
                        'HF_HOME',
                        'PYTORCH_ENABLE_MPS_FALLBACK',
                        'TRANSFORMERS_OFFLINE',
                        'HF_DATASETS_OFFLINE',
                        'CUDA_VISIBLE_DEVICES',
                        'PYTORCH_CUDA_ALLOC_CONF',
                        'OMP_NUM_THREADS',
                        'MKL_NUM_THREADS'
                    ]
                    
                    for var in env_vars_to_control:
                        if var in os.environ:
                            old_env[var] = os.environ[var]
                    
                    # Set ultra-aggressive memory controls
                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS memory pooling completely
                    os.environ['MPS_DISABLE_LARGE_MEMORY_ALLOCATION'] = '1'  # Disable large allocations
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable fallback to CPU
                    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'  # Use temp cache
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
                    os.environ['HF_DATASETS_OFFLINE'] = '1'  # Force datasets offline
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit CUDA allocation
                    os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
                    os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
                    
                    logger.info("Applied ULTRA-AGGRESSIVE memory environment controls")
                    
                    # Pre-emptive VMS check before model creation
                    try:
                        if psutil:
                            process = psutil.Process(os.getpid())
                            pre_vms_gb = process.memory_info().vms / (1024**3)
                            logger.info(f"🔍 Pre-SentenceTransformer VMS: {pre_vms_gb:.1f}GB")
                            
                            if pre_vms_gb > 200:
                                logger.warning(f"⚠️  Pre-SentenceTransformer VMS already high: {pre_vms_gb:.1f}GB")
                    except Exception as e:
                        logger.debug(f"Could not check pre-VMS: {e}")
                    
                    # Force garbage collection before model creation
                    import gc
                    gc.collect()
                    
                    # Load with ultra-conservative settings and VMS monitoring
                    logger.info("🧠 Creating SentenceTransformer with ultra-conservative settings...")
                    
                    # Try multiple strategies if VMS gets too high
                    self.hf_embedding_model = self._create_sentence_transformer_with_vms_monitoring(
                        str(local_model_path)
                    )
                    
                    # Immediate post-creation VMS check
                    try:
                        if psutil:
                            process = psutil.Process(os.getpid())
                            post_vms_gb = process.memory_info().vms / (1024**3)
                            logger.info(f"📊 Post-SentenceTransformer VMS: {post_vms_gb:.1f}GB")
                            
                            logger.info(f"📊 Final VMS after SentenceTransformer creation: {post_vms_gb:.1f}GB")
                    except Exception as e:
                        logger.debug(f"Could not check post-VMS: {e}")
                    
                finally:
                    # Restore original environment
                    for var in env_vars_to_control:
                        if var in old_env:
                            os.environ[var] = old_env[var]
                        elif var in os.environ:
                            del os.environ[var]
                    logger.info("Restored original environment variables")
                
                # Log memory immediately after SentenceTransformer creation
                self._log_memory_baseline("AFTER_SENTENCE_TRANSFORMER_CREATION")
                
                logger.info(f"Successfully loaded HF embedding model: {embedding_model_name}")
            else:
                logger.warning(f"HF embedding model not found at {local_model_path}")
        except Exception as e:
            logger.error(f"Failed to load HF embedding model: {e}")
            self.hf_embedding_model = None
        
        # Load completion model
        completion_model_name = completion_model or self.default_hf_completion_model
        try:
            local_model_path = self.hf_model_dir / completion_model_name
            if local_model_path.exists():
                logger.info(f"Loading HF completion model from: {local_model_path}")
                
                # Log memory before completion model cleanup
                self._log_memory_baseline("BEFORE_COMPLETION_CLEANUP")
                
                self._cleanup_memory()
                
                # Log memory after cleanup, before tokenizer creation
                self._log_memory_baseline("AFTER_CLEANUP_BEFORE_TOKENIZER")
                
                # Load tokenizer and model with trust_remote_code and half precision
                logger.info("🔄 CREATING AutoTokenizer - monitoring VMS...")
                self.hf_completion_tokenizer = AutoTokenizer.from_pretrained(
                    str(local_model_path), 
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Log memory after tokenizer creation
                self._log_memory_baseline("AFTER_TOKENIZER_CREATION")
                
                logger.info("🔄 CREATING AutoModelForCausalLM - monitoring VMS...")
                self.hf_completion_model = AutoModelForCausalLM.from_pretrained(
                    str(local_model_path), 
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # Use half precision to reduce memory
                    device_map="cpu"  # Keep on CPU initially
                )
                
                # Log memory after completion model creation
                self._log_memory_baseline("AFTER_COMPLETION_MODEL_CREATION")
                
                # Set pad token if needed
                if self.hf_completion_tokenizer.pad_token is None:
                    self.hf_completion_tokenizer.pad_token = self.hf_completion_tokenizer.eos_token
                
                logger.info(f"Successfully loaded HF completion model: {completion_model_name}")
            else:
                logger.warning(f"HF completion model not found at {local_model_path}")
        except Exception as e:
            logger.error(f"Failed to load HF completion model: {e}")
            self.hf_completion_model = None
            self.hf_completion_tokenizer = None
        
        # Final memory check
        self._log_memory_baseline("BEFORE_FINAL_CLEANUP")
        self._cleanup_memory()
        self._log_memory_baseline("AFTER_FINAL_CLEANUP")

    def _cleanup_memory(self, force: bool = False):
        """Clean up GPU and system memory and monitor virtual memory."""
        try:
            import time
            
            # Throttle cleanup calls to prevent excessive overhead (unless forced)
            current_time = time.time()
            if not force and (current_time - self._last_cleanup_time) < self._cleanup_throttle_seconds:
                logger.debug(f"Skipping cleanup - throttled (last cleanup {current_time - self._last_cleanup_time:.1f}s ago)")
                return
            
            self._last_cleanup_time = current_time
            # Check virtual memory before cleanup
            try:
                import platform
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                vms_gb = memory_info.vms / (1024**3)  # Convert to GB
                
                # Check both VMS and RSS (actual RAM usage)
                rss_gb = memory_info.rss / (1024**3)  # Convert to GB
                
                # Check RAM usage - this is the real issue
                if rss_gb > 50:  # Critical RAM limit - terminate to prevent system kill
                    logger.error(f"RAM usage {rss_gb:.1f}GB exceeds critical limit (50GB), terminating to prevent system kill")
                    raise RuntimeError(f"RAM memory exhaustion: {rss_gb:.1f}GB allocated")
                elif rss_gb > 25:  # Warning RAM limit - force aggressive cleanup
                    logger.warning(f"RAM usage {rss_gb:.1f}GB exceeds safe limit, forcing aggressive cleanup")
                    force = True
                elif rss_gb > 10:  # Moderate RAM usage - log warning
                    logger.info(f"RAM usage {rss_gb:.1f}GB is getting high")
                
                # Platform-aware VMS limits
                is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
                
                if is_apple_silicon:
                    # Apple Silicon has much higher baseline VMS (~390GB), so use higher limits
                    vms_critical_limit = 500  # 500GB critical limit for Apple Silicon
                    vms_warning_limit = 450   # 450GB warning limit for Apple Silicon
                    # logger.debug(f"Using Apple Silicon VMS limits: critical={vms_critical_limit}GB, warning={vms_warning_limit}GB")
                else:
                    # Traditional x86/Intel limits
                    vms_critical_limit = 250  # 250GB critical limit for Intel/x86
                    vms_warning_limit = 150   # 150GB warning limit for Intel/x86
                    # logger.debug(f"Using Intel/x86 VMS limits: critical={vms_critical_limit}GB, warning={vms_warning_limit}GB")
                
                # Check VMS against platform-appropriate limits
                if vms_gb > vms_critical_limit:  # Critical VMS limit - terminate to prevent system kill
                    logger.error(f"Virtual memory size {vms_gb:.1f}GB exceeds critical limit ({vms_critical_limit}GB), terminating to prevent system kill")
                    raise RuntimeError(f"Virtual memory exhaustion: {vms_gb:.1f}GB allocated")
                elif vms_gb > vms_warning_limit:  # Warning VMS limit - force aggressive cleanup
                    logger.info(f"Virtual memory size {vms_gb:.1f}GB exceeds safe limit ({vms_warning_limit}GB), forcing aggressive cleanup")
                    force = True
                    try:
                        # Clear environment variables that might be causing VMS bloat
                        if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
                            del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']
                        if 'MPS_DISABLE_LARGE_MEMORY_ALLOCATION' in os.environ:
                            del os.environ['MPS_DISABLE_LARGE_MEMORY_ALLOCATION']
                        # Set more restrictive environment
                        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                        os.environ['MPS_DISABLE_LARGE_MEMORY_ALLOCATION'] = '1'
                        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism
                        logger.debug("Cleared environment variables")
                    except Exception as e:
                        logger.warning(f"Failed to clear environment variables: {e}")
                else:
                    logger.debug(f"Memory usage: RSS {rss_gb:.1f}GB, VMS {vms_gb:.1f}GB (within safe limits)")
                    
            except RuntimeError:
                # Re-raise RuntimeError for emergency memory termination
                raise
            except Exception as mem_check_error:
                logger.debug(f"Could not check virtual memory: {mem_check_error}")
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            
            # If force cleanup, clear model caches and unload main models
            if force:
                self.embedding_model_cache.clear()
                self.completion_model_cache.clear()
                self.current_model = None
                
                # Aggressive model cleanup - unload main models if memory pressure is critical
                self._unload_models_if_critical()
                logger.info("Cleared all model caches and unloaded models if critical")
                
        except RuntimeError:
            # Re-raise RuntimeError for emergency memory termination
            raise  
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def _log_memory_baseline(self, stage: str) -> None:
        """Log detailed memory usage at a specific stage."""
        try:
            import gc
            if psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                vms_gb = memory_info.vms / (1024**3)
                rss_gb = memory_info.rss / (1024**3)
                percent = process.memory_percent()
                
                # Get system memory info
                system_memory = psutil.virtual_memory()
                available_gb = system_memory.available / (1024**3)
                
                # Get garbage collection stats
                gc_stats = gc.get_stats() if hasattr(gc, 'get_stats') else []
                gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)
                
                logger.debug(
                    f"📊 MEMORY_DIAGNOSTIC | {stage} | "
                    f"VMS: {vms_gb:.1f}GB | RSS: {rss_gb:.1f}GB | "
                    f"Percent: {percent:.1f}% | Available: {available_gb:.1f}GB | "
                    f"GC_Collections: {gc_collections}"
                )
                
                # Log warning if VMS is getting high (platform-aware)
                import platform
                is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
                vms_warning_threshold = 450 if is_apple_silicon else 350  # Higher threshold for Apple Silicon
                
                if vms_gb > vms_warning_threshold:
                    platform_info = "Apple Silicon" if is_apple_silicon else "Intel/x86"
                    logger.warning(f"⚠️  HIGH VMS DETECTED at {stage}: {vms_gb:.1f}GB (platform: {platform_info})")
                    
            else:
                logger.warning(f"MEMORY_DIAGNOSTIC | {stage} | psutil not available")
                
        except Exception as e:
            logger.warning(f"Memory baseline logging failed at {stage}: {e}")
    
    def _create_sentence_transformer_with_vms_monitoring(self, model_path: str):
        """Create SentenceTransformer with aggressive VMS monitoring and fallback strategies."""
        import gc
        from sentence_transformers import SentenceTransformer
        
        strategies = [
            {
                "name": "Strategy 1: Standard CPU-only",
                "kwargs": {
                    "local_files_only": True,
                    "trust_remote_code": True,
                    "device": 'cpu',
                    "cache_folder": '/tmp/sentence_transformers_cache'
                }
            },
            {
                "name": "Strategy 2: Minimal cache with thread limiting",
                "kwargs": {
                    "local_files_only": True,
                    "trust_remote_code": True,
                    "device": 'cpu',
                    "cache_folder": None  # Disable caching
                }
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                logger.debug(f"🔬 Attempting {strategy['name']}...")
                
                # Pre-strategy VMS check
                pre_vms_gb = None
                if psutil:
                    try:
                        process = psutil.Process(os.getpid())
                        pre_vms_gb = process.memory_info().vms / (1024**3)
                        logger.debug(f"📊 Pre-strategy VMS: {pre_vms_gb:.1f}GB")
                        
                        # If VMS is already critically high, try emergency cleanup (platform-aware)
                        import platform
                        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
                        vms_cleanup_threshold = 450 if is_apple_silicon else 200  # 450GB for Apple Silicon, 200GB for Intel/x86
                        
                        if pre_vms_gb > vms_cleanup_threshold:
                            logger.warning(f"⚠️  VMS critically high before model creation, attempting emergency cleanup...")
                            gc.collect()
                            # Force unload any existing models
                            if hasattr(self, 'hf_embedding_model') and self.hf_embedding_model is not None:
                                del self.hf_embedding_model
                                self.hf_embedding_model = None
                            gc.collect()
                            
                            # Re-check VMS after cleanup
                            new_vms_gb = process.memory_info().vms / (1024**3)
                            logger.debug(f"📉 VMS after emergency cleanup: {new_vms_gb:.1f}GB (reduced by {pre_vms_gb - new_vms_gb:.1f}GB)")
                            pre_vms_gb = new_vms_gb
                    except Exception as e:
                        logger.debug(f"Could not check pre-strategy VMS: {e}")
                
                # Create model with current strategy
                model = SentenceTransformer(model_path, **strategy['kwargs'])
                
                # Post-strategy VMS check
                if psutil and pre_vms_gb is not None:
                    try:
                        process = psutil.Process(os.getpid())
                        post_vms_gb = process.memory_info().vms / (1024**3)
                        vms_increase = post_vms_gb - pre_vms_gb
                        
                        logger.debug(f"✅ {strategy['name']} succeeded!")
                        logger.debug(f"📈 VMS increase: {vms_increase:.1f}GB (from {pre_vms_gb:.1f}GB to {post_vms_gb:.1f}GB)")
                        
                        # If VMS is reasonable for platform, accept this strategy
                        import platform
                        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
                        vms_acceptable_limit = 450 if is_apple_silicon else 300  # 450GB for Apple Silicon, 300GB for Intel/x86
                        
                        if post_vms_gb < vms_acceptable_limit:  # Total VMS under platform limit
                            logger.debug(f"✨ Strategy successful with acceptable VMS: {post_vms_gb:.1f}GB")
                            return model
                        else:
                            logger.warning(f"⚠️  Strategy created model but VMS too high: {post_vms_gb:.1f}GB")
                            if i < len(strategies):
                                logger.debug(f"🔄 Trying next strategy...")
                                del model
                                gc.collect()
                                continue
                            else:
                                logger.warning(f"⚠️  All strategies exhausted, using last model despite high VMS")
                                return model
                                
                    except Exception as e:
                        logger.debug(f"Could not check post-strategy VMS: {e}")
                        # If we can't check VMS, assume success
                        logger.debug(f"✅ {strategy['name']} completed (VMS check failed)")
                        return model
                else:
                    # If no VMS monitoring available, return first successful model
                    logger.debug(f"✅ {strategy['name']} completed (no VMS monitoring)")
                    return model
                    
            except Exception as e:
                logger.error(f"❌ {strategy['name']} failed: {e}")
                if i < len(strategies):
                    logger.info(f"🔄 Trying next strategy...")
                    gc.collect()
                    continue
                else:
                    logger.error(f"💥 All SentenceTransformer creation strategies failed!")
                    raise RuntimeError(f"Failed to create SentenceTransformer with all strategies. Last error: {e}")
        
        # Should never reach here, but just in case
        raise RuntimeError("Unexpected end of SentenceTransformer creation strategies")
    
    def _unload_models_if_critical(self):
        """Progressive model unloading strategy based on memory pressure levels."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            rss_gb = memory_info.rss / (1024**3)
            vms_gb = memory_info.vms / (1024**3)
            
            # Progressive unloading strategy based on memory thresholds
            unloaded_something = False
            
            # Platform-aware VMS thresholds
            import platform
            is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
            if is_apple_silicon:
                vms_stage1 = 420  # 420GB for Apple Silicon Stage 1
                vms_stage2 = 440  # 440GB for Apple Silicon Stage 2  
                vms_stage3 = 460  # 460GB for Apple Silicon Stage 3
            else:
                vms_stage1 = 100  # 100GB for Intel/x86 Stage 1
                vms_stage2 = 150  # 150GB for Intel/x86 Stage 2
                vms_stage3 = 200  # 200GB for Intel/x86 Stage 3
            
            # Stage 1: High memory pressure (>8GB RAM or platform VMS threshold) - unload completion model first
            if (rss_gb > 8 or vms_gb > vms_stage1) and self.hf_completion_model is not None:
                logger.warning(f"Stage 1: High memory pressure (RSS: {rss_gb:.1f}GB, VMS: {vms_gb:.1f}GB) - unloading completion model")
                try:
                    if hasattr(self.hf_completion_model, 'to'):
                        self.hf_completion_model.to('cpu')
                    del self.hf_completion_model
                    self.hf_completion_model = None
                    logger.info("✅ Unloaded HF completion model (Stage 1)")
                    unloaded_something = True
                except Exception as e:
                    logger.warning(f"Failed to unload completion model: {e}")
            
            # Stage 2: Critical memory pressure (>12GB RAM or platform VMS threshold) - unload tokenizer
            if (rss_gb > 12 or vms_gb > vms_stage2) and self.hf_completion_tokenizer is not None:
                logger.warning(f"Stage 2: Critical memory pressure (RSS: {rss_gb:.1f}GB, VMS: {vms_gb:.1f}GB) - unloading tokenizer")
                try:
                    del self.hf_completion_tokenizer
                    self.hf_completion_tokenizer = None
                    logger.info("✅ Unloaded HF completion tokenizer (Stage 2)")
                    unloaded_something = True
                except Exception as e:
                    logger.warning(f"Failed to unload tokenizer: {e}")
            
            # Stage 3: Emergency memory pressure (>15GB RAM or platform VMS threshold) - unload embedding model (last resort)
            if (rss_gb > 15 or vms_gb > vms_stage3) and self.hf_embedding_model is not None:
                logger.error(f"Stage 3: EMERGENCY memory pressure (RSS: {rss_gb:.1f}GB, VMS: {vms_gb:.1f}GB) - unloading embedding model")
                try:
                    if hasattr(self.hf_embedding_model, 'to'):
                        self.hf_embedding_model.to('cpu')
                    del self.hf_embedding_model
                    self.hf_embedding_model = None
                    logger.info("✅ Unloaded HF embedding model (Stage 3 - EMERGENCY)")
                    unloaded_something = True
                except Exception as e:
                    logger.warning(f"Failed to unload embedding model: {e}")
            
            # Force cleanup after any unloading
            if unloaded_something:
                gc.collect()
                logger.info("🧹 Performed garbage collection after progressive model unloading")
                
                # Check memory reduction after cleanup
                new_memory_info = process.memory_info()
                new_rss_gb = new_memory_info.rss / (1024**3)
                new_vms_gb = new_memory_info.vms / (1024**3)
                logger.debug(f"📊 Memory after progressive unloading: RSS {new_rss_gb:.1f}GB (was {rss_gb:.1f}GB), VMS {new_vms_gb:.1f}GB (was {vms_gb:.1f}GB)")
            else:
                logger.debug(f"No model unloading needed - memory within acceptable limits: RSS {rss_gb:.1f}GB, VMS {vms_gb:.1f}GB")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.warning(f"Failed to check memory for model unloading: {e}")
    
    def reload_models_if_needed(self):
        """Reload models if they were unloaded due to memory pressure."""
        if self.hf_embedding_model is None or self.hf_completion_model is None:
            logger.info("Reloading models that were unloaded due to memory pressure")
            self.load_hf_models()

    def _manage_model_cache(self, model_key: str, model_obj, cache_dict: dict):
        """Manage model cache size and memory."""
        # Remove oldest models if cache is full
        if len(cache_dict) >= self.max_cache_size:
            oldest_key = next(iter(cache_dict))
            removed_model = cache_dict.pop(oldest_key)
            logger.debug(f"Removed model {oldest_key} from cache")
            
            # Explicit cleanup of removed model
            try:
                if hasattr(removed_model, 'to'):
                    removed_model.to('cpu')
                del removed_model
                self._cleanup_memory()
            except:
                pass
        
        # Add new model to cache
        cache_dict[model_key] = model_obj
        logger.debug(f"Added model {model_key} to cache")

    def _validate_model(self, model_name: str, model_type: str, provider: str = None) -> bool:
        """Validate model name against configuration.

        Parameters:
            model_name (str): Name of the model to validate
            model_type (str): Type of model (completion, embedding, ocr)
            provider (str): Optional provider specification (azure_openai, huggingface)

        Returns:
            Union[bool, Tuple[bool, str]]: 
                - If provider is specified: returns bool indicating if model is valid
                - If provider is not specified: returns tuple (is_valid, detected_provider)
        """
        try:
            # Check if model type exists
            if model_type not in self.model_config["validation_rules"]["models"]:
                logger.error(f"Invalid model type: {model_type}")
                return False
            
            model_rules = self.model_config["validation_rules"]["models"][model_type]            
            # If provider is specified, check only that provider
            if provider:
                if provider not in model_rules:
                    logger.error(f"Invalid provider: {provider}")
                    return False
                else:
                    return model_name in model_rules[provider]
            else:
                logger.error(f"Model provider not given")
                return False
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False

    def _validate_parameters(self, params: Dict[str, Any], model_type: str) -> bool:
        """Validate model parameters against configuration.

        Parameters:
            params (Dict[str, Any]): Parameters to validate
            model_type (str): Type of model (completion, embedding)

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        try:
            param_rules = self.model_config["validation_rules"]["model_parameters"][model_type]
            
            for param_name, param_value in params.items():
                if param_name in param_rules:
                    rules = param_rules[param_name]
                    if not (rules["min"] <= param_value <= rules["max"]):
                        logger.error(f"Parameter {param_name} value {param_value} outside valid range [{rules['min']}, {rules['max']}]")
                        return False
            return True
        except KeyError:
            logger.error(f"Invalid model type or missing parameter rules: {model_type}")
            return False

    def refresh_token(self) -> str:
        """Refreshes the Azure API token.

        Returns:
            str: The refreshed Azure token
        """
        try:
            # Get token with Azure credentials
            client_credentials = ClientSecretCredential(
                self.tenant_id, 
                self.client_id, 
                self.client_secret
            )
            access_token = client_credentials.get_token(self.scope).token
            logger.info("Successfully refreshed Azure token")
            return access_token

        except Exception as e:
            logger.error("Failed to refresh token: %s", e)
            raise

    def get_completion(self, 
                      prompt: str, 
                      prompt_id: Optional[int] = None,
                      system_prompt: Optional[str] = None,
                      model: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      top_p: Optional[float] = None,
                      top_k: Optional[float] = None,
                      frequency_penalty: Optional[float] = None,
                      presence_penalty: Optional[float] = None,
                      stop: Optional[List[str]] = None,
                      seed: Optional[int] = None,
                      logprobs: Optional[bool] = None,
                      json_schema: Optional[Dict[str, Any]] = None,
                      return_full_response: Optional[bool] = False,
                      num_beam: Optional[int] = None,
                      store_result: Optional[bool] = False,
                      stored_df: Optional[pd.DataFrame] = None) -> Union[Dict[str, Any], str]:
        """Get text completion using specified model or fall back to HuggingFace model.
        
        Parameters:
            prompt (str): Input prompt or chat messages
            prompt_id (int): Identifier for the prompt
            model (str): Specific model to use (e.g., "gpt-4", "Mistral-7B-v0.2", "gemini-1.5-pro", "claude-3-opus")
                                 If None or invalid, falls back to initialized HuggingFace model
            temperature (float): Sampling temperature (0-1)
            max_tokens (int): Maximum tokens in response
            top_p (float): Nucleus sampling parameter
            top_k (float):
            frequency_penalty (float): Frequency penalty parameter
            presence_penalty (float): Presence penalty parameter
            seed (Optional[int]): Random seed for reproducibility, only available when using azure openai
            logprobs (bool): Whether to return log probabilities
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            store_result (bool): Whether to store results in stored_df DataFrame
            stored_df (Optional[pd.DataFrame]): DataFrame to store results in
        
        Returns:
            Dict[str, Any]: Completion response containing:
                - prompt_id: Identifier for the prompt
                - prompt: Original prompt
                - response: Generated response
                - perplexity: Perplexity score
                - tokens_in: Number of input tokens
                - tokens_out: Number of output tokens
                - model: Model used
                - seed: Random seed used
                - description: Response description
                - top_p: Top-p value used
                - temperature: Temperature used
        """
        # Initialize a dataframe to store the results if store_result is enabled
        if store_result == True and stored_df is None:
            stored_df = pd.DataFrame()

        if model is None:
            logger.warning("Model not specified, falling back to the default HuggingFace model")
            model = self.default_hf_completion_model
        
        try:
            if self._validate_model(model, "completion", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI model '{model}' for completion")
                    df_result, final_response = self._get_azure_completion(
                        prompt=prompt,
                        prompt_id=prompt_id,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        logprobs=logprobs,
                        json_schema=json_schema
                    )
                except Exception as e:
                    logger.warning(f"Azure OpenAI completion failed: {e}")
            
            # Try Google Gemini if it's a Gemini model
            elif self._validate_model(model, "completion", "vertex"):
                try:
                    logger.info(f"Using Google Gemini model '{model}' for completion")
                    df_result, final_response = self._get_gemini_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": json_schema,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Google Gemini completion failed: {e}")
            
            # Try Anthropic Claude if it's a Claude model
            elif self._validate_model(model, "completion", "anthropic"):
                try:
                    logger.info(f"Using Anthropic Claude model '{model}' for completion")
                    df_result, final_response = self._get_claude_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                        # no option to restrict json schema output, however, could be prompted. #
                    )
                except Exception as e:
                    logger.warning(f"Anthropic Claude completion failed: {e}")
            
            # Try HuggingFace if it's a HF model
            elif self._validate_model(model, "completion", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace model '{model}' for completion")
                    df_result, final_response = self._get_hf_completion(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        num_beam=num_beam,
                        # presence_penalty=presence_penalty,
                        # stop=stop,
                        # seed=seed,
                        logprobs=logprobs
                    )
                    logger.info(f"HuggingFace completion successful with df_result: {df_result}")
                    logger.info(f"HuggingFace completion successful with final_response: {final_response}")
                except Exception as e:
                    logger.warning(f"HuggingFace completion failed: {e}")
            
            else:
                logger.info(f"Using default HuggingFace model '{model}' for completion")
                df_result, final_response = self._get_hf_completion(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    model=self.default_hf_completion_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    num_beam=num_beam, 
                    # presence_penalty=presence_penalty,
                    # stop=stop,
                    # seed=seed,
                    logprobs=logprobs
                )
            
            # Store results if enabled
            if store_result == True:
                logger.info(f"Storing results in DataFrame")
                logger.info("Number of rows in stored_df before: ", len(stored_df))
                try:
                    # Convert df_result to DataFrame
                    df_formatted = self.datautility.format_conversion(df_result, "dataframe")
                    if isinstance(df_formatted, pd.DataFrame):
                        # Use df_formatted.T (transposed) as the new row
                        result_to_add = df_formatted.T
                        logger.info("Number of rows in result_to_add: ", len(result_to_add))
                    else:
                        # Create a DataFrame from the result if format_conversion doesn't return one
                        result_to_add = pd.DataFrame([df_result])
                        logger.info("Number of rows in result_to_add: ", len(result_to_add))
                    
                    # If stored_df is empty, copy structure from result_to_add
                    if stored_df.empty and not result_to_add.empty:
                        for col in result_to_add.columns:
                            stored_df[col] = pd.Series(dtype=result_to_add[col].dtype)
                    
                    # Append the new row(s) to stored_df
                    for idx, row in result_to_add.iterrows():
                        stored_df.loc[len(stored_df)] = row
                    logger.info("Number of rows in stored_df after: ", len(stored_df))    

                except Exception as e:
                    logger.warning(f"Failed to store result: {e}")
            else:
                logger.info("store_result is False, not storing result")
            if return_full_response: 
                return df_result
            else: 
                return final_response

        except Exception as e:
            logger.error(f"Completion failed: {str(e)}")
            raise
            
    def _get_azure_completion(self,
                            prompt_id: int,
                            prompt: str, # Union[str, List[Dict[str, str]]],
                            system_prompt: Optional[str] = None,
                            model: Optional[str] = "gpt-4o",  
                            temperature: Optional[float] = 1,
                            max_tokens: Optional[int] = 3000,
                            top_p: Optional[float] = 1,
                            top_k: Optional[float] = 10,
                            frequency_penalty: Optional[float] = 1.1,
                            presence_penalty: Optional[float] = 1,
                            seed: Optional[int] = 100,
                            logprobs: Optional[bool] = False,
                            json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Internal method for Azure OpenAI API completion."""        
        # Process prompt into messages
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Fill in the random seed
        seed = self.statsutility.set_random_seed(min_value = 0, max_value = 100) if seed is None else seed
        
        # Determine response format
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        elif re.search(r'\bJSON\b', prompt, re.IGNORECASE):
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        # Allow a number of attempts when calling API    
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token
                access_token = self.refresh_token()
                client = AzureOpenAI(
                    api_version=self.api_version, 
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token = access_token
                )

                # Make API call
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                    seed=seed,
                    logprobs=logprobs,
                    extra_headers = {
                        'x-correlation-id': str(uuid.uuid4()),
                        'x-subscription-key': self.subscription_key
                    }
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Process JSON response if needed
                if response_format["type"] == "json_object":
                    try:
                        response_text = json.loads(response_text.strip('```json').strip('```'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise ValueError("Invalid JSON response from model")
                
                # Calculate perplexity
                log_probs = response.choices[0].logprobs if logprobs else None
                perplexity = self._calculate_perplexity(log_probs) if log_probs else None
                
                # Prepare result dictionary
                results = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response_text,
                    "perplexity": perplexity if perplexity is not None else None,
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens,
                    "model": model,
                    "seed": seed if seed is not None else None,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature
                }
                logger.debug("Successfully got completion from Azure OpenAI")
                return (results, response_text)
                
            except Exception as e:
                logger.warning("Azure attempt %d failed: %s", attempt + 1, e)
                if attempt < self.default_max_attempts - 1:
                    self.refresh_token()
                else:
                    raise
    
    def _get_gemini_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = "gemini-1.5-pro",
                         temperature: Optional[float] = 0.7,
                         max_tokens: Optional[int] = 2048,
                         top_p: Optional[float] = 0.95,
                         top_k: Optional[int] = 40,
                         json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from Google Gemini model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt
            prompt (str): Input prompt text
            system_prompt (Optional[str]): System instructions to guide the model
            model (Optional[str]): Google Gemini model to use (gemini-1.5-pro, gemini-1.5-flash, etc.)
            temperature (Optional[float]): Sampling temperature (0-1)
            max_tokens (Optional[int]): Maximum tokens in response
            top_p (Optional[float]): Nucleus sampling parameter
            top_k (Optional[int]): Top-k sampling parameter
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys
                - The generated completion as text
        """
        if not self.gemini_api_key:
            raise ValueError("Google Vertex API key is not set. Please set the GEMINI_API_KEY environment variable.")
        client = genai.Client(api_key=self.gemini_api_key)
        try:
            if json_schema:
                # Create the prompt with system instructions if provided
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                        response_schema=json_schema,
                    )
                )
                response_format = "json_schema"
            else:
                # Create the prompt with system instructions if provided
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_tokens,
                        response_mime_type="text/plain",
                    )
                )
                if re.search(r'\bJSON\b', prompt, re.IGNORECASE):
                    response_format = "json_object"
                else:
                    response_format = "text"
            
            logger.debug(f"Response format: {response_format}")
            logger.info(f"Initiating response parsing")
            if response_format == "json_object":
                response_text = json.loads(response.text)
                
            elif response_format == "json_schema":
                response_text = response.parsed

            elif response_format == "text":
                response_text = response.text
            
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
            
            logger.info(f"Response parsed successfully")
            # TO DO - to include perplexity, statistics and specific output parser for different output type.

            # Create a standardized result dictionary
            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response_text,
                "perplexity": None,  # Gemini doesn't provide log probabilities for perplexity calculation
                "tokens_in": None,    # Estimate if available
                "tokens_out": None,  # Estimate if available
                "model": model,
                "seed": None,        # Gemini doesn't provide a seed
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            
            logger.debug("Successfully got completion from Google Gemini")
            return (result, response_text)
            
        except Exception as e:
            logger.error(f"Error getting completion from Google Gemini: {e}")
            raise
            
    def _get_claude_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = "claude-3-opus-20240229",
                         temperature: Optional[float] = 0.7,
                         max_tokens: Optional[int] = 4096,
                         top_p: Optional[float] = 0.95,
                         json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from Anthropic Claude model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt
            prompt (str): Input prompt text
            system_prompt (Optional[str]): System instructions to guide the model
            model (Optional[str]): Claude model to use (claude-3-opus-20240229, claude-3-sonnet-20240229, etc.)
            temperature (Optional[float]): Sampling temperature (0-1)
            max_tokens (Optional[int]): Maximum tokens in response
            top_p (Optional[float]): Nucleus sampling parameter
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys
                - The generated completion as text
        """
        if not self.claude_api_key:
            raise ValueError("Anthropic Claude API key is not set. Please set the CLAUDE_API_KEY environment variable.")
        
        try:
            # Initialize the Claude client
            client = Anthropic(api_key=self.claude_api_key)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Send the request to Claude API
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=messages,
                system=system_prompt if system_prompt else None
            )
            
            # Extract the response text
            response_text = response.content
            
            # Calculate usage 
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Create a standardized result dictionary
            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response_text,
                "perplexity": None,  # Claude doesn't provide log probabilities for perplexity calculation
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "model": model,
                "seed": None,        # Claude doesn't provide a seed
                "top_p": top_p,
                "temperature": temperature
            }
            
            logger.debug("Successfully got completion from Anthropic Claude")
            return (result, response_text)
            
        except Exception as e:
            logger.error(f"Error getting completion from Anthropic Claude: {e}")
            raise

    def _get_hf_completion(self,
                         prompt_id: int,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         model: Optional[str] = None,
                         temperature: Optional[float] = 1,
                         max_tokens: Optional[int] = 2000,
                         top_p: Optional[float] = 1,
                         top_k: Optional[float] = 10,
                         frequency_penalty: Optional[float] = 1.3,
                         num_beam: Optional[int] = None,
                         logprobs: Optional[bool] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from HuggingFace model."""
        # if system_prompt:
        #     messages = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": prompt}
        #     ]
        # else:
        #     messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            messages = f"User: {prompt}\nAssistant:"
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        orig_model = model
        
        try:
            # Use pre-loaded models if available (for default models)
            if model is None and self.hf_completion_model is not None and self.hf_completion_tokenizer is not None:
                logger.debug("Using pre-loaded HF completion model and tokenizer")
                hf_model = self.hf_completion_model.to(device)
                hf_tokenizer = self.hf_completion_tokenizer
            else:
                # Fallback to loading on-demand for non-default models
                if model:
                    model_path = self.hf_model_dir / model
                else:
                    model_path = self.hf_model_dir / self.default_hf_completion_model
                
                model_path_str = str(model_path)
                logger.warning("Pre-loaded HF completion model not available, loading on-demand (may cause VMS issues)")
                logger.info(f"Attempting to load local HuggingFace completion model from path: {model_path_str}")

                if not model_path.is_dir():
                    error_msg = f"Local HuggingFace completion model path is not a directory or does not exist: {model_path_str}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                logger.debug(f"Path {model_path_str} is a directory. Attempting to load tokenizer and model.")
                hf_tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
                hf_model = AutoModelForCausalLM.from_pretrained(model_path_str, local_files_only=True)
                hf_model = hf_model.to(device)
                logger.info(f"Successfully loaded local HuggingFace completion model and tokenizer from: {model_path_str}")
                
                if hf_tokenizer.pad_token is None:
                    hf_tokenizer.pad_token = hf_tokenizer.eos_token

            # Customised the stop token id
            user_token_ids = hf_tokenizer("User:", add_special_tokens=False).input_ids
            if len(user_token_ids) == 1:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids
            else:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids[1:]
            # Craft the message template
            # message_template = hf_tokenizer.apply_chat_template(messages,
            #                                                     tokenize=False,
            #                                                     add_generation_prompt=True)
            # Tokenize the message template
            model_inputs = hf_tokenizer(messages, return_tensors="pt", padding=True).to(hf_model.device)
            generation_config = GenerationConfig(
                max_new_tokens = max_tokens,  
                do_sample = True if num_beam is None else False,  # Enable sampling
                num_beams = num_beam if num_beam is not None else 1,  # Use beam search
                temperature=temperature, 
                top_k=top_k,
                top_p=top_p,     
                repetition_penalty = frequency_penalty,  # Penalize repeated tokens #SC to double check
                pad_token_id=hf_tokenizer.pad_token_id,
                bos_token_id=hf_tokenizer.bos_token_id,
                eos_token_id=combined_eos_ids
            )

            # Start model inferencing
            with torch.no_grad():
                generated_ids = hf_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=generation_config
                )
                # Create labels from generated ids
                labels = generated_ids.clone()
                # Mask out the prompt tokens when computing the loss and perplexity score
                for i, input_id in enumerate(model_inputs.input_ids):
                    prompt_length = input_id.size(0)
                    labels[i, :prompt_length] = -100    
                outputs = hf_model(generated_ids, labels = labels)
                loss_value = outputs.loss.item()
                perplexity_score = self._calculate_perplexity(loss_value)
            
            # Remove prompt tokens from the generated sequence
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up stop tokens from the response
            stop_phrases = ["\nUser:", "User:", "\nHuman:", "Human:"]
            for stop_phrase in stop_phrases:
                if response.endswith(stop_phrase):
                    response = response[:-len(stop_phrase)].rstrip()
                    break
            # Also handle cases where stop phrase appears anywhere at the end
            response = re.sub(r'\s*(User|Human):\s*$', '', response).rstrip()
            
            # Calculate token counts
            tokens_in  = len(model_inputs.input_ids[0])
            tokens_out = len(generated_ids[0])
        
            results = {
                "prompt_id": prompt_id if prompt_id is not None else None,
                "prompt": prompt,
                "response": response,
                "perplexity": perplexity_score if logprobs else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": orig_model,
                "seed": None,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            logger.debug("Successfully got completion from HuggingFace model")
            return (results, response)
            
        except Exception as e:
            logger.error(f"Failed to get completion from HuggingFace model: {e}")
            raise
            
    def _get_mlx_completion(self,
                             prompt_id: int,
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             model: Optional[str] = None,
                             temperature: Optional[float] = 0.7,
                             max_tokens: Optional[int] = 2000,
                             top_p: Optional[float] = 1,
                             top_k: Optional[int] = 10,
                            #  frequency_penalty: Optional[float] = 1.1,
                            #  num_beam: Optional[int] = None,
                             logprobs: Optional[bool] = None) -> Tuple[Dict[str, Any], str]:
        """Get completion from a model loaded with `mlx_lm`.

        This path enables lightweight inference on Apple-Silicon or CPU-only hosts while
        preserving the same return schema as other provider helpers.
        """
        if mlx_load is None or mlx_generate is None:
            raise ImportError("mlx_lm package is required for MLX inference. Install with `pip install mlx-lm`. ")

        device = (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )

        # Compose chat-style prompt similar to HF path
        if system_prompt:
            prompt_template = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        else:
            prompt_template = f"User: {prompt}\nAssistant:"

        try:
            model_name = model or self.default_hf_completion_model
            cache_key = f"mlx::{model_name}"

            # Re-use cached model if available
            if cache_key in self.completion_model_cache:
                mlx_model, mlx_tokenizer = self.completion_model_cache[cache_key]
            else:
                mlx_model, mlx_tokenizer = mlx_load(model_name)
                mlx_model.eval()
                # Some MLX models may not implement .to(), so guard with try/except
                try:
                    mlx_model.to(device)
                except Exception:
                    pass
                self._manage_model_cache(cache_key, (mlx_model, mlx_tokenizer), self.completion_model_cache)

            # Text generation
            response_text = mlx_generate(
                mlx_model,
                mlx_tokenizer,
                prompt=prompt_template,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Perplexity calculation via loss
            enc = mlx_tokenizer(prompt_template, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            with torch.no_grad():
                outputs = mlx_model(input_ids=input_ids, labels=input_ids)
                loss_val = outputs.loss.item()
            perplexity_score = self._calculate_perplexity(loss_val)

            tokens_in = len(input_ids[0])
            tokens_out = len(mlx_tokenizer.encode(response_text))

            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response_text,
                "perplexity": perplexity_score if logprobs else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": model_name,
                "seed": None,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }

            logger.debug("Successfully got completion from MLX model")
            return (result, response_text)

        except Exception as e:
            logger.error(f"Failed to get completion from MLX model: {e}")
            raise

    def get_embeddings(self, 
              text: Union[str, List[str]], 
              model: Optional[str] = None,
              output_format: Optional[str] = "Array",
              batch_size: Optional[int] = None,
              max_tokens_per_batch: int = 8000,
              buffer_ratio: float = 0.9):
        """Get embeddings using specified model or fall back to HuggingFace model.

        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (Optional[str]): Specific model to use (e.g., "text-embedding-ada-002", "text-embedding-gecko", "claude-3-embedding")
                              If None or invalid, falls back to initialized HuggingFace model
            output_format (str): Output format, either "Array" or "List"
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                 based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)

        Returns:
            Union[List[List[float]], np.ndarray]: Embedding vectors in the requested format
        """            
        try:
            # Verify HuggingFace fallback model is available
            if not self._validate_model(self.default_hf_embedding_model, "embedding", "huggingface"):
                raise ValueError(f"Fallback HuggingFace model '{self.default_hf_embedding_model}' is not available")

            response = None
            # If specific model provided, try to use it
            if model:
                # Try Azure OpenAI first if it's an Azure model
                if self._validate_model(model, "embedding", "azure_openai"):
                    try:
                        logger.info(f"Using Azure OpenAI model '{model}' for embeddings")
                        # Use batch processing for both single and multiple texts
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_azure_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response, list):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Azure OpenAI embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try Google Vertex if it's a Vertex model
                elif self._validate_model(model, "embedding", "vertex") and not response:
                    try:
                        logger.info(f"Using Google Vertex model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_vertex_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response[0], list):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Google Vertex embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try Anthropic if it's an Anthropic model
                elif self._validate_model(model, model_type="embedding", provider="anthropic") and not response:
                    try:
                        logger.info(f"Using Anthropic model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_anthropic_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and not isinstance(response[0], list):
                            response = [response]
                    except Exception as e:
                        logger.error(f"Anthropic embeddings failed due to error {e}, falling back to HuggingFace default model")
                
                # Try HuggingFace if it's a HF model
                elif self._validate_model(model, model_type="embedding", provider="huggingface") and not response:
                    try:
                        logger.info(f"Using HuggingFace model '{model}' for embeddings")
                        # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                        kwargs = {
                            "text": text,
                            "model": model,
                            "max_tokens_per_batch": max_tokens_per_batch,
                            "buffer_ratio": buffer_ratio
                        }
                        if batch_size is not None:
                            kwargs["batch_size"] = batch_size
                            
                        response = self._get_hf_embeddings(**kwargs)
                        # Ensure response is in a list format for standardization
                        if isinstance(text, str) and isinstance(response, np.ndarray) and response.ndim == 1:
                            response = [response.tolist()]
                        elif isinstance(text, str) and isinstance(response, list) and not isinstance(response[0], list):
                            response = [response]
                    except Exception as e:
                        # If model is invalid, log warning and continue to fallback
                        logger.error(f"Specified model '{model}' failed due to error {e}, falling back to HuggingFace default model")
            
            # If no response yet or model not specified, use HuggingFace fallback
            if response is None or not model:
                logger.info(f"Using HuggingFace fallback model '{self.default_hf_embedding_model}' for embeddings")
                # Only pass batch_size if explicitly provided, otherwise let the child function calculate it
                kwargs = {
                    "text": text,
                    "model": self.default_hf_embedding_model,
                    "max_tokens_per_batch": max_tokens_per_batch,
                    "buffer_ratio": buffer_ratio
                }
                if batch_size is not None:
                    kwargs["batch_size"] = batch_size
                    
                response = self._get_hf_embeddings(**kwargs)
                
                # Ensure response is in a list format for standardization
                if isinstance(text, str) and isinstance(response, np.ndarray) and response.ndim == 1:
                    response = [response.tolist()]
                elif isinstance(text, str) and isinstance(response, list) and not isinstance(response[0], list):
                    response = [response]

            # Format response according to requested output format
            if output_format and output_format.lower() == "array":
                # Convert to numpy array for array format
                if isinstance(response, list):
                    return np.array(response)
                elif isinstance(response, np.ndarray):
                    return response
            else: 
                # Convert to list for list format
                if isinstance(response, list):
                    return response
                else:
                    return response.tolist()


        except Exception as e:
            logger.error(f"Embeddings failed: {str(e)}")
            raise
    
    def _get_azure_embeddings(self, text: Union[str, List[str]], model: str, batch_size: Optional[int] = None, max_tokens_per_batch: int = 8000, buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Azure OpenAI API with batch processing support.

        Parameters:
            text (Union[str, List[str]]): Text or list of texts to embed
            model (str): Azure OpenAI model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                     based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        # Handle single text case
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        # Prepare batching
        try:
            # Use the get_tokenisation method for token counting
            token_counts = [self.get_tokenisation(t, model="cl100k_base") for t in texts]
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using character count as proxy for tokens")
            token_counts = [len(t) // 4 for t in texts]  # Rough approximation
        
        # Dynamically calculate batch_size if not provided
        if batch_size is None or batch_size <= 0:
            total_texts = len(texts)
            if total_texts > 1:
                total_tokens = sum(token_counts)
                avg_tokens_per_text = total_tokens / total_texts
                dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                batch_size = dynamic_batch_size
            else:
                batch_size = 1
        
        # Create batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
            if tokens > max_tokens_per_batch:
                logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                # Truncate or handle oversized text
                continue
                
            if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                if current_batch:  # Add the current batch if not empty
                    batches.append(current_batch)
                current_batch = [t]
                current_tokens = tokens
            else:
                current_batch.append(t)
                current_tokens += tokens
        
        if current_batch:  # Add the last batch if not empty
            batches.append(current_batch)
        
        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        all_embeddings = []
        
        for batch_idx, batch_texts in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
        
            for attempt in range(self.default_max_attempts):
                try:
                    # Refresh token and create client
                    access_token = self.refresh_token()
                    if not access_token:
                        raise ValueError("Failed to refresh Azure AD token")
                        
                    client = AzureOpenAI(
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint,
                        azure_ad_token=access_token
                    )
                    
                    # Get embeddings from Azure OpenAI
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts,
                        extra_headers={'x-correlation-id': str(uuid.uuid4()), 'x-subscription-key': self.subscription_key}
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == self.default_max_attempts - 1:
                        logger.error(f"Failed to get embeddings for batch {batch_idx+1} after {self.default_max_attempts} attempts: {e}")
                        # Try fallback to HuggingFace if available
                        try:
                            logger.warning(f"Attempting fallback to HuggingFace model for batch {batch_idx+1}")
                            fallback_embeddings = self._get_hf_embeddings(batch_texts, self.default_hf_embedding_model)
                            if isinstance(fallback_embeddings, np.ndarray):
                                all_embeddings.extend(fallback_embeddings)
                            else:
                                all_embeddings.extend(fallback_embeddings)
                        except Exception as fallback_error:
                            logger.error(f"Fallback to HuggingFace failed: {fallback_error}")
                            raise e  # Re-raise the original error if fallback fails
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for batch {batch_idx+1}, retrying: {e}")
        
        # Return single embedding or list based on input type
        return all_embeddings[0] if is_single_text else all_embeddings

    def _get_vertex_embeddings(self,
                        text: Union[str, List[str]],
                        model: str = "text-embedding-gecko",
                        batch_size: Optional[int] = None,
                        max_tokens_per_batch: int = 8000,
                        buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Google Vertex AI embeddings with batch processing.
        
        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (str): Google Vertex embedding model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                      based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)
            
        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        if not self.gemini_api_key:
            raise ValueError("Google Vertex API key is not set. Please set the GEMINI_API_KEY environment variable.")
            
        try:
            # Handle single text case
            is_single_text = isinstance(text, str)
            texts = [text] if is_single_text else text
            
            # Prepare batching
            try:
                token_counts = [self.get_tokenisation(t, model="gemini-2.5-pro") for t in texts]
            except ImportError:
                logger.warning("tiktoken not installed, using character count as proxy for tokens")
                token_counts = [len(t) // 4 for t in texts]  # Rough approximation
            
            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                total_texts = len(texts)
                if total_texts > 1:
                    total_tokens = sum(token_counts)
                    avg_tokens_per_text = total_tokens / total_texts
                    dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                    batch_size = dynamic_batch_size
                else:
                    batch_size = 1
            
            # Create batches
            batches = []
            current_batch = []
            current_tokens = 0
            
            for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                    # Skip oversized text for now
                    continue
                    
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:  # Add the current batch if not empty
                        batches.append(current_batch)
                    current_batch = [t]
                    current_tokens = tokens
                else:
                    current_batch.append(t)
                    current_tokens += tokens
            
            if current_batch:  # Add the last batch if not empty
                batches.append(current_batch)
            
            logger.info(f"Processing {len(texts)} texts in {len(batches)} batches using Google Vertex")
            
            all_embeddings = []
            
            # Initialize the embedding model once outside the batch loop
            embedding_model = genai.GenerativeModel(model_name=model)
            
            # Process each batch
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
                
                batch_embeddings = []
                for single_text in batch_texts:
                    try:
                        # Generate embedding
                        embedding_response = embedding_model.embed_content(
                            model=model,
                            content=single_text
                        )
                        
                        # Extract and store the embedding vector
                        embedding_vector = embedding_response.embedding
                        batch_embeddings.append(embedding_vector)
                    except Exception as e:
                        logger.error(f"Error generating embedding for text in batch {batch_idx+1}: {str(e)}")
                        # Add a zero vector as placeholder for failed embedding
                        # Get the embedding dimension for this model or use a default
                        embedding_dim = self._get_embedding_dimensions(model)
                        batch_embeddings.append([0.0] * embedding_dim)
                
                all_embeddings.extend(batch_embeddings)
            
            # Return a single embedding for a single input, otherwise return list of embeddings
            if is_single_text:
                embedding_dim = self._get_embedding_dimensions(model)
                return all_embeddings[0] if all_embeddings else [0.0] * embedding_dim
            else:
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Error generating Google Vertex embeddings: {str(e)}")
            raise
    
    def _get_anthropic_embeddings(self,
                          text: Union[str, List[str]],
                          model: str = "claude-3-embedding",
                          batch_size: Optional[int] = None,
                          max_tokens_per_batch: int = 8000,
                          buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]]]:
        """Get embeddings using Anthropic Claude embedding model with batch processing.
        
        Parameters:
            text (Union[str, List[str]]): Text(s) to embed
            model (str): Anthropic Claude embedding model to use
            batch_size (Optional[int]): Maximum number of texts per batch. If None, will be dynamically calculated
                                      based on token counts and max_tokens_per_batch
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)
            
        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s)
        """
        if not self.claude_api_key:
            raise ValueError("Anthropic API key is not set. Please set the CLAUDE_API_KEY environment variable.")
            
        try:
            # Initialize the Anthropic client
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            
            # Handle single text case
            is_single_text = isinstance(text, str)
            texts = [text] if is_single_text else text
            
            # Prepare batching
            try:
                token_counts = [self.get_tokenisation(t, model="claude-3-7-sonnet-20250219") for t in texts]
            except ImportError:
                logger.warning("tiktoken not installed, using character count as proxy for tokens")
                token_counts = [len(t) // 4 for t in texts]  # Rough approximation
            
            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                total_texts = len(texts)
                if total_texts > 1:
                    total_tokens = sum(token_counts)
                    avg_tokens_per_text = total_tokens / total_texts
                    dynamic_batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {dynamic_batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                    batch_size = dynamic_batch_size
                else:
                    batch_size = 1
            
            # Create batches
            batches = []
            current_batch = []
            current_tokens = 0
            
            for idx, (t, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.warning(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), truncating")
                    # Skip oversized text for now
                    continue
                    
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:  # Add the current batch if not empty
                        batches.append(current_batch)
                    current_batch = [t]
                    current_tokens = tokens
                else:
                    current_batch.append(t)
                    current_tokens += tokens
            
            if current_batch:  # Add the last batch if not empty
                batches.append(current_batch)
            
            logger.info(f"Processing {len(texts)} texts in {len(batches)} batches using Anthropic")
            
            all_embeddings = []
            
            # Process each batch
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts")
                
                batch_embeddings = []
                for single_text in batch_texts:
                    try:
                        # Generate embedding
                        embedding_response = client.embeddings.create(
                            model=model,
                            input=single_text
                        )
                        
                        # Extract and store the embedding vector
                        embedding_vector = embedding_response.embeddings[0].embedding
                        batch_embeddings.append(embedding_vector)
                    except Exception as e:
                        logger.error(f"Error generating embedding for text in batch {batch_idx+1}: {str(e)}")
                        # Add a zero vector as placeholder for failed embedding
                        # Get the embedding dimension for this model or use a default
                        embedding_dim = self._get_embedding_dimensions(model)
                        batch_embeddings.append([0.0] * embedding_dim)
                
                all_embeddings.extend(batch_embeddings)
            
            # Return a single embedding for a single input, otherwise return list of embeddings
            if is_single_text:
                embedding_dim = self._get_embedding_dimensions(model)
                return all_embeddings[0] if all_embeddings else [0.0] * embedding_dim
            else:
                return all_embeddings
                
        except Exception as e:
            logger.error(f"Error generating Anthropic embeddings: {str(e)}")
            raise
    
    def _get_hf_embeddings(self,
              text: Union[str, List[str]],
              model: str,
              batch_size: Optional[int] = None,
              max_tokens_per_batch: int = 8000,
              buffer_ratio: float = 0.9) -> Union[List[float], List[List[float]], np.ndarray]:
        """Get embeddings using HuggingFace model with simple download-once, load-locally strategy.

        Parameters:
            text (Union[str, List[str]]): Text to embed
            model (str): HuggingFace model name
            batch_size (Optional[int]): Batch size for processing
            max_tokens_per_batch (int): Maximum tokens per batch
            buffer_ratio (float): Safety buffer for batch sizing

        Returns:
            Union[List[float], List[List[float]], np.ndarray]: Embedding vector(s)
        """
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text
        
        try:
            # Use pre-loaded model if available, reload if needed, otherwise fall back to cache/load
            if self.hf_embedding_model is not None:
                logger.debug("Using pre-loaded HF embedding model")
                embed_model = self.hf_embedding_model
            elif model == self.default_hf_embedding_model:
                # Model was unloaded due to memory pressure, try to reload
                logger.info("Default embedding model was unloaded, attempting to reload")
                self.reload_models_if_needed()
                if self.hf_embedding_model is not None:
                    embed_model = self.hf_embedding_model
                else:
                    raise RuntimeError("Failed to reload embedding model after memory cleanup")
            else:
                # Fallback to original cache logic for non-default models
                model_key = str(model)
                if model_key in self.embedding_model_cache:
                    logger.debug(f"Using cached embedding model: {model_key}")
                    embed_model = self.embedding_model_cache[model_key]
                else:
                    logger.warning("Pre-loaded HF embedding model not available, loading on-demand (may cause VMS issues)")
                    # Check if model exists locally, if not download it
                    local_model_path = self.hf_model_dir / model
                    logger.info("Model path exist ", os.path.exists(local_model_path))
                    logger.info(f"Model path type: {type(local_model_path)}")
                    logger.info(f"Loading embedding model from local directory: {local_model_path}")
                    
                    # Clear memory before loading new model
                    self._cleanup_memory()
                    
                    try:
                        # Configure model loading to prevent VMS explosion
                        import torch
                        if torch.backends.mps.is_available():
                            # Set MPS memory fraction to prevent excessive virtual memory allocation
                            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable high watermark
                            os.environ['MPS_DISABLE_LARGE_MEMORY_ALLOCATION'] = '1'  # Disable large allocations
                        
                        # Load model with memory constraints
                        embed_model = SentenceTransformer(
                            str(local_model_path), 
                            local_files_only=True,
                            device='cpu'
                        )
                        
                    except Exception as load_error:
                        logger.warning(f"Failed to load embedding model: {load_error}")
                        try:
                            transformer = models.Transformer(
                                model_name_or_path=str(local_model_path), 
                                local_files_only=True,
                                device='cpu'
                            )
                            pooling = models.Pooling(transformer.get_word_embedding_dimension(), 
                                                    pooling_mode_mean_tokens=True,
                                                    pooling_mode_cls_token=False, 
                                                    pooling_mode_max_tokens=False)
                            normalize = models.Normalize()
                            embed_model = SentenceTransformer(modules=[transformer, pooling, normalize])
                        except Exception as alternative_error:
                            logger.warning(f"Failed to apply pooling: {alternative_error}")
                            try: 
                                embed_model = SentenceTransformer(
                                    model, 
                                    cache_dir = self.hf_model_dir,
                                    device='cpu'
                                )
                            except Exception as final_error:
                                logger.warning(f"Failed to download embedding model: {final_error}")
                                raise
                    
                    # Check virtual memory after model loading
                    self._cleanup_memory()  # This will check VMS and warn/terminate if needed
                    
                    # Add to cache
                    self._manage_model_cache(model_key, embed_model, self.embedding_model_cache)
                    self.current_model = model_key
                
            # Enhanced batch processing with memory management
            if batch_size is None:
                # Ultra-conservative adaptive batch size for extreme memory management
                avg_text_length = sum(len(t) for t in texts) / len(texts)
                if avg_text_length > 2000:  # Long texts
                    batch_size = 1  # Ultra conservative
                elif avg_text_length > 1000:  # Medium texts
                    batch_size = 2  # Reduced further
                else:  # Short texts
                    batch_size = 4  # Reduced further
            
            # Ensure batch_size doesn't exceed total texts
            batch_size = min(batch_size, len(texts))
            
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i, batch_start in enumerate(range(0, len(texts), batch_size)):
                batch_end = min(batch_start + batch_size, len(texts))
                batch = texts[batch_start:batch_end]
                
                try:
                    # Check virtual memory before processing each batch
                    if i % 5 == 0:  # Check every 5 batches to avoid overhead
                        self._cleanup_memory()  # This will check VMS and terminate if needed
                    
                    # Force model to CPU and clear GPU memory before embedding
                    if hasattr(embed_model, 'to'):
                        embed_model.to('cpu')
                    
                    batch_embeddings = embed_model.encode(
                        batch, 
                        convert_to_numpy=True,
                        show_progress_bar=False,  # Disable progress bar to reduce memory
                        device='cpu'  # Explicitly force CPU
                    )
                    
                    # Convert to numpy immediately and clear intermediate tensors
                    if hasattr(batch_embeddings, 'cpu'):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    if batch_embeddings.ndim == 1:
                        all_embeddings.append(batch_embeddings.copy())  # Ensure copy to avoid references
                    else:
                        all_embeddings.extend([emb.copy() for emb in batch_embeddings])  # Deep copy each embedding
                    
                    # Clear the batch_embeddings variable immediately
                    del batch_embeddings
                    
                    # Strategic memory cleanup for larger intervals
                    if (i + 1) % 10 == 0:  # Clean up every 10 batches to reduce overhead
                        self._cleanup_memory()
                        logger.debug(f"Memory cleanup after batch {i + 1}")
                    
                    # Progress logging for large datasets
                    if total_batches > 5 and (i + 1) % max(1, total_batches // 4) == 0:
                        logger.info(f"Processed embedding batch {i + 1}/{total_batches}")
                        
                except RuntimeError as mem_error:
                    if "out of memory" in str(mem_error).lower():
                        logger.warning(f"Memory error in batch {i + 1}, reducing batch size and retrying")
                        # Clear memory before retry
                        self._cleanup_memory()
                        
                        # Retry with much smaller batch size
                        smaller_batch_size = 1  # Process one at a time when memory fails
                        for j in range(batch_start, batch_end, smaller_batch_size):
                            try:
                                small_batch = texts[j:j + smaller_batch_size]
                                
                                # Force CPU for retry
                                if hasattr(embed_model, 'to'):
                                    embed_model.to('cpu')
                                
                                small_embeddings = embed_model.encode(
                                    small_batch, 
                                    convert_to_numpy=True,
                                    show_progress_bar=False,
                                    device='cpu'
                                )
                                
                                # Convert and copy immediately
                                if hasattr(small_embeddings, 'cpu'):
                                    small_embeddings = small_embeddings.cpu().numpy()
                                
                                if small_embeddings.ndim == 1:
                                    all_embeddings.append(small_embeddings.copy())
                                else:
                                    all_embeddings.extend([emb.copy() for emb in small_embeddings])
                                
                                # Clear immediately
                                del small_embeddings
                                
                                # Clean up after each single embedding to prevent accumulation
                                if j % 5 == 0:  # Every 5 single embeddings
                                    self._cleanup_memory()
                                    
                            except RuntimeError as single_mem_error:
                                if "out of memory" in str(single_mem_error).lower():
                                    logger.error(f"Failed to process even single text at index {j}. Skipping this text.")
                                    # Add zero embedding as fallback
                                    embedding_dim = self._get_embedding_dimensions(model)
                                    all_embeddings.append(np.zeros(embedding_dim))
                                else:
                                    raise single_mem_error
                    else:
                        raise
            
            # Aggressive cleanup after processing all batches
            self._cleanup_memory()
            embed_model = None
            
            # Check if we should recreate the model to prevent memory accumulation
            # if total_batches > 20:  # For large document processing, recreate model
            #     logger.info(f"Recreating embedding model after {total_batches} batches to prevent memory accumulation")
            #     if model_key in self.embedding_model_cache:
            #         del self.embedding_model_cache[model_key]
            #         self._cleanup_memory(force=True)
            
            
            # Return based on input type
            if is_single_text:
                return all_embeddings[0] if all_embeddings else np.zeros(384)  # Default embedding dim
            else:
                return np.array(all_embeddings)
                
        except Exception as e:
            logger.error(f"Failed to get embeddings using HuggingFace models: {e}")
            # Return zeros as fallback
            # Attempt to get embedding_dim; if model string is problematic, use a default.
            try:
                embedding_dim = self._get_embedding_dimensions(model)
            except Exception as dim_exc:
                logger.error(f"Could not determine embedding dimension for model '{model}' during error handling: {dim_exc}. Using default from hf provider.")
                embedding_dim = self.default_embedding_dimensions.get("huggingface", 384) # Fallback dimension

            if is_single_text:
                return np.zeros(embedding_dim)
            else:
                return np.array([np.zeros(embedding_dim) for _ in range(len(texts))])

    def _get_embedding_dimensions(self, model_name: str) -> int:
        """Get the embedding dimensions for a specific model.

        This method attempts to determine the embedding dimensions in the following order:
        1. Check the cache for previously determined dimensions.
        2. Determine the model provider.
        3. Check if the model has specific dimensions in the config_model.json.
        4. If the provider is HuggingFace and dimensions are not in config_model.json,
           try to determine dimensions dynamically by loading the model.
        5. Use provider-specific defaults if dimensions are still not found.
        6. Use an absolute fallback if all other methods fail.

        Parameters:
            model_name (str): Name of the embedding model.

        Returns:
            int: Embedding dimensions for the model.
        """
        # 1. Check cache
        if model_name in self.embedding_dimensions_cache:
            return self.embedding_dimensions_cache[model_name]

        embedding_dim: Optional[int] = None
        provider: Optional[str] = None

        # 2. Determine provider
        for p_name in ["azure_openai", "anthropic", "vertex", "huggingface"]:
            if self._validate_model(model_name, "embedding", p_name):
                provider = p_name
                logger.debug(f"Determined provider for {model_name}: {provider}")
                break
        
        if not provider:
            logger.warning(f"Could not determine provider for model: {model_name}. Will rely on fallbacks.")

        # 3. Check config_model.json for pre-configured dimensions
        if "model_dimensions" in self.model_config and model_name in self.model_config["model_dimensions"]:
            embedding_dim = self.model_config["model_dimensions"][model_name]
            logger.info(f"Using pre-configured embedding dimension for {model_name} from config_model.json: {embedding_dim}")
        
        # 4. If provider is HuggingFace and dimension not in config, try dynamic determination
        elif provider == "huggingface" and embedding_dim is None:
            try:
                model_path = self.hf_model_dir / model_name
                if not model_path.is_dir():
                    logger.warning(
                        f"Local directory {model_path} for HuggingFace model {model_name} not found. "
                        f"Cannot dynamically determine dimension for this path. Will rely on defaults."
                    )
                else:
                    logger.info(f"Attempting to dynamically determine embedding dimension for HuggingFace model path: {model_path}")
                    try:
                        embed_model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=True)
                        test_embedding = embed_model.encode("test")
                        embedding_dim = len(test_embedding)
                        logger.info(f"Dynamically determined embedding dimension for HF model {model_name} ({model_path}): {embedding_dim}")
                    except Exception as model_error:
                        logger.warning(f"Failed to load SentenceTransformer model from {model_path}: {model_error}. Using default dimension.")
                        raise model_error  # Re-raise to trigger outer exception handling
            except Exception as e:
                logger.warning(
                    f"Could not dynamically determine embedding dimensions for HuggingFace model {model_name} "
                    f"from path {self.hf_model_dir / model_name}: {e}. Will use default for provider."
                )

        # 5. Use provider-specific defaults if dimension still not found
        if embedding_dim is None and provider:
            embedding_dim = self.default_embedding_dimensions.get(provider)
            if embedding_dim:
                logger.info(f"Using default embedding dimension for provider {provider} ({model_name}): {embedding_dim}")
            else:
                logger.warning(f"No default embedding dimension found for provider {provider} ({model_name}).")

        # 6. Final fallback
        if embedding_dim is None:
            embedding_dim = 384  # Absolute fallback
            logger.warning(f"Could not determine embedding dimensions for {model_name} through any method, using absolute fallback: {embedding_dim}")
            
        # Cache the result for future use
        self.embedding_dimensions_cache[model_name] = embedding_dim
        return embedding_dim
    
    def get_reranking(self,
                    query: str,
                    passages: List[str],
                    model: Optional[str] = None,
                    batch_size: int = 32,
                    return_scores: bool = True) -> Union[List[Tuple[str, float]], List[str]]:
        """Get reranking scores using cross-encoder models.
        
        Parameters:
            query (str): The query to rank passages against
            passages (List[str]): List of passages to be reranked
            model (Optional[str]): Specific cross-encoder model to use (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
                                 If None or invalid, falls back to default cross-encoder model
            batch_size (int): Batch size for processing passages
            return_scores (bool): Whether to return scores with passages
            
        Returns:
            Union[List[Tuple[str, float]], List[str]]: Reranked passages with scores (if return_scores=True) or just passages
        """
        if model:
            model = Path.cwd() / "local_models" / model
        else:
            model = Path.cwd() / "local_models" / self.default_hf_reranker_model
            
        # Use MPS if available, else CPU
        device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device for CrossEncoder: {device}")

        try:
            reranker = CrossEncoder(model, device=device)
            logger.info(f"Loaded cross-encoder model: {model} on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model '{model}' on {device}: {e}")
            raise
        
        try:
            # Create query-passage pairs for scoring
            query_passage_pairs = [[query, passage] for passage in passages]
            
            # Get scores from the model
            scores = reranker.predict(query_passage_pairs, batch_size=batch_size)
        
            # Combine passages with scores
            passage_score_pairs = list(zip(passages, scores))
        
            # Sort by score in descending order
            reranked_pairs = sorted(passage_score_pairs, key=lambda x: x[1], reverse=True)
        
            if return_scores:
                return reranked_pairs
            else:
                return [pair[0] for pair in reranked_pairs]
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise

    def get_agentic(self,
                    prompt: str,
                    prompt_id: Optional[int] = None,
                    system_prompt: Optional[str] = None,
                    model: Optional[str] = None,
                    return_full_response: Optional[bool] = False,
                    **kwargs) -> Union[StandardizedAgenticResponse, str]:
        """
        Orchestrates an agentic interaction with an LLM, potentially involving multiple
        tool calls via MCP, and returns the final response.

        Args:
            prompt (str): The initial user prompt.
            prompt_id (Optional[int]): An optional ID for the prompt.
            system_prompt (Optional[str]): An optional system prompt to guide the model.
            model (Optional[str]): The specific model to use. If None, a default will be chosen.
            return_full_response (Optional[bool]): If True, returns a StandardizedAgenticResponse TypedDict.
                                                 If False (default), returns only the final response string.
            **kwargs: Additional keyword arguments passed to the underlying provider-specific methods
                      (e.g., temperature, max_tokens, top_p, top_k, max_tool_calls).

        Returns:
            Union[StandardizedAgenticResponse, str]:
                If return_full_response is True, returns a StandardizedAgenticResponse TypedDict
                containing detailed information about the interaction, including the full
                standardized conversation history and token counts.
                If return_full_response is False, returns the final textual response string from the LLM.
        """
        logger.info(f"get_agentic called with model: {model}, prompt: '{prompt[:100]}...'")

        if model is None:
            try: # Default to a capable Anthropic model if available
                model = self.model_config["validation_rules"]["models"]["completion"]["anthropic"][0]
                logger.warning(f"Model not specified for get_agentic, defaulting to Anthropic model: {model}")
            except (KeyError, IndexError, TypeError):
                logger.warning("No default Anthropic model found in config for get_agentic. Trying general default.")
                model = self.default_completion_model
                if not model:
                    raise ValueError("No model specified and no default agentic model could be determined for get_agentic.")
                logger.warning(f"Falling back to general default completion model for get_agentic: {model}")

        # Prepare initial messages list, common for chat-based models
        initial_messages = [{"role": "user", "content": prompt}]

        agentic_kwargs = {
            "prompt_id": prompt_id,
            "system_prompt": system_prompt,
            "return_full_response": return_full_response,
            **kwargs
        }

        # Route to the correct provider-specific agentic method
        if self._validate_model(model, "completion", "anthropic"):
            logger.debug(f"Routing to _get_claude_agentic for model {model}")
            return self._get_claude_agentic(model=model, initial_messages=initial_messages, **agentic_kwargs)
        elif self._validate_model(model, "completion", "azure_openai"):
            logger.debug(f"Routing to _get_azure_agentic for model {model}")
            return self._get_azure_agentic(model=model, initial_messages=initial_messages, **agentic_kwargs)
        elif self._validate_model(model, "completion", "vertex"): # Gemini
            logger.debug(f"Routing to _get_gemini_agentic for model {model}")
            return self._get_gemini_agentic(model=model, initial_messages=initial_messages, **agentic_kwargs)
        elif self._validate_model(model, "completion", "huggingface"):
            logger.debug(f"Routing to _get_hf_agentic for model {model}")
            # For HF, initial_messages might need to be converted to a flat prompt string
            return self._get_hf_agentic(model=model, initial_prompt_text=prompt, **agentic_kwargs)
        else:
            logger.error(f"Unsupported model or provider for get_agentic: {model}. Check config_model.json.")
            raise ValueError(f"Model {model} is not configured for agentic calls with a known provider.")

    def _get_claude_agentic(self, model: str, initial_messages: List[Dict[str, str]], **kwargs) -> Union[StandardizedAgenticResponse, str]:
        """
        Handles agentic interaction with an Anthropic Claude model, including MCP tool calls.
        Returns a standardized response structure if `return_full_response` is True in kwargs.

        Args:
            model (str): The Claude model name.
            initial_messages (List[Dict[str, str]]): The initial messages for the conversation.
            **kwargs: See `get_agentic` for other relevant kwargs like `return_full_response`,
                      `prompt_id`, `system_prompt`, `temperature`, `max_tokens`, etc.

        Returns:
            Union[StandardizedAgenticResponse, str]: Standardized response or final text string.
        """
        logger.info(f"Executing Claude agentic call for model {model}")
        self.mcp_client_manager.ensure_initialized()

        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 2048)
        top_p = kwargs.get('top_p', None)
        system_prompt = kwargs.get('system_prompt', None)
        prompt_id = kwargs.get('prompt_id', None)
        initial_prompt_text = initial_messages[0]['content'] if initial_messages and initial_messages[0]['role'] == 'user' else ""
        return_full_response = kwargs.get('return_full_response', False)

        client = Anthropic(api_key=self.claude_api_key)
        if not self.claude_api_key:
            raise ValueError("Anthropic Claude API key is not set.")

        final_response_text = ""
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0
        MAX_TOOL_CALLS = kwargs.get('max_tool_calls', 5)
        tool_calls_count = 0

        # History for Claude API (native format)
        claude_api_history = [msg for msg in initial_messages]
        # History for standardized output
        standardized_history: List[StandardizedMessage] = []
        for msg in initial_messages: # Convert initial messages
            standardized_history.append(StandardizedMessage(role=msg["role"], content=msg["content"]))


        while tool_calls_count < MAX_TOOL_CALLS:
            api_params = {
                "model": model, "max_tokens": max_tokens, "messages": claude_api_history,
                "system": system_prompt, "temperature": temperature,
            }
            if top_p is not None: api_params["top_p"] = top_p

            if self.mcp_client_manager.available_tools:
                # Claude tools are directly compatible with MCP tool schema (name, description, input_schema)
                api_params["tools"] = self.mcp_client_manager.available_tools
                logger.info(f"Claude Agentic: Passing {len(self.mcp_client_manager.available_tools)} tools to Claude API.")
            else:
                logger.info("Claude Agentic: No MCP tools available/loaded to pass to Claude API.")

            logger.debug(f"Claude API call ({tool_calls_count + 1}) with messages: {claude_api_history}")
            response = client.messages.create(**api_params)

            if response.usage:
                accumulated_input_tokens += response.usage.input_tokens
                accumulated_output_tokens += response.usage.output_tokens

            has_tool_use_this_turn = False

            # Process response content blocks for Claude API history and Standardized History
            assistant_api_content_blocks = [] # For Claude API history
            assistant_standardized_content_parts: List[StandardizedMessageContentPart] = [] # For Standardized History

            for content_block in response.content:
                assistant_api_content_blocks.append(content_block.model_dump()) # For Claude's history

                if content_block.type == 'text':
                    logger.debug(f"Claude agentic text response block: {content_block.text}")
                    assistant_standardized_content_parts.append(StandardizedMessageContentPart(type="text", text=content_block.text))

                elif content_block.type == 'tool_use':
                    has_tool_use_this_turn = True
                    tool_name, tool_use_id, tool_input = content_block.name, content_block.id, content_block.input
                    logger.info(f"Claude agentic requested tool: {tool_name} (ID: {tool_use_id}) with input: {tool_input}")

                    assistant_standardized_content_parts.append(StandardizedMessageContentPart(
                        type="tool_use", id=tool_use_id, name=tool_name, input=tool_input
                    ))

                    # Add assistant's response (including tool_use) to Claude API history *before* calling the tool
                    claude_api_history.append({"role": "assistant", "content": assistant_api_content_blocks})
                    # Also add to standardized history
                    standardized_history.append(StandardizedMessage(role="assistant", content=assistant_standardized_content_parts))

                    # Call the tool
                    tool_session = asyncio.run(self.mcp_client_manager.get_tool_session(tool_name))
                    tool_result_content_for_api, is_error_result = "", False
                    tool_result_content_for_standardized: Union[str, Dict[str, Any], List[Any]]

                    if tool_session:
                        try:
                            mcp_tool_result = asyncio.run(tool_session.call_tool(tool_name, arguments=tool_input))
                            # For Claude API, content should be a string or JSON serializable dict/list
                            if isinstance(mcp_tool_result.content, (dict, list)):
                                tool_result_content_for_api = json.dumps(mcp_tool_result.content)
                                tool_result_content_for_standardized = mcp_tool_result.content # Keep original structure
                            else:
                                tool_result_content_for_api = str(mcp_tool_result.content)
                                tool_result_content_for_standardized = str(mcp_tool_result.content)
                            logger.info(f"Tool '{tool_name}' executed. Result preview: {tool_result_content_for_api[:200]}...")
                        except Exception as e:
                            logger.error(f"Error calling MCP tool {tool_name} via Claude Agentic: {e}", exc_info=True)
                            error_msg = f"Error executing tool {tool_name}: {str(e)}"
                            tool_result_content_for_api, is_error_result = error_msg, True
                            tool_result_content_for_standardized = {"error": error_msg}
                    else:
                        error_msg = f"Tool {tool_name} is not available or configured."
                        tool_result_content_for_api, is_error_result = error_msg, True
                        tool_result_content_for_standardized = {"error": error_msg}
                        logger.warning(f"Tool {tool_name} requested by Claude agentic not found.")

                    # Prepare tool result for Claude API history
                    claude_tool_result_msg_content = {"type": "tool_result", "tool_use_id": tool_use_id, "content": tool_result_content_for_api}
                    if is_error_result: claude_tool_result_msg_content["is_error"] = True
                    claude_api_history.append({"role": "user", "content": [claude_tool_result_msg_content]})

                    # Prepare tool result for Standardized History
                    standardized_tool_result_part = StandardizedMessageContentPart(
                        type="tool_result", tool_use_id=tool_use_id, content=tool_result_content_for_standardized
                    )
                    if is_error_result: standardized_tool_result_part["is_error"] = True
                    standardized_history.append(StandardizedMessage(role="tool", name=tool_name, content=[standardized_tool_result_part]))

                    assistant_api_content_blocks = [] # Reset for next potential assistant turn
                    assistant_standardized_content_parts = []
                    break # Break from content_block loop to re-prompt model with tool result

            if not has_tool_use_this_turn:
                # This is the final response from the assistant (no more tool calls this turn)
                if assistant_api_content_blocks: # Should contain only text parts now
                    claude_api_history.append({"role": "assistant", "content": assistant_api_content_blocks})
                if assistant_standardized_content_parts:
                    standardized_history.append(StandardizedMessage(role="assistant", content=assistant_standardized_content_parts))

                final_response_text = " ".join(part.get("text","") for part in assistant_standardized_content_parts if part.get("type") == "text")
                logger.info(f"Claude agentic call finished. Final text: {final_response_text[:200]}...")
                break # Exit the while loop for tool calls

            tool_calls_count += 1
            if tool_calls_count >= MAX_TOOL_CALLS:
                logger.warning("Reached maximum tool calls for Claude agentic interaction.")
                # If the last turn was a tool request, we don't have a final text response from the model yet.
                # The history will reflect the last tool call attempt.
                # We might want to extract any text provided by the assistant *before* the tool call that hit the limit.
                final_response_text = "Reached maximum tool calls."
                # If assistant_standardized_content_parts has text before the tool_use part that caused max_out:
                text_before_max_tool_call = " ".join(p.get("text","") for p in assistant_standardized_content_parts if p.get("type") == "text")
                if text_before_max_tool_call.strip():
                    final_response_text = text_before_max_tool_call.strip() + " (Reached maximum tool calls)"

                # Ensure the final (incomplete) assistant turn is added to history if it contained the maxed-out tool_use
                if assistant_api_content_blocks and not claude_api_history[-1]['role'] == 'assistant':
                     claude_api_history.append({"role": "assistant", "content": assistant_api_content_blocks})
                if assistant_standardized_content_parts and not standardized_history[-1]['role'] == 'assistant':
                     standardized_history.append(StandardizedMessage(role="assistant", content=assistant_standardized_content_parts))
                break

        if return_full_response:
            return StandardizedAgenticResponse(
                prompt_id=prompt_id,
                prompt=initial_prompt_text,
                response=final_response_text,
                full_conversation_history=standardized_history,
                tokens_in=accumulated_input_tokens,
                tokens_out=accumulated_output_tokens,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=kwargs.get('top_k'), # Claude doesn't use top_k directly in messages.create
                tool_calls_made=tool_calls_count
            )
        return final_response_text

    def _get_hf_agentic(self, model_name: str, initial_prompt_text: str, **kwargs) -> Union[StandardizedAgenticResponse, str]:
        """
        Handles agentic interaction with a HuggingFace model, emulating MCP tool calls
        through specific prompting strategies and output parsing.
        Returns a standardized response structure if `return_full_response` is True in kwargs.

        This method relies on the model's ability to follow instructions to format tool
        call requests as JSON and to understand tool results provided in the prompt.

        Args:
            model_name (str): The name of the HuggingFace model (directory name under self.hf_model_dir)
                              or a pre-loaded default model identifier.
            initial_prompt_text (str): The initial user prompt text.
            **kwargs: See `get_agentic` for other relevant kwargs like `return_full_response`,
                      `prompt_id`, `system_prompt`, `temperature`, `max_tokens` (max_new_tokens for HF),
                      `top_p`, `top_k`, `repetition_penalty`, `max_tool_calls`.

        Returns:
            Union[StandardizedAgenticResponse, str]:
                If return_full_response is True, returns a StandardizedAgenticResponse TypedDict.
                If False, returns only the final textual response string.
        """
        logger.info(f"Executing HuggingFace agentic call for model {model_name}")
        self.mcp_client_manager.ensure_initialized()

        # Parameters from kwargs
        prompt_id = kwargs.get('prompt_id')
        system_prompt_content = kwargs.get('system_prompt')
        return_full_response = kwargs.get('return_full_response', False)
        max_tool_calls = kwargs.get('max_tool_calls', 3) # Lower default for HF due to complexity

        # Generation parameters for HF model
        temperature = kwargs.get('temperature', 0.7)
        max_new_tokens = kwargs.get('max_tokens', 512) # Max tokens for each generation step
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 50)
        repetition_penalty = kwargs.get('repetition_penalty', 1.1) # Common param name

        hf_model = None
        hf_tokenizer = None
        actual_model_name_used = model_name

        # --- Model Loading ---
        # Consolidate model loading logic
        if model_name == self.default_hf_completion_model and self.hf_completion_model and self.hf_completion_tokenizer:
            logger.info(f"Using pre-loaded default HF completion model: {model_name}")
            hf_model, hf_tokenizer = self.hf_completion_model, self.hf_completion_tokenizer
            actual_model_name_used = self.default_hf_completion_model
        elif model_name == self.default_hf_reasoning_model and hasattr(self, 'hf_reasoning_model') and self.hf_reasoning_model and hasattr(self, 'hf_reasoning_tokenizer') and self.hf_reasoning_tokenizer:
            logger.info(f"Using pre-loaded default HF reasoning model: {model_name}")
            hf_model, hf_tokenizer = self.hf_reasoning_model, self.hf_reasoning_tokenizer
            actual_model_name_used = self.default_hf_reasoning_model
        elif model_name: # Specific model requested, load on demand
            logger.warning(f"Attempting to load HF model '{model_name}' on-demand for agentic call.")
            model_path = self.hf_model_dir / model_name
            if not model_path.is_dir():
                raise FileNotFoundError(f"HuggingFace model directory not found: {model_path}")
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.float16, device_map="auto"
                )
                if hf_tokenizer.pad_token is None: hf_tokenizer.pad_token = hf_tokenizer.eos_token
                logger.info(f"Successfully loaded HF model '{model_name}' and tokenizer on-demand.")
                actual_model_name_used = model_name
            except Exception as e:
                logger.error(f"Failed to load HF model '{model_name}' on-demand: {e}", exc_info=True)
                raise ValueError(f"Could not load specified HF model: {model_name}") from e
        else: # No specific model, try default completion, then default reasoning
            logger.info("No specific HF model provided for agentic call, trying defaults.")
            if self.hf_completion_model and self.hf_completion_tokenizer:
                hf_model, hf_tokenizer = self.hf_completion_model, self.hf_completion_tokenizer
                actual_model_name_used = self.default_hf_completion_model
                logger.info(f"Using pre-loaded default HF completion model: {actual_model_name_used}")
            elif hasattr(self, 'hf_reasoning_model') and self.hf_reasoning_model and hasattr(self, 'hf_reasoning_tokenizer') and self.hf_reasoning_tokenizer:
                hf_model, hf_tokenizer = self.hf_reasoning_model, self.hf_reasoning_tokenizer
                actual_model_name_used = self.default_hf_reasoning_model
                logger.info(f"Using pre-loaded default HF reasoning model: {actual_model_name_used}")
            else:
                raise ValueError("No suitable default HuggingFace model (completion or reasoning) is pre-loaded/specified for agentic calls.")

        if not hf_model or not hf_tokenizer:
             raise ValueError(f"Failed to obtain a valid HuggingFace model and tokenizer for '{model_name}'.")

        device = hf_model.device

        # --- Histories & Tool Info ---
        standardized_history: List[StandardizedMessage] = []
        if system_prompt_content:
            standardized_history.append(StandardizedMessage(role="system", content=system_prompt_content, name=None))
        standardized_history.append(StandardizedMessage(role="user", content=initial_prompt_text, name=None))

        available_tools_json_str = "No tools available."
        if self.mcp_client_manager.available_tools:
            formatted_tools_for_prompt = []
            for tool_spec in self.mcp_client_manager.available_tools:
                params = {name: schema.get("type", "any") for name, schema in tool_spec.get("input_schema", {}).get("properties", {}).items()}
                formatted_tools_for_prompt.append({"name": tool_spec["name"], "description": tool_spec["description"], "parameters": params})
            if formatted_tools_for_prompt:
                 available_tools_json_str = json.dumps({"tools": formatted_tools_for_prompt}, indent=2)

        # Construct the initial HF prompt string
        current_hf_prompt_str = ""
        if system_prompt_content: current_hf_prompt_str += f"System: {system_prompt_content}\n\n"
        current_hf_prompt_str += f"You have access to the following tools:\n[AVAILABLE_TOOLS_START]\n{available_tools_json_str}\n[AVAILABLE_TOOLS_END]\n\n"
        current_hf_prompt_str += "When you need to use a tool, you MUST respond with a JSON object formatted EXACTLY as follows, and nothing else:\n"
        current_hf_prompt_str += "{\"tool_call\": {\"name\": \"<tool_name>\", \"arguments\": {\"<arg_name1>\": \"<value1>\"}}}\n"
        current_hf_prompt_str += "If you do not need to use a tool, respond directly to the user.\n\n"
        current_hf_prompt_str += f"User: {initial_prompt_text}\nAssistant:" # Start prompting for assistant's first response

        final_response_text = ""
        tool_calls_count = 0
        accumulated_input_tokens = 0
        accumulated_output_tokens = 0

        # Determine if the loaded model was an on-demand one for later cleanup
        was_loaded_on_demand = not (
            (actual_model_name_used == self.default_hf_completion_model and hf_model == self.hf_completion_model and self.hf_completion_model is not None) or
            (hasattr(self, 'hf_reasoning_model') and actual_model_name_used == self.default_hf_reasoning_model and hf_model == self.hf_reasoning_model and self.hf_reasoning_model is not None)
        )
        if not model_name: # If no specific model was given, it must have used a default (pre-loaded)
            was_loaded_on_demand = False


        for i in range(max_tool_calls + 1):
            logger.debug(f"HF Agentic - Turn {i + 1} - Current Prompt for HF Model (last 1000 chars):\n...{current_hf_prompt_str[-1000:]}")

            inputs = hf_tokenizer(current_hf_prompt_str, return_tensors="pt", padding=False, truncation=True, max_length=hf_tokenizer.model_max_length or 4096).to(device)
            current_turn_input_tokens = inputs.input_ids.shape[1] # Tokens for this specific turn's input
            accumulated_input_tokens += current_turn_input_tokens

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, pad_token_id=hf_tokenizer.pad_token_id,
                eos_token_id=hf_tokenizer.eos_token_id,
                # Consider adding stop sequences if model tends to hallucinate after JSON
            )

            with torch.no_grad():
                outputs = hf_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=generation_config)

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            raw_assistant_output = hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            accumulated_output_tokens += len(generated_ids) # Tokens for this specific turn's output

            logger.info(f"HF Model Raw Output (Turn {i+1}): '{raw_assistant_output}'")

            tool_call_detected_this_turn = False
            parsed_tool_call_data = None
            text_part_from_assistant = raw_assistant_output

            try:
                # More robust JSON extraction: find first '{' and last '}'
                json_start_index = raw_assistant_output.find("{")
                json_end_index = raw_assistant_output.rfind("}")
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    potential_json_str = raw_assistant_output[json_start_index : json_end_index+1]
                    logger.debug(f"Potential JSON for tool call: {potential_json_str}")
                    potential_json = json.loads(potential_json_str)
                    if "tool_call" in potential_json and isinstance(potential_json["tool_call"], dict):
                        tc_data = potential_json["tool_call"]
                        if "name" in tc_data and "arguments" in tc_data and isinstance(tc_data["arguments"], dict):
                            tool_call_detected_this_turn = True
                            parsed_tool_call_data = tc_data
                             # If there was text before the JSON, capture it. Otherwise, empty.
                            text_part_from_assistant = raw_assistant_output[:json_start_index].strip()
                            logger.info(f"Detected tool call: {parsed_tool_call_data['name']} with args {parsed_tool_call_data['arguments']}")
                            if text_part_from_assistant:
                                logger.info(f"Text preceding tool call: {text_part_from_assistant}")
                        else:
                            parsed_tool_call_data = None
                else: # No JSON structure found
                     logger.debug("No JSON-like structure found for tool call.")
            except json.JSONDecodeError:
                logger.debug("Output is not valid JSON or not a tool_call structure. Treating as text.")
            except Exception as e_parse: # Catch any other parsing error
                logger.error(f"Error parsing assistant output for tool call: {e_parse}", exc_info=True)


            assistant_content_parts_for_std: List[StandardizedMessageContentPart] = []
            if text_part_from_assistant:
                 assistant_content_parts_for_std.append(StandardizedMessageContentPart(type="text", text=text_part_from_assistant))

            if tool_call_detected_this_turn and parsed_tool_call_data:
                assistant_content_parts_for_std.append(StandardizedMessageContentPart(
                    type="tool_use", name=parsed_tool_call_data["name"], input=parsed_tool_call_data["arguments"]
                    # 'id' for tool_use is optional in our schema, and HF emulation doesn't naturally provide one here.
                ))

            if assistant_content_parts_for_std:
                standardized_history.append(StandardizedMessage(role="assistant", content=assistant_content_parts_for_std, name=actual_model_name_used))

            current_hf_prompt_str += raw_assistant_output # Append raw output for next turn's prompt

            if not tool_call_detected_this_turn:
                final_response_text = text_part_from_assistant # This is the final answer
                logger.info(f"HF Agentic: No tool call. Final response: {final_response_text[:200]}...")
                break

            # --- Handle Tool Call ---
            tool_name = parsed_tool_call_data["name"]
            tool_args = parsed_tool_call_data["arguments"]
            tool_calls_count += 1

            tool_session = asyncio.run(self.mcp_client_manager.get_tool_session(tool_name))
            tool_result_str_for_hf_prompt: str
            tool_result_for_std_history: Union[str, Dict, List]
            is_error_result = False

            if tool_session:
                try:
                    mcp_tool_result = asyncio.run(tool_session.call_tool(tool_name, arguments=tool_args))
                    tool_result_for_std_history = mcp_tool_result.content
                    tool_result_str_for_hf_prompt = json.dumps(mcp_tool_result.content) if isinstance(mcp_tool_result.content, (dict, list)) else str(mcp_tool_result.content)
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    tool_result_str_for_hf_prompt = json.dumps({"error": error_msg}) # Ensure result is JSON string for prompt
                    tool_result_for_std_history = {"error": error_msg}
                    is_error_result = True
            else:
                error_msg = f"Tool {tool_name} is not available or configured."
                logger.warning(error_msg)
                tool_result_str_for_hf_prompt = json.dumps({"error": error_msg})
                tool_result_for_std_history = {"error": error_msg}
                is_error_result = True

            # Append structured tool result to HF prompt history
            current_hf_prompt_str += f"\n[TOOL_RESULT_START]\n{{\"tool_name\": \"{tool_name}\", \"result\": {tool_result_str_for_hf_prompt}}}\n[TOOL_RESULT_END]\nAssistant:"

            std_tool_msg_part = StandardizedMessageContentPart(type="tool_result", content=tool_result_for_std_history)
            # tool_use_id is not available from HF emulated call request, so cannot be set here.
            if is_error_result: std_tool_msg_part["is_error"] = True
            standardized_history.append(StandardizedMessage(role="tool", name=tool_name, content=[std_tool_msg_part]))

            if tool_calls_count >= max_tool_calls:
                logger.warning("Reached maximum tool calls for HF agentic interaction.")
                final_response_text = f"Reached maximum tool calls. Last tool '{tool_name}' was called with result: {tool_result_str_for_hf_prompt[:200]}..."
                break

        if was_loaded_on_demand: # Only cleanup if loaded on-demand
             logger.info(f"Cleaning up on-demand loaded HF model '{actual_model_name_used}'.")
             del hf_model
             del hf_tokenizer
             self._cleanup_memory(force=True)

        if return_full_response:
            return StandardizedAgenticResponse(
                prompt_id=prompt_id, prompt=initial_prompt_text, response=final_response_text,
                full_conversation_history=standardized_history,
                tokens_in=accumulated_input_tokens, tokens_out=accumulated_output_tokens,
                model=actual_model_name_used, temperature=temperature, top_p=top_p, top_k=top_k,
                tool_calls_made=tool_calls_count
            )
        return final_response_text

    def _get_azure_agentic(self, model: str, initial_messages: List[Dict[str, str]], **kwargs) -> Union[StandardizedAgenticResponse, str]:
        """
        Handles agentic interaction with an Azure OpenAI model, including MCP tool calls
        using Azure's function/tool calling capabilities.
        Returns a standardized response structure if `return_full_response` is True in kwargs.

        Args:
            model (str): The Azure OpenAI deployment ID.
            initial_messages (List[Dict[str, str]]): The initial messages for the conversation.
            **kwargs: See `get_agentic` for other relevant kwargs like `return_full_response`,
                      `prompt_id`, `system_prompt`, `temperature`, `max_tokens`, etc.

        Returns:
            Union[StandardizedAgenticResponse, str]: Standardized response or final text string.
        """
        logger.info(f"Executing Azure OpenAI agentic call for model {model} (deployment ID)")
        self.mcp_client_manager.ensure_initialized()

        if not self.azure_endpoint or not (self.client_secret or self.subscription_key or os.getenv("AZURE_OPENAI_API_KEY")):
            raise ValueError("Azure OpenAI credentials (Entra ID or API Key) or endpoint not fully configured for agentic calls.")

        temperature = kwargs.get('temperature', 0.7)
        max_tokens_completion = kwargs.get('max_tokens', 2048)
        top_p = kwargs.get('top_p', None)
        system_prompt_content = kwargs.get('system_prompt', None)
        prompt_id = kwargs.get('prompt_id', None)
        initial_prompt_text = initial_messages[0]['content'] if initial_messages and initial_messages[0]['role'] == 'user' else ""
        return_full_response = kwargs.get('return_full_response', False)

        if self.tenant_id and self.client_id and self.client_secret :
            access_token = self.refresh_token()
            client = AzureOpenAI(api_version=self.api_version, azure_endpoint=self.azure_endpoint, azure_ad_token=access_token)
            logger.info("Using Entra ID (token) for Azure OpenAI authentication.")
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            client = AzureOpenAI(api_version=self.api_version, azure_endpoint=self.azure_endpoint, api_key=os.getenv("AZURE_OPENAI_API_KEY"))
            logger.info("Using API Key for Azure OpenAI authentication.")
        else:
             client = AzureOpenAI(api_version=self.api_version, azure_endpoint=self.azure_endpoint, api_key=self.subscription_key)
             logger.warning("Using subscription_key as API key for Azure OpenAI. Ensure this is intended.")

        final_response_text = ""
        accumulated_prompt_tokens = 0
        accumulated_completion_tokens = 0
        MAX_TOOL_CALLS = kwargs.get('max_tool_calls', 5)
        tool_calls_count = 0

        azure_api_history = []
        if system_prompt_content:
            azure_api_history.append({"role": "system", "content": system_prompt_content})
        azure_api_history.extend(initial_messages)

        standardized_history: List[StandardizedMessage] = []
        if system_prompt_content:
            standardized_history.append(StandardizedMessage(role="system", content=system_prompt_content, name=None))
        for msg in initial_messages:
            standardized_history.append(StandardizedMessage(role=msg["role"], content=msg["content"], name=None))

        azure_tools_formatted = []
        if self.mcp_client_manager.available_tools:
            for mcp_tool in self.mcp_client_manager.available_tools:
                try:
                    azure_tools_formatted.append({
                        "type": "function",
                        "function": {
                            "name": mcp_tool["name"],
                            "description": mcp_tool["description"],
                            "parameters": mcp_tool["input_schema"]
                        }
                    })
                except KeyError as e:
                    logger.warning(f"Skipping MCP tool {mcp_tool.get('name', 'Unknown')} for Azure due to schema issue: {e}")
            logger.info(f"Azure Agentic: Formatted {len(azure_tools_formatted)} tools for Azure API.")
        else:
            logger.info("Azure Agentic: No MCP tools available/loaded.")

        while tool_calls_count < MAX_TOOL_CALLS:
            api_params = {
                "model": model, # Azure deployment ID
                "messages": azure_api_history,
                "temperature": temperature,
                "max_tokens": max_tokens_completion,
            }
            if top_p is not None: api_params["top_p"] = top_p
            if azure_tools_formatted: api_params["tools"] = azure_tools_formatted
            # api_params["tool_choice"] = "auto" # Default

            logger.debug(f"Azure API call ({tool_calls_count + 1}) with messages: {azure_api_history}")
            chat_response = client.chat.completions.create(**api_params)

            if chat_response.usage: # Accumulate tokens
                 accumulated_prompt_tokens += chat_response.usage.prompt_tokens
                 accumulated_completion_tokens += chat_response.usage.completion_tokens

            response_message = chat_response.choices[0].message
            # Add assistant's response (which might include tool_calls) to history
            azure_api_history.append(response_message.model_dump(exclude_none=True))


            if response_message.tool_calls:
                logger.info(f"Azure OpenAI requested tool_calls: {response_message.tool_calls}")
                for tool_call in response_message.tool_calls:
                    if tool_call.type == "function":
                        function_name = tool_call.function.name
                        function_args_str = tool_call.function.arguments # This is a string
                        tool_call_id = tool_call.id

                        logger.info(f"Azure Agentic: Function call to '{function_name}' (ID: {tool_call_id}) with args string: {function_args_str}")

                        try:
                            # Arguments from Azure are a JSON string
                            function_args_dict = json.loads(function_args_str)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON arguments for {function_name}: {function_args_str}")
                            tool_result_content = f"Error: Could not parse arguments for {function_name}."
                        else:
                            tool_session = asyncio.run(self.mcp_client_manager.get_tool_session(function_name))
                            if tool_session:
                                try:
                                    # Use shlex.quote to sanitize function_name and json.dumps for function_args_dict
                                    sanitized_function_name = shlex.quote(function_name)  # import shlex
                                    sanitized_args = json.dumps(function_args_dict)
                                    mcp_tool_result = asyncio.run(tool_session.call_tool(sanitized_function_name, arguments=json.loads(sanitized_args)))
                                    # Azure expects tool output as a string
                                    tool_result_content = json.dumps(mcp_tool_result.content) if isinstance(mcp_tool_result.content, (dict, list)) else str(mcp_tool_result.content)
                                    logger.info(f"MCP Tool '{function_name}' executed. Result preview: {tool_result_content[:100]}...")
                                except Exception as e:
                                    logger.error(f"Error calling MCP tool {function_name} via Azure: {e}", exc_info=True)
                                    tool_result_content = f"Error executing tool {function_name}: {str(e)}"
                            else:
                                tool_result_content = f"Tool {function_name} is not available or configured."
                                logger.warning(f"Tool {function_name} requested by Azure not found in MCPClientManager.")

                        # Append the tool result message to history for the next API call
                        azure_api_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": tool_result_content
                        })


                tool_calls_count += 1 # Increment after processing all tool calls in a turn
                if tool_calls_count >= MAX_TOOL_CALLS:
                    logger.warning("Reached maximum tool calls for Azure agentic interaction.")
                    final_response_text = "Reached maximum tool calls; processing stopped."
                    break
                # Continue to the next iteration of the while loop to send tool results to Azure
                continue
            else: # No tool_calls in the response_message
                final_response_text = response_message.content if response_message.content else ""
                logger.info(f"Azure agentic call finished. Final text: {final_response_text[:200]}...")
                break # Exit while loop

        if return_full_response:
            return {"prompt_id": prompt_id, "prompt": initial_prompt_text, "response": final_response_text,
                    "full_conversation_history": azure_api_history,
                    "tokens_in": accumulated_prompt_tokens,
                    "tokens_out": accumulated_completion_tokens,
                    "model": model, "temperature": temperature, "top_p": top_p,
                    "tool_calls_made": tool_calls_count}
        return final_response_text

    def _get_gemini_agentic(self, model: str, initial_messages: List[Dict[str, str]], **kwargs) -> Union[StandardizedAgenticResponse, str]:
        """
        Handles agentic interaction with a Google Gemini model, including MCP tool calls.
        Returns a standardized response structure if `return_full_response` is True in kwargs.

        Args:
            model (str): The Gemini model name (e.g., "gemini-1.5-flash").
            initial_messages (List[Dict[str, str]]): The initial messages for the conversation.
            **kwargs: See `get_agentic` for other relevant kwargs like `return_full_response`,
                      `prompt_id`, `system_prompt`, `temperature`, `max_tokens`, etc.

        Returns:
            Union[StandardizedAgenticResponse, str]: Standardized response or final text string.
        """
        logger.info(f"Executing Gemini agentic call for model {model}")
        self.mcp_client_manager.ensure_initialized() # Ensure MCP manager is ready

        if not self.gemini_api_key:
            raise ValueError("Google Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")

        # Configure Gemini client (done globally in __init__ if api_key is present)
        # genai.configure(api_key=self.gemini_api_key) # Already done if key exists
        gemini_model = genai.GenerativeModel(model_name=model)

        temperature = kwargs.get('temperature', 0.7)
        max_output_tokens = kwargs.get('max_tokens', 2048) # Gemini uses max_output_tokens
        top_p = kwargs.get('top_p', None)
        top_k = kwargs.get('top_k', None)
        system_prompt_content = kwargs.get('system_prompt', None) # Gemini uses system_instruction in GenerateContentConfig
        prompt_id = kwargs.get('prompt_id', None)
        initial_prompt_text = initial_messages[0]['content'] if initial_messages and initial_messages[0]['role'] == 'user' else ""
        return_full_response = kwargs.get('return_full_response', False)

        final_response_text = ""
        # For Gemini, token counting is usually done via model.count_tokens() before/after.
        # The response object might contain some usage metadata.
        accumulated_prompt_tokens = 0
        accumulated_candidates_tokens = 0 # Gemini uses this term

        MAX_TOOL_CALLS = kwargs.get('max_tool_calls', 5)
        tool_calls_count = 0

        # History for Gemini API (native Content objects)
        gemini_api_history: List[genai_types.Content] = []

        # History for standardized output
        standardized_history: List[StandardizedMessage] = []

        # Handle initial system prompt for standardized history
        if system_prompt_content:
            standardized_history.append(StandardizedMessage(role="system", content=system_prompt_content, name=None))

        # Convert initial messages to both Gemini's Content format and StandardizedMessage format
        for msg in initial_messages:
            role = msg["role"]
            content = msg["content"]
            # Add to Gemini API history
            if role == "user":
                gemini_api_history.append(genai_types.Content(role="user", parts=[genai_types.Part(text=content)]))
            elif role == "assistant": # Standardized role, map to "model" for Gemini API
                gemini_api_history.append(genai_types.Content(role="model", parts=[genai_types.Part(text=content)]))

            # Add to standardized history
            standardized_history.append(StandardizedMessage(role=role, content=content, name=None)) # Assuming simple text for initial messages

        gemini_tools_formatted = []
        if self.mcp_client_manager.available_tools:
            for mcp_tool in self.mcp_client_manager.available_tools:
                try:
                    # MCP input_schema should be directly usable as Gemini's function parameters schema
                    func_decl = genai_types.FunctionDeclaration(
                        name=mcp_tool["name"],
                        description=mcp_tool["description"],
                        parameters=mcp_tool["input_schema"] # Assuming direct compatibility
                    )
                    gemini_tools_formatted.append(genai_types.Tool(function_declarations=[func_decl]))
                except Exception as e:
                    logger.warning(f"Skipping MCP tool {mcp_tool.get('name', 'Unknown')} for Gemini due to schema conversion issue: {e}")
            logger.info(f"Gemini Agentic: Formatted {len(gemini_tools_formatted)} tools for Gemini API.")
        else:
            logger.info("Gemini Agentic: No MCP tools available/loaded.")

        generation_config_dict = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if top_p is not None: generation_config_dict["top_p"] = top_p
        if top_k is not None: generation_config_dict["top_k"] = top_k

        safety_settings = [ # Example safety settings, adjust as needed
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        while tool_calls_count < MAX_TOOL_CALLS:
            logger.debug(f"Gemini API call ({tool_calls_count + 1}) with contents: {[c.to_dict() for c in gemini_api_history]}")

            request_args = {
                "contents": gemini_api_history,
                "generation_config": genai_types.GenerationConfig(**generation_config_dict),
                "safety_settings": safety_settings
            }
            # Add system_instruction only on the first turn if provided
            if system_prompt_content and tool_calls_count == 0:
                request_args["system_instruction"] = genai_types.Content(parts=[genai_types.Part(text=system_prompt_content)])

            if gemini_tools_formatted:
                request_args["tools"] = gemini_tools_formatted

            try:
                response = gemini_model.generate_content(**request_args)
                if response.usage_metadata:
                    accumulated_prompt_tokens += response.usage_metadata.prompt_token_count
                    accumulated_candidates_tokens += response.usage_metadata.candidates_token_count
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}", exc_info=True)
                final_response_text = f"Error calling Gemini API: {e}"
                break # Exit loop on API error

            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                logger.error("Gemini response had no candidates.")
                final_response_text = "Error: No response from Gemini model."
                break

            gemini_api_history.append(candidate.content) # Add model's response to API history

            # Standardize assistant's response for standardized_history
            assistant_content_parts_for_std: List[StandardizedMessageContentPart] = []
            has_tool_call_this_turn = False

            for part in candidate.content.parts:
                if part.text:
                    assistant_content_parts_for_std.append(StandardizedMessageContentPart(type="text", text=part.text))
                if part.function_call:
                    has_tool_call_this_turn = True
                    function_call = part.function_call
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)

                    # Gemini doesn't provide a tool_call_id in the request part,
                    # we'll generate one for internal tracking if needed, or rely on name for now.
                    # For StandardizedMessageContentPart, 'id' is optional for 'tool_use'.
                    assistant_content_parts_for_std.append(StandardizedMessageContentPart(
                        type="tool_use", name=tool_name, input=tool_args # id is optional
                    ))
                    logger.info(f"Gemini Agentic: Requested tool '{tool_name}' with args: {tool_args}")

                    tool_session = asyncio.run(self.mcp_client_manager.get_tool_session(tool_name))
                    tool_result_content_for_gemini_api: Any
                    tool_result_content_for_standardized: Union[str, Dict, List]
                    is_error_result = False

                    if tool_session:
                        try:
                            mcp_tool_result = asyncio.run(tool_session.call_tool(tool_name, arguments=tool_args))
                            tool_result_content_for_gemini_api = mcp_tool_result.content
                            tool_result_content_for_standardized = mcp_tool_result.content
                            logger.info(f"MCP Tool '{tool_name}' executed. Result type: {type(tool_result_content_for_gemini_api)}")
                        except Exception as e:
                            logger.error(f"Error calling MCP tool {tool_name} via Gemini: {e}", exc_info=True)
                            error_payload = {"error": f"Error executing tool {tool_name}: {str(e)}"}
                            tool_result_content_for_gemini_api = error_payload
                            tool_result_content_for_standardized = error_payload
                            is_error_result = True
                    else:
                        logger.warning(f"Tool {tool_name} requested by Gemini not found in MCPClientManager.")
                        error_payload = {"error": f"Tool {tool_name} is not available or configured."}
                        tool_result_content_for_gemini_api = error_payload
                        tool_result_content_for_standardized = error_payload
                        is_error_result = True

                    # Add tool result to Gemini API history
                    gemini_api_history.append(genai_types.Content(
                        parts=[genai_types.Part.from_function_response(
                            name=tool_name,
                            response=tool_result_content_for_gemini_api
                        )]
                        # Gemini's 'user' role for function response is implicit by structure
                    ))
                    # Add tool result to Standardized history
                    std_tool_result_part = StandardizedMessageContentPart(
                        type="tool_result",
                        # tool_use_id will be None as Gemini FC doesn't have IDs in request
                        content=tool_result_content_for_standardized
                    )
                    if is_error_result: std_tool_result_part["is_error"] = True
                    standardized_history.append(StandardizedMessage(role="tool", name=tool_name, content=[std_tool_result_part]))

            # Add assistant's full turn (text and/or tool_use parts) to standardized_history
            if assistant_content_parts_for_std:
                 standardized_history.append(StandardizedMessage(role="assistant", content=assistant_content_parts_for_std, name=None))


            if not has_tool_call_this_turn:
                final_response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
                logger.info(f"Gemini agentic call finished. Final text: {final_response_text[:200]}...")
                break # Exit while loop

            tool_calls_count += 1
            if tool_calls_count >= MAX_TOOL_CALLS:
                logger.warning("Reached maximum tool calls for Gemini agentic interaction.")
                final_response_text = "Reached maximum tool calls; processing stopped."
                # Extract any text from the last assistant turn before maxing out
                current_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
                if current_text: final_response_text = current_text + " (Reached maximum tool calls)"
                break

        if return_full_response:
            return StandardizedAgenticResponse(
                prompt_id=prompt_id,
                prompt=initial_prompt_text,
                response=final_response_text,
                full_conversation_history=standardized_history,
                tokens_in=accumulated_prompt_tokens,
                tokens_out=accumulated_candidates_tokens, # Use candidates_token_count for output
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                tool_calls_made=tool_calls_count
            )
        return final_response_text

    # TBC on the needs for a distinct function for calling reasoning models
    def get_reasoning_simulator(self,
                            prompt_id: int,
                            prompt: str,
                            store_result: Optional[bool] = False,
                            stored_df: Optional[pd.DataFrame] = None,
                            system_prompt: Optional[str] = None,
                            model: Optional[str] = None,  
                            return_full_response: Optional[bool] = False) -> Union[Dict[str, Any], str]:
        """Get reasoning simulation output using a specified model.
            
            Parameters:
                prompt_id (int): Identifier for the prompt.
                prompt (str): The input prompt.
                store_result (bool): Whether to store results in stored_df DataFrame.
                stored_df (Optional[pd.DataFrame]): DataFrame to store results in.
                system_prompt (Optional[str]): Optional system prompt.
                model (Optional[str]): Specific model to use (e.g., "o1", "o3_mini", or a HuggingFace reasoning model identifier).
                return_full_response (Optional[bool]): If True, returns the full response (e.g., DataFrame); otherwise, returns just the response text.
            
            Returns:
                Union[Dict[str, Any], str]: A standardized output dictionary containing:
                    - prompt_id: Identifier for the prompt
                    - prompt: Original prompt
                    - response: Generated reasoning output
                    - perplexity: Perplexity score (if available)
                    - tokens_in: Number of input tokens
                    - tokens_out: Number of output tokens
                    - model: Model used
                    - seed: Random seed used (if applicable)
                    - top_p: Top-p value used (if applicable)
                    - temperature: Temperature used (if applicable)
            """
        # Initialize a dataframe to store the results if store_result is enabled
        if store_result and stored_df is None:
            stored_df = pd.DataFrame()
        
        # Use a default reasoning model if none is specified
        if model is None:
            logger.warning("Model not specified, falling back to the default reasoning model")
            model = self.default_reasoning_model  # assumes this is defined in your class
        
        try:
            # Try using Azure OpenAI reasoning simulator if applicable
            if self._validate_model(model, "reasoning", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI reasoning model '{model}'")
                    df_result, final_response = self._get_azure_reasoning_simulator(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model
                        # Additional reasoning-specific parameters can be added here if needed
                    )
                except Exception as e:
                    logger.warning(f"Azure OpenAI reasoning simulator failed: {e}")
                    # Optionally, one could set a fallback flag here.
                    raise
            # Try using HuggingFace reasoning simulator if validated as such
            elif self._validate_model(model, "reasoning", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace reasoning simulator model '{model}'")
                    df_result, final_response = self._get_hf_reasoning_simulator(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model
                    )
                except Exception as e:
                    logger.warning(f"HuggingFace reasoning simulator failed: {e}")
                    raise
            else:
                # Fall back to default HuggingFace reasoning simulator if model is not explicitly validated
                logger.info(f"Using default reasoning model '{model}'")
                df_result, final_response = self._get_hf_reasoning_simulator(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=self.default_reasoning_model
                )
            
            # Store the result using a utility method to ensure consistent formatting (only if store_result is enabled)
            if store_result and stored_df is not None:
                stored_df = pd.concat([stored_df, self.datautility.format_conversion(df_result, "dataframe").T], ignore_index=True)
            
            if return_full_response:
                return df_result
            else:
                return final_response
        
        except Exception as e:
            logger.error(f"Reasoning simulator failed: {str(e)}")
            raise
            
    def _get_azure_reasoning_simulator(self,
                                    prompt_id: int,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    model: Optional[str] = "gpt-4o_reasoning",  
                                    temperature: Optional[float] = 1,
                                    max_tokens: Optional[int] = 3000,
                                    top_p: Optional[float] = 1,
                                    top_k: Optional[float] = 10,
                                    frequency_penalty: Optional[float] = 1.1,
                                    presence_penalty: Optional[float] = 1,
                                    seed: Optional[int] = 100,
                                    logprobs: Optional[bool] = False,
                                    json_schema: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Internal method for Azure OpenAI reasoning simulation.
        
        Parameters:
            prompt_id (int): Identifier for the prompt.
            prompt (str): The input prompt.
            system_prompt (Optional[str]): Optional system prompt.
            model (Optional[str]): Specific Azure reasoning model.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens in response.
            top_p (float): Nucleus sampling parameter.
            top_k (float): Top-K sampling parameter.
            frequency_penalty (float): Frequency penalty parameter.
            presence_penalty (float): Presence penalty parameter.
            seed (Optional[int]): Random seed for reproducibility.
            logprobs (bool): Whether to return log probabilities.
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation.
        
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A dictionary with standardized result keys.
                - The generated reasoning output as text.
        """
        # Process prompt into messages; include system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Set random seed if not provided
        seed = self.statsutility.set_random_seed(min_value=0, max_value=100) if seed is None else seed
        
        # Determine response format based on prompt content
        if json_schema:
            response_format = {"type": "json_schema", "json_schema": json_schema}
        elif re.search(r'\bJSON\b', prompt, re.IGNORECASE):
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
        
        # Allow a number of attempts when calling the API    
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token if necessary
                access_token = self.refresh_token()
                client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token=access_token
                )
                
                # Make API call to the reasoning endpoint (assumed similar to chat completions)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    response_format=response_format,
                    extra_header={
                        'x-correlation-id': str(uuid.uuid4()),
                        'x-subscription-key': self.subscription_key
                    }
                )
                
                # Extract response text; for reasoning tasks, we expect chain-of-thought output
                response_text = response.choices[0].message.content
                
                # Process JSON response if needed
                if response_format["type"] == "json_object":
                    try:
                        response_text = json.loads(response_text.strip('```json').strip('```'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise ValueError("Invalid JSON response from model")
                
                # Calculate perplexity if log probabilities are available
                # log_probs = response.choices[0].logprobs if logprobs else None
                # perplexity = self._calculate_perplexity(log_probs) if log_probs else None
                
                # Prepare the standardized result dictionary
                results = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response_text,
                    "perplexity": None,
                    "tokens_in": response.usage.prompt_tokens,
                    "tokens_out": response.usage.completion_tokens,
                    "model": model,
                    "seed": None,
                    "top_p": None,
                    "top_k": None,
                    "temperature": None
                }
                logger.debug("Successfully got reasoning simulation from Azure OpenAI")
                return (results, response_text)
                
            except Exception as e:
                logger.warning("Azure reasoning attempt %d failed: %s", attempt + 1, e)
                if attempt < self.default_max_attempts - 1:
                    self.refresh_token()
                else:
                    raise

    def _get_hf_reasoning_simulator(self,
                                    prompt_id: int,
                                    prompt: str,
                                    system_prompt: Optional[str] = None,
                                    model: Optional[str] = None,
                                    temperature: Optional[float] = 1,
                                    max_tokens: Optional[int] = 2000,
                                    top_p: Optional[float] = 1,
                                    top_k: Optional[float] = 10,
                                    frequency_penalty: Optional[float] = 1.3,
                                    num_beam: Optional[int] = None,
                                    logprobs: Optional[bool] = False) -> Tuple[Dict[str, Any], str]:
        """
        Get reasoning simulation output from a HuggingFace model.
        
        Parameters:
            prompt_id (int): Identifier for the prompt.
            prompt (str): The input prompt.
            system_prompt (Optional[str]): Optional system prompt.
            model (Optional[str]): HuggingFace model identifier or local model name.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.
            top_p (float): Nucleus sampling parameter.
            top_k (float): Top-K sampling parameter.
            frequency_penalty (float): Penalty to discourage repetition.
            num_beam (Optional[int]): Number of beams for beam search (if any).
            logprobs (bool): Whether to return log probabilities.
        
        Returns:
            Tuple[Dict[str, Any], str]: A tuple containing:
                - A standardized dictionary with the reasoning simulation results.
                - The generated reasoning output as text.
        """
        # Construct the message template; include a "Reasoning:" marker to prompt chain-of-thought output
        if system_prompt:
            messages = f"System: {system_prompt}\nUser: {prompt}\nReasoning:"
        else:
            messages = f"User: {prompt}\nReasoning:"
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        orig_model = model
        
        try:
            # Use pre-loaded models if available (for default models)
            if model is None and self.hf_completion_model is not None and self.hf_completion_tokenizer is not None:
                logger.debug("Using pre-loaded HF completion model and tokenizer for reasoning")
                hf_model = self.hf_completion_model.to(device)
                hf_tokenizer = self.hf_completion_tokenizer
            else:
                # Fallback to loading on-demand for non-default models
                if model:
                    model_path = self.hf_model_dir / model
                else:
                    model_path = self.hf_model_dir / self.default_hf_reasoning_model
                
                model_path_str = str(model_path)
                logger.warning("Pre-loaded HF completion model not available for reasoning, loading on-demand (may cause VMS issues)")
                logger.info(f"Attempting to load local HuggingFace reasoning model from path: {model_path_str}")

                if not model_path.is_dir():
                    error_msg = f"Local HuggingFace reasoning model path is not a directory or does not exist: {model_path_str}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                hf_tokenizer = AutoTokenizer.from_pretrained(model_path_str, local_files_only=True)
                hf_model = AutoModelForCausalLM.from_pretrained(model_path_str, local_files_only=True)
                hf_model = hf_model.to(device)
                logger.info(f"Successfully loaded local HuggingFace reasoning model and tokenizer from: {model_path_str}")
                
                if hf_tokenizer.pad_token is None:
                    hf_tokenizer.pad_token = hf_tokenizer.eos_token

            # Customised the stop token id
            user_token_ids = hf_tokenizer("User:", add_special_tokens=False).input_ids
            if len(user_token_ids) == 1:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids
            else:
                combined_eos_ids = [hf_tokenizer.eos_token_id] + user_token_ids[1:]
            model_inputs = hf_tokenizer(messages, return_tensors="pt", padding=True).to(hf_model.device)
            
            # Set up the generation configuration
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=True if num_beam is None else False,
                num_beams=num_beam if num_beam is not None else 1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=frequency_penalty,
                pad_token_id=hf_tokenizer.pad_token_id,
                bos_token_id=hf_tokenizer.bos_token_id,
                eos_token_id=combined_eos_ids
            )
            
            # Generate output from the model
            with torch.no_grad():
                generated_ids = hf_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    generation_config=generation_config
                )
                # Create labels to mask out prompt tokens for loss calculation if needed
                labels = generated_ids.clone()
                for i, input_id in enumerate(model_inputs.input_ids):
                    prompt_length = input_id.size(0)
                    labels[i, :prompt_length] = -100
                outputs = hf_model(generated_ids, labels=labels)
                loss_value = outputs.loss.item()
                perplexity_score = self._calculate_perplexity(loss_value)
            
            # Remove prompt tokens from the generated sequence
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up stop tokens from the response
            stop_phrases = ["\nUser:", "User:", "\nHuman:", "Human:"]
            for stop_phrase in stop_phrases:
                if response.endswith(stop_phrase):
                    response = response[:-len(stop_phrase)].rstrip()
                    break
            # Also handle cases where stop phrase appears anywhere at the end
            response = re.sub(r'\s*(User|Human):\s*$', '', response).rstrip()
            
            # Calculate token counts
            tokens_in = len(model_inputs.input_ids[0])
            tokens_out = len(generated_ids[0])
            
            # Prepare standardized results
            results = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response,
                "perplexity": perplexity_score if logprobs else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": orig_model,
                "seed": None,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            }
            logger.debug("Successfully got reasoning simulation from HuggingFace model")
            return (results, response)
        
        except Exception as e:
            logger.error(f"Failed to get reasoning simulation from HuggingFace model: {e}")
            raise

    def get_ocr(self, 
                pdf_file_path: str, 
                model: Optional[str] = None,
                output_format: str = "markdown") -> None:
        """Get OCR results using specified model or fall back to HuggingFace model.

        Parameters:
            file_path (str): Path to file(s) to process (PDF)
            model (Optional[str]): Specific model to use
                            If None or invalid, falls back to initialized HuggingFace model
            output_format (str): Format to save output ("markdown", "json", "txt", etc.)

        Returns:
            None
        """
        if pdf_file_path:
            # Extract the PDF file name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
            
            # Create the output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(pdf_file_path), "db", "preprocess")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output file path with .md extension in the preprocess folder
            output_file = os.path.join(output_dir, f"{pdf_name}.{output_format}")

        try:
            # Verify HuggingFace fallback model is available
            if not self._validate_model(self.default_hf_ocr_model, "ocr", "huggingface"):
                raise ValueError(f"Fallback HuggingFace model '{self.default_hf_ocr_model}' is not available")

            # If specific model provided, try to use it
            if model:
                # Try Mistral first if it's a Mistral model
                if self._validate_model(model, "ocr", "mistral"):
                    try:
                        logger.info(f"Using Mistral model '{model}' for OCR")
                        self._get_mistral_ocr(model=model, 
                                            pdf_file_path=pdf_file_path, 
                                            output_file=output_file,
                                            output_format=output_format)
                        logger.info("Successfully got OCR from Mistral model")
                        return None 
                    except Exception as e:
                        logger.error(f"Mistral OCR failed due to error {e}, falling back to HuggingFace default model")
                
                # Try HuggingFace if it's an HF model
                elif self._validate_model(model, "ocr", "huggingface"):
                    try:
                        logger.info(f"Using HuggingFace model '{model}' for OCR")
                        self._get_hf_ocr(model=model,
                                        pdf_file_path=pdf_file_path,
                                        output_file=output_file,
                                        output_format=output_format)
                        logger.info("Successfully got OCR from HuggingFace model")
                        return None # Explicitly return None
                    except Exception as e:
                        logger.error(f"HuggingFace OCR for model '{model}' failed: {e}", exc_info=True)
                        # Fallback to default HF model if specific HF model fails
                        logger.warning(f"Falling back to default HuggingFace OCR model due to error with '{model}'.")
                        # This creates a nested try-except, which is fine.
                        # The outer try-except will catch failures of the default model.
                        pass # Let it fall through to the default HF model logic outside this if/elif block if 'model' was not None

                # If 'model' was specified but wasn't 'mistral' and wasn't a validated 'huggingface' ocr model,
                # or if it was a validated HF model but failed and we passed the except block above.
                # This 'else' will now primarily be hit if the model is not recognized by the above conditions.
                # Or if a specific HF model failed and the 'pass' was hit.
                # The logic below handles the case where 'model' is None (use default) or specified but failed/unrecognized.

            # If model was specified (and failed above or was unhandled) OR model was not specified at all, try default HF OCR.
            current_model_for_fallback = self.default_hf_ocr_model
            if model and model != current_model_for_fallback:
                # This case covers:
                # 1. Specified HF model failed, and 'pass' was hit.
                # 2. Specified model was not 'mistral' and not a validated 'huggingface' model initially.
current_model_for_fallback = self.default_hf_ocr_model
            if model and model != current_model_for_fallback:
                # This case covers:
                # 1. Specified HF model failed, and 'pass' was hit.
                # 2. Specified model was not 'mistral' and not a validated 'huggingface' model initially.
                logger.info("Attempting HuggingFace default OCR model '{}' as fallback (previous attempt with '{}' failed or model type was unhandled/invalid).".format(current_model_for_fallback, model))  # import html
            elif not model: # model was None from the start
                 logger.info("Model not specified, using HuggingFace default model '{}' for OCR.".format(current_model_for_fallback))  # import html
            # If 'model' was specified and IS current_model_for_fallback, and it failed the first try, this path is still hit.
            # The log above would already indicate it's a fallback for itself, which is acceptable.

            try:
                self._get_hf_ocr(model=current_model_for_fallback,
                                pdf_file_path=pdf_file_path,
                                output_file=output_file,
                                output_format=output_format)
                logger.info("Successfully got OCR from HuggingFace default model ('{}'}.".format(current_model_for_fallback))  # import html
                return None
            except Exception as e_default:
                # If the default model itself fails here, it means all paths (specific model, then default model) have failed.
                logger.error("HuggingFace default OCR model ('{}') also failed: {}".format(current_model_for_fallback, e_default), exc_info=True)  # import html
                raise e_default # Re-raise the specific error from the default model's attempt.

        except Exception as e:
            # This top-level except handles:
            # - Initial validation error for default_hf_ocr_model.
            # - Failure of Mistral model if it was tried and raised.
            # - Failure of the default HF model if it was tried and raised (and not caught by a more specific except).
            logger.error("OCR processing ultimately failed: {}".format(str(e)), exc_info=True)  # import html
            raise
            elif not model: # model was None from the start
                 logger.info(f"Model not specified, using HuggingFace default model '{current_model_for_fallback}' for OCR.")
            # If 'model' was specified and IS current_model_for_fallback, and it failed the first try, this path is still hit.
            # The log above would already indicate it's a fallback for itself, which is acceptable.

            try:
                self._get_hf_ocr(model=current_model_for_fallback,
                                pdf_file_path=pdf_file_path,
                                output_file=output_file,
                                output_format=output_format)
                logger.info(f"Successfully got OCR from HuggingFace default model ('{current_model_for_fallback}').")
                return None
            except Exception as e_default:
                # If the default model itself fails here, it means all paths (specific model, then default model) have failed.
                logger.error(f"HuggingFace default OCR model ('{current_model_for_fallback}') also failed: {e_default}", exc_info=True)
                raise e_default # Re-raise the specific error from the default model's attempt.

        except Exception as e:
            # This top-level except handles:
            # - Initial validation error for default_hf_ocr_model.
            # - Failure of Mistral model if it was tried and raised.
            # - Failure of the default HF model if it was tried and raised (and not caught by a more specific except).
            logger.error(f"OCR processing ultimately failed: {str(e)}", exc_info=True)
            raise

    def _get_mistral_ocr(self,
                  model: str = "mistral-ocr-latest",
                  pdf_file_path: str = None,
                  output_file: str = None,
                  output_format: str = "markdown") -> None:
        """Get OCR results using Mistral OCR API.

        Parameters:
            model (str): Mistral OCR model to use
            pdf_file_path (str): Path to PDF file to process
            output_format (str): Format to save output ("markdown", "json", etc.)

        Returns:
            None
        """
        try:
            client = Mistral(api_key=self.mistral_api_key)
            # Check if API key is available
            if not self.mistral_api_key:
                raise ValueError("Mistral API key not found in environment variables")

            with open(pdf_file_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            
            # Get base64 encoded PDF
            ocr_response = client.ocr.process(
                model = model,
                document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
                include_image_base64=True
            )  
            with open(output_file, 'w') as f:
                f.write(ocr_response.pages)
            logger.info("Successfully got OCR from Mistral model")
            return None
        except Exception as e:
            logger.error(f"Mistral OCR processing failed: {e}")
            raise

def _get_hf_ocr(self,
                model: str,
                pdf_file_path: str,
                output_file: str,
                output_format: str) -> None:
    """
    Convert PDF file to text using a specified HuggingFace OCR model.
    Handles general HuggingFace OCR models (like GOT-OCR) and specific ones (like Nanonets).

    Parameters:
        model (str): Specific HuggingFace OCR model name to use.
        pdf_file_path (str): Path to the PDF file.
        output_file (str): Path to save the OCR output.
        output_format (str): Desired output format ("markdown", "text", etc.).

    Raises:
        ValueError: If pdf_file_path is not provided or if the model_name_to_use is invalid.
        Exception: For underlying OCR processing or model loading errors.
    """
    if not pdf_file_path:
        logger.error("PDF file path must be provided for HuggingFace OCR.")
        raise ValueError("PDF file path must be provided")

    if not self._validate_model(model, "ocr", "huggingface"):
        logger.warning(f"The HuggingFace OCR model '{model}' is not in the validated list in config_model.json. This may lead to unexpected behavior if it requires special handling not implemented in the generic path.")
        # Allow proceeding, but the warning is important.

    # Validate user role and permissions using server-side session data
    if not self._check_user_permissions("ocr"):
        logger.error("User does not have permission to perform OCR operations.")
        raise PermissionError("Insufficient permissions for OCR operations")

    logger.info(f"Performing OCR with HuggingFace model: {model} on {pdf_file_path} with format: {output_format}")

    # Rest of the function remains unchanged
    ...
                    model: str,
                    pdf_file_path: str,
                    output_file: str,
                    output_format: str) -> None:
        """
        Convert PDF file to text using a specified HuggingFace OCR model.
        Handles general HuggingFace OCR models (like GOT-OCR) and specific ones (like Nanonets).

        Parameters:
            model (str): Specific HuggingFace OCR model name to use.
            pdf_file_path (str): Path to the PDF file.
            output_file (str): Path to save the OCR output.
            output_format (str): Desired output format ("markdown", "text", etc.).

        Raises:
            ValueError: If pdf_file_path is not provided or if the model_name_to_use is invalid.
            Exception: For underlying OCR processing or model loading errors.
        """
        if not pdf_file_path:
            logger.error("PDF file path must be provided for HuggingFace OCR.")
            raise ValueError("PDF file path must be provided")

        if not self._validate_model(model, "ocr", "huggingface"):
def _get_hf_ocr(self,
                    model: str,
                    pdf_file_path: str,
                    output_file: str,
                    output_format: str) -> None:
        """
        Convert PDF file to text using a specified HuggingFace OCR model.
        Handles general HuggingFace OCR models (like GOT-OCR) and specific ones (like Nanonets).

        Parameters:
            model (str): Specific HuggingFace OCR model name to use.
            pdf_file_path (str): Path to the PDF file.
            output_file (str): Path to save the OCR output.
            output_format (str): Desired output format ("markdown", "text", etc.).

        Raises:
            ValueError: If pdf_file_path is not provided or if the model_name_to_use is invalid.
            Exception: For underlying OCR processing or model loading errors.
        """
        if not pdf_file_path:
            logger.error("PDF file path must be provided for HuggingFace OCR.")
            raise ValueError("PDF file path must be provided")

        if not self._validate_model(model, "ocr", "huggingface"):
            # import html  # Used for HTML escaping to prevent log injection
            logger.warning(f"The HuggingFace OCR model '{html.escape(model)}' is not in the validated list in config_model.json. This may lead to unexpected behavior if it requires special handling not implemented in the generic path.")
            # Allow proceeding, but the warning is important.

        logger.info(f"Performing OCR with HuggingFace model: {html.escape(model)} on {html.escape(pdf_file_path)} with format: {html.escape(output_format)}")

        model_path = os.path.join(self.hf_model_dir, model)
        all_page_texts = []
        temp_image_path = None # For cleanup in finally block

        try:
            from PIL import Image # Ensure PIL is imported here for both paths
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {html.escape(pdf_file_path)}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("")
                    logger.info(f"Created empty output file at {html.escape(output_file)}.")
                return None

            logger.info(f"Extracted {len(images)} page(s) from '{html.escape(pdf_file_path)}'. Processing with HuggingFace OCR model {html.escape(model)}...")

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic
                logger.debug(f"Using Nanonets-specific path for model {html.escape(model)}")
                nanonets_model_instance = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
                )
                nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Successfully loaded Nanonets model, tokenizer, and processor from {html.escape(model_path)}.")

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with Nanonets model.")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image", "image": f"file://{temp_image_path}"},
                                {"type": "text", "text": nanonets_prompt},
                            ]},
                        ]
                        text_prompt_for_processor = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt_for_processor], images=[pil_image], padding=True, return_tensors="pt"
                        )
                        inputs = inputs.to(nanonets_model_instance.device)
                        output_ids = nanonets_model_instance.generate(**inputs, max_new_tokens=8192, do_sample=False)
                        current_input_ids = inputs.input_ids[0]
                        current_output_ids = output_ids[0]
                        generated_part_ids = current_output_ids[len(current_input_ids):]
                        page_text = processor.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset for next iteration or final cleanup

                    # Common page text processing logic (moved outside individual try-finally for page)
                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

{page_text.strip()}")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text": # and other formats
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Nanonets OCR returned no text for page {page_num}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

[No text extracted]
")
                        else:
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
[No text extracted]
")


            else: # Generic OCR model path (e.g. GOT-OCR2_0)
                logger.debug(f"Using generic OCR path for model {html.escape(model)}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                hf_model_instance = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True,
                )
                hf_model_instance = hf_model_instance.eval()
                logger.info(f"Successfully loaded generic HuggingFace model '{html.escape(model)}' and tokenizer.")

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with {html.escape(model)} (generic logic).")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                        page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset

                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

{page_text.strip()}

")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text":
                            if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Generic OCR returned no text for page {page_num} of '{html.escape(pdf_file_path)}' using {html.escape(model)}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

[No text extracted]

")
                        else:
                             if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
[No text extracted]
")

            # Final assembly and writing to file (common to both paths if successful)
            if output_format == "json": # Though JSON population above is commented out
                final_output_content = json.dumps(all_page_texts, indent=2) if isinstance(all_page_texts, list) and all_page_texts and isinstance(all_page_texts[0], dict) else "JSON output format not fully implemented for this model path or data."
                logger.warning("JSON output format for generic OCR is not fully implemented or data was not structured for it.")
            elif output_format == "markdown":
                 final_output_content = "".join(all_page_texts).strip() # Markdown typically joined without extra newlines if page separators handle it
            else: # text or default
                final_output_content = "
".join(all_page_texts).strip()

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_output_content)
                logger.info(f"Successfully saved HuggingFace OCR output to {html.escape(output_file)} for model {html.escape(model)}")
            else:
                logger.warning(f"No output_file was specified for HuggingFace OCR (model {html.escape(model)}). Text not saved.")

        except Exception as e:
            logger.error(f"Overall HuggingFace OCR processing failed for model '{html.escape(model)}' on file '{html.escape(pdf_file_path)}': {e}", exc_info=True)
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"[Error during OCR processing for model {html.escape(model)}: {html.escape(str(e))}]")
                except Exception as write_err:
                    logger.error(f"Failed to write error to output file {html.escape(output_file)}: {write_err}")
            raise # Re-raise the exception to be caught by the calling get_ocr method
            # Allow proceeding, but the warning is important.

        logger.info(f"Performing OCR with HuggingFace model: {model} on {pdf_file_path} with format: {output_format}")

logger.info(f"Performing OCR with HuggingFace model: {model} on {pdf_file_path} with format: {output_format}")

        # import os.path  # Used for secure path joining
        model_path = os.path.normpath(os.path.join(self.hf_model_dir, model))
        if not model_path.startswith(os.path.normpath(self.hf_model_dir)):
            raise ValueError("Invalid model path")

        all_page_texts = []
        temp_image_path = None # For cleanup in finally block
        all_page_texts = []
        temp_image_path = None # For cleanup in finally block

        try:
            from PIL import Image # Ensure PIL is imported here for both paths
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {pdf_file_path}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("")
                    logger.info(f"Created empty output file at {output_file}.")
                return None

logger.info(f"Created empty output file at {output_file}.")
                return None

            # import html  # For HTML escaping
            logger.info(f"Extracted {len(images)} page(s) from '{html.escape(pdf_file_path)}'. Processing with HuggingFace OCR model {html.escape(model)}...")

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic
logger.warning(f"The HuggingFace OCR model '{model}' is not in the validated list in config_model.json. This may lead to unexpected behavior if it requires special handling not implemented in the generic path.")
            # Allow proceeding, but the warning is important.

def _get_hf_ocr(self,
                    model: str,
                    pdf_file_path: str,
                    output_file: str,
                    output_format: str) -> None:
        """
        Convert PDF file to text using a specified HuggingFace OCR model.
        Handles general HuggingFace OCR models (like GOT-OCR) and specific ones (like Nanonets).

        Parameters:
            model (str): Specific HuggingFace OCR model name to use.
            pdf_file_path (str): Path to the PDF file.
            output_file (str): Path to save the OCR output.
            output_format (str): Desired output format ("markdown", "text", etc.).

        Raises:
            ValueError: If pdf_file_path is not provided or if the model_name_to_use is invalid.
            Exception: For underlying OCR processing or model loading errors.
        """
        if not pdf_file_path:
            logger.error("PDF file path must be provided for HuggingFace OCR.")
            raise ValueError("PDF file path must be provided")

        if not self._validate_model(model, "ocr", "huggingface"):
            logger.warning(f"The HuggingFace OCR model '{model}' is not in the validated list in config_model.json. This may lead to unexpected behavior if it requires special handling not implemented in the generic path.")
            # Allow proceeding, but the warning is important.

        # import html  # For HTML escaping
        logger.info(f"Performing OCR with HuggingFace model: {html.escape(model)} on {html.escape(pdf_file_path)} with format: {html.escape(output_format)}")

        model_path = os.path.join(self.hf_model_dir, model)
        all_page_texts = []
        temp_image_path = None # For cleanup in finally block

        try:
            from PIL import Image # Ensure PIL is imported here for both paths
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {html.escape(pdf_file_path)}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("")
                    logger.info(f"Created empty output file at {html.escape(output_file)}.")
                return None

            logger.info(f"Extracted {len(images)} page(s) from '{html.escape(pdf_file_path)}'. Processing with HuggingFace OCR model {html.escape(model)}...")

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic
                logger.debug(f"Using Nanonets-specific path for model {html.escape(model)}")
                nanonets_model_instance = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
                )
                nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Successfully loaded Nanonets model, tokenizer, and processor from {html.escape(model_path)}.")

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with Nanonets model.")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image", "image": f"file://{temp_image_path}"},
                                {"type": "text", "text": nanonets_prompt},
                            ]},
                        ]
                        text_prompt_for_processor = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt_for_processor], images=[pil_image], padding=True, return_tensors="pt"
                        )
                        inputs = inputs.to(nanonets_model_instance.device)
                        output_ids = nanonets_model_instance.generate(**inputs, max_new_tokens=8192, do_sample=False)
                        current_input_ids = inputs.input_ids[0]
                        current_output_ids = output_ids[0]
                        generated_part_ids = current_output_ids[len(current_input_ids):]
                        page_text = processor.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset for next iteration or final cleanup

                    # Common page text processing logic (moved outside individual try-finally for page)
                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

{page_text.strip()}")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text": # and other formats
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Nanonets OCR returned no text for page {page_num}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

[No text extracted]
")
                        else:
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
[No text extracted]
")


            else: # Generic OCR model path (e.g. GOT-OCR2_0)
                logger.debug(f"Using generic OCR path for model {html.escape(model)}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                hf_model_instance = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True,
                )
                hf_model_instance = hf_model_instance.eval()
                logger.info(f"Successfully loaded generic HuggingFace model '{html.escape(model)}' and tokenizer.")

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with {html.escape(model)} (generic logic).")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                        page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset

                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

{page_text.strip()}

")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text":
                            if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Generic OCR returned no text for page {page_num} of '{html.escape(pdf_file_path)}' using {html.escape(model)}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

[No text extracted]

")
                        else:
                             if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
[No text extracted]
")

            # Final assembly and writing to file (common to both paths if successful)
            if output_format == "json": # Though JSON population above is commented out
                final_output_content = json.dumps(all_page_texts, indent=2) if isinstance(all_page_texts, list) and all_page_texts and isinstance(all_page_texts[0], dict) else "JSON output format not fully implemented for this model path or data."
                logger.warning("JSON output format for generic OCR is not fully implemented or data was not structured for it.")
            elif output_format == "markdown":
                 final_output_content = "".join(all_page_texts).strip() # Markdown typically joined without extra newlines if page separators handle it
            else: # text or default
                final_output_content = "
".join(all_page_texts).strip()

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_output_content)
                logger.info(f"Successfully saved HuggingFace OCR output to {html.escape(output_file)} for model {html.escape(model)}")
            else:
                logger.warning(f"No output_file was specified for HuggingFace OCR (model {html.escape(model)}). Text not saved.")

        except Exception as e:
            logger.error(f"Overall HuggingFace OCR processing failed for model '{html.escape(model)}' on file '{html.escape(pdf_file_path)}': {e}", exc_info=True)
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"[Error during OCR processing for model {html.escape(model)}: {html.escape(str(e))}]")
                except Exception as write_err:
                    logger.error(f"Failed to write error to output file {html.escape(output_file)}: {write_err}")
            raise # Re-raise the exception to be caught by the calling get_ocr method

        model_path = os.path.join(self.hf_model_dir, model)
        all_page_texts = []
        temp_image_path = None # For cleanup in finally block

        try:
            from PIL import Image # Ensure PIL is imported here for both paths
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {pdf_file_path}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("")
                    logger.info(f"Created empty output file at {output_file}.")
                return None

            logger.info(f"Extracted {len(images)} page(s) from '{pdf_file_path}'. Processing with HuggingFace OCR model {model}...")

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic
                logger.debug("Using Nanonets-specific path for model %s", model)  # import logging
                nanonets_model_instance = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
                )
                nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Successfully loaded Nanonets model, tokenizer, and processor from {model_path}.")

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug("Processing page %d/%d with Nanonets model.", page_num, len(images))
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image", "image": f"file://{temp_image_path}"},
                                {"type": "text", "text": nanonets_prompt},
                            ]},
                        ]
                        text_prompt_for_processor = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt_for_processor], images=[pil_image], padding=True, return_tensors="pt"
                        )
                        inputs = inputs.to(nanonets_model_instance.device)
                        output_ids = nanonets_model_instance.generate(**inputs, max_new_tokens=8192, do_sample=False)
                        current_input_ids = inputs.input_ids[0]
                        current_output_ids = output_ids[0]
                        generated_part_ids = current_output_ids[len(current_input_ids):]
                        page_text = processor.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset for next iteration or final cleanup

                    # Common page text processing logic (moved outside individual try-finally for page)
                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

{page_text.strip()}")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text": # and other formats
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Nanonets OCR returned no text for page {page_num}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

[No text extracted]
")
                        else:
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
[No text extracted]
")


            else: # Generic OCR model path (e.g. GOT-OCR2_0)
                logger.debug("Using generic OCR path for model %s", model)  # import logging
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                hf_model_instance = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True,
                )
                hf_model_instance = hf_model_instance.eval()
                logger.info(f"Successfully loaded generic HuggingFace model '{model}' and tokenizer.")

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug("Processing page %d/%d with %s (generic logic).", page_num, len(images), model)
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                        page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset

                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

{page_text.strip()}

")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text":
                            if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Generic OCR returned no text for page {page_num} of '{pdf_file_path}' using {model}.")
                        if output_format == "markdown":
                nanonets_model_instance = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
                )
                nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                # import html  # For HTML escaping
                logger.info(f"Successfully loaded Nanonets model, tokenizer, and processor from {html.escape(model_path)}.")  # Sanitize the model_path

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                for i, pil_image in enumerate(images):
                    page_num = i + 1
for i, pil_image in enumerate(images):
                    page_num = i + 1
                    # Use string formatting to avoid potential log injection
                    logger.debug("Processing page %d/%d with Nanonets model.", page_num, len(images))
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
from PIL import Image # Ensure PIL is imported here for both paths
            images = convert_from_path(pdf_file_path)

            if not images:
                logger.warning(f"No images were extracted from PDF: {pdf_file_path}")
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write("")
                    logger.info(f"Created empty output file at {output_file}.")
                return None

            logger.info(f"Extracted {len(images)} page(s) from '{pdf_file_path}'. Processing with HuggingFace OCR model {model}...")

            if model == "nanonets/Nanonets-OCR-s":
                # Nanonets-specific logic
                logger.debug(f"Using Nanonets-specific path for model {model}")
                nanonets_model_instance = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
                )
                nanonets_model_instance.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Successfully loaded Nanonets model, tokenizer, and processor from {model_path}.")

                nanonets_prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with Nanonets model.")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            # import os.path  # For secure path handling
                            # Use os.path.basename to ensure only the filename is used, preventing path traversal
                            safe_filename = os.path.basename(tmp_img_file.name)
                            pil_image.save(safe_filename, format='PNG')
                            temp_image_path = safe_filename

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image", "image": f"file://{temp_image_path}"},
                                {"type": "text", "text": nanonets_prompt},
                            ]},
                        ]
                        text_prompt_for_processor = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt_for_processor], images=[pil_image], padding=True, return_tensors="pt"
                        )
                        inputs = inputs.to(nanonets_model_instance.device)
                        output_ids = nanonets_model_instance.generate(**inputs, max_new_tokens=8192, do_sample=False)
                        current_input_ids = inputs.input_ids[0]
                        current_output_ids = output_ids[0]
                        generated_part_ids = current_output_ids[len(current_input_ids):]
                        page_text = processor.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset for next iteration or final cleanup

                    # Common page text processing logic (moved outside individual try-finally for page)
                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

{page_text.strip()}")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text": # and other formats
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Nanonets OCR returned no text for page {page_num}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

---

Page {page_num}

---

[No text extracted]
")
                        else:
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
[No text extracted]
")


            else: # Generic OCR model path (e.g. GOT-OCR2_0)
                logger.debug(f"Using generic OCR path for model {model}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                hf_model_instance = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True,
                )
                hf_model_instance = hf_model_instance.eval()
                logger.info(f"Successfully loaded generic HuggingFace model '{model}' and tokenizer.")

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with {model} (generic logic).")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            # import os.path  # For secure path handling
                            # Use os.path.basename to ensure only the filename is used, preventing path traversal
                            safe_filename = os.path.basename(tmp_img_file.name)
                            pil_image.save(safe_filename, format='PNG')
                            temp_image_path = safe_filename

                        page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                        page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset

                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

{page_text.strip()}

")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text":
                            if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:
{page_text.strip()}
")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Generic OCR returned no text for page {page_num} of '{pdf_file_path}' using {model}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"

## Page {page_num}

[No text extracted]

")
                        else:
                             if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---
[No text extracted]
")

            # Final assembly and writing to file (common to both paths if successful)
            if output_format == "json": # Though JSON population above is commented out
                final_output_content = json.dumps(all_page_texts, indent=2) if isinstance(all_page_texts, list) and all_page_texts and isinstance(all_page_texts[0], dict) else "JSON output format not fully implemented for this model path or data."
                logger.warning("JSON output format for generic OCR is not fully implemented or data was not structured for it.")
            elif output_format == "markdown":
                 final_output_content = "".join(all_page_texts).strip() # Markdown typically joined without extra newlines if page separators handle it
                            temp_image_path = tmp_img_file.name

                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image", "image": f"file://{temp_image_path}"},
                                {"type": "text", "text": nanonets_prompt},
                            ]},
                        ]
                        text_prompt_for_processor = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt_for_processor], images=[pil_image], padding=True, return_tensors="pt"
                        )
                        inputs = inputs.to(nanonets_model_instance.device)
                        output_ids = nanonets_model_instance.generate(**inputs, max_new_tokens=8192, do_sample=False)
                        current_input_ids = inputs.input_ids[0]
                        current_output_ids = output_ids[0]
                        generated_part_ids = current_output_ids[len(current_input_ids):]
                        page_text = processor.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset for next iteration or final cleanup

                    # Common page text processing logic (moved outside individual try-finally for page)
                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"\n\n## Page {page_num}\n\n{page_text.strip()}\n\n")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text": # and other formats
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:\n{page_text.strip()}\n")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:\n{page_text.strip()}\n")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Nanonets OCR returned no text for page {page_num}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"\n\n---\n\nPage {page_num}\n\n---\n\n[No text extracted]\n")
                        else:
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:\n[No text extracted]\n")


            else: # Generic OCR model path (e.g. GOT-OCR2_0)
                logger.debug(f"Using generic OCR path for model {model}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                hf_model_instance = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='auto', use_safetensors=True,
                )
                hf_model_instance = hf_model_instance.eval()
                logger.info(f"Successfully loaded generic HuggingFace model '{model}' and tokenizer.")

                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    logger.debug(f"Processing page {page_num}/{len(images)} with {model} (generic logic).")
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img_file:
                            pil_image.save(tmp_img_file.name, format='PNG')
                            temp_image_path = tmp_img_file.name

                        page_text_result = hf_model_instance.chat(tokenizer, temp_image_path, ocr_type="ocr")
                        page_text = page_text_result if isinstance(page_text_result, str) else str(page_text_result)
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            temp_image_path = None # Reset

                    if page_text.strip():
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"\n\n## Page {page_num}\n\n{page_text.strip()}\n\n")
                            else: all_page_texts.append(page_text.strip())
                        elif output_format == "text":
                            if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---\n{page_text.strip()}\n")
                            else: all_page_texts.append(page_text.strip())
                        else: # Default to text
                            if len(images) > 1: all_page_texts.append(f"Page {page_num}:\n{page_text.strip()}\n")
                            else: all_page_texts.append(page_text.strip())
                    else:
                        logger.warning(f"Generic OCR returned no text for page {page_num} of '{pdf_file_path}' using {model}.")
                        if output_format == "markdown":
                            if len(images) > 1: all_page_texts.append(f"\n\n## Page {page_num}\n\n[No text extracted]\n\n")
                        else:
                             if len(images) > 1: all_page_texts.append(f"--- Page {page_num} ---\n[No text extracted]\n")

            # Final assembly and writing to file (common to both paths if successful)
            if output_format == "json": # Though JSON population above is commented out
                final_output_content = json.dumps(all_page_texts, indent=2) if isinstance(all_page_texts, list) and all_page_texts and isinstance(all_page_texts[0], dict) else "JSON output format not fully implemented for this model path or data."
                logger.warning("JSON output format for generic OCR is not fully implemented or data was not structured for it.")
            elif output_format == "markdown":
                 final_output_content = "".join(all_page_texts).strip() # Markdown typically joined without extra newlines if page separators handle it
            else: # text or default
                final_output_content = "\n".join(all_page_texts).strip()

            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_output_content)
                logger.info(f"Successfully saved HuggingFace OCR output to {output_file} for model {model}")
            else:
                logger.warning(f"No output_file was specified for HuggingFace OCR (model {model}). Text not saved.")

        except Exception as e:
            logger.error(f"Overall HuggingFace OCR processing failed for model '{model}' on file '{pdf_file_path}': {e}", exc_info=True)
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"[Error during OCR processing for model {model}: {str(e)}]")
                except Exception as write_err:
                    logger.error(f"Failed to write error to output file {output_file}: {write_err}")
            raise # Re-raise the exception to be caught by the calling get_ocr method


    def _calculate_perplexity(self, logprobs: Any) -> Optional[float]:
        """Calculate perplexity score from token log probabilities.
        
        Perplexity is a measurement of how well a probability model predicts a sample.
        Lower perplexity indicates better prediction (lower uncertainty).
        
        Parameters:
            logprobs (Dict[str, Any]): Log probabilities from model response
            
        Returns:
            float: Perplexity score, lower is better
        """
        if logprobs is None:
            return None
        # Handle direct numeric loss values (e.g. HuggingFace / MLX NLL loss)
        if isinstance(logprobs, (int, float)):
            try:
                return math.exp(float(logprobs))
            except Exception:
                return None
        # Handle dictionary-based log probabilities (e.g. Azure / Anthropic responses)
        if not isinstance(logprobs, Dict):
            return None
            return None
            
        try:
            # Extract token logprobs if available
            token_logprobs = logprobs.get('token_logprobs', [])
            if not token_logprobs or len(token_logprobs) == 0:
                return None
                
            # Filter out None values that might be in the logprobs
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]
            if not valid_logprobs or len(valid_logprobs) == 0:
                return None
                
            # Calculate average negative log probability
            avg_negative_logprob = -sum(valid_logprobs) / len(valid_logprobs)
            
            # Perplexity is the exponentiation of the average negative log probability
            perplexity = math.exp(avg_negative_logprob)
            
            return perplexity
            
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")
            return None
    
    def get_tokenisation(self, text: Union[str, List[str]], model: Optional[str] = None) -> int:
        """Initialize or return cached tiktoken encoder.
        
        Args:
            model: Name of the tokeniser to use. Common options:
                - "o200k_base" (used by GPT-4o)
                - "cl100k_base" (default, used by GPT-4, GPT-3.5)
                - "p50k_base" (used by GPT-3)

        Returns:
            Length of the tokenised text
        """
        # Initialize the encoder
        if model is None:
            logger.warning("Model not specified, falling back to the default HuggingFace model")
            model = self.default_hf_tokeniser_model
        
        try:
            if self._validate_model(model, "tokeniser", "azure_openai"):
                try:
                    logger.info(f"Using Azure OpenAI model '{model}' for tokeniser")
                    access_token = self.refresh_token()
                    client = AzureOpenAI(
                        api_version=self.api_version, 
                        azure_endpoint=self.azure_endpoint,
                        azure_ad_token = access_token
                    )
                    response = tiktoken.get_encoding(model)
                    token_length = len(response.encode(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"Azure OpenAI tokeniser failed: {e}")
            elif self._validate_model(model, "tokeniser", "anthropic"):
                try:
                    logger.info(f"Using Anthropic model '{model}' for tokenisation")
                    response = client.messages.count_tokens(
                        model=model,
                        messages=[{"role": "user", "content": text}]
                    )
                    token_length = response['input_tokens']
                    return token_length
                except Exception as e:
                    logger.warning(f"Anthropic tokenisation failed: {e}")
            elif self._validate_model(model, "tokeniser", "vertex"):
                try:
                    response = client.models.count_tokens(
                        model=model,
                        contents=text
                    )
                    token_length = response
                    return token_length
                except Exception as e:
                    logger.warning(f"Vertex tokeniser failed: {e}")
            elif self._validate_model(model, "tokeniser", "huggingface"):
                try:
                    logger.info(f"Using HuggingFace model '{model}' for tokeniser")
                    tokenizer = AutoTokenizer.from_pretrained(model)
                    token_length = len(tokenizer.tokenize(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"HuggingFace tokeniser failed: {e}")
            else:
                # fall back to default huggingface model
                try:
                    logger.info(f"Using default HuggingFace model '{self.default_hf_tokeniser_model}' for tokeniser")
                    tokenizer = AutoTokenizer.from_pretrained(self.default_hf_tokeniser_model)
                    token_length = len(tokenizer.tokenize(text))
                    return token_length
                except Exception as e:
                    logger.warning(f"Default HuggingFace tokeniser failed: {e}")

        except Exception as e:
            logger.error(f"Invalid model '{model}' for tokeniser")
            raise        

class MetaGenerator:
    def __init__(self, generator = None):
        self.aiutility = AIUtility()
        self.generator = generator if generator else Generator()

    def get_meta_completion(self, 
                           application: str,
                           category: str,
                           action: str,
                           prompt_id: Optional[int] = 100,
                           system_prompt: Optional[str] = None,
                           model: Optional[str] = "Qwen3-1.7B",  
                           temperature: Optional[float] = 1,
                           max_tokens: Optional[int] = 1000,
                           top_p: Optional[float] = 1,
                           top_k: Optional[int] = 10,
                           frequency_penalty: Optional[float] = 1, # only available for OpenAI model
                           presence_penalty: Optional[float] = 1,  # only available for OpenAI model
                           seed: Optional[int] = None,
                           logprobs: Optional[bool] = True,
                           num_beam: Optional[int] = None,
                           json_schema: Optional[Dict[str, Any]] = None,
                           return_full_response: Optional[bool] = False,
                           **kwargs) -> Optional[str]:
        """
        Execute a meta-prompt by retrieving the template and filling in the placeholders.
        
        Args:
            application: Application area (e.g., 'metaprompt', 'metaresponse')
            category: Category of the template (e.g., 'manipulation', 'evaluation')
            action: Specific action within the category
            prompt_id: ID for tracking the prompt
            system_prompt: System prompt to guide the model
            model: Model to use for completion
            temperature: Temperature for generation
            **kwargs: Values for template placeholders
            
        Returns:
            Generated response if successful, None otherwise
            
        Raises:
            ValueError: If template is not found or required keys are missing
        """
        
        formatted_prompt = self.aiutility.apply_meta_prompt(application, category, action, **kwargs)
        """
        # Get the meta-prompt template
        template = self.aiutility.get_meta_prompt(application, category, action)
        
        if not template:
            raise ValueError(f"Template not found for {application}/{category}/{action}")
            
        # Parse required keys from template
        formatter = string.Formatter()
        required_keys = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None}
        
        # Check for missing keys
        missing = required_keys - kwargs.keys()
        if missing:
            raise ValueError(f"Missing required keys for meta-prompt: {missing}")
        
        # Check if elements of kwargs are lists and flattern out
        if "task_prompt" in kwargs and kwargs["task_prompt"] is not None:
            if isinstance(kwargs["task_prompt"], list):
                kwargs["task_prompt"] = self.aiutility.format_text_list(kwargs["task_prompt"], "prompt")
            elif isinstance(kwargs["task_prompt"], dict):
                kwargs["task_prompt"] = str(kwargs["task_prompt"])
        if "response" in kwargs and kwargs["response"] is not None:
            if isinstance(kwargs["response"], list):
                kwargs["response"] = self.aiutility.format_text_list(kwargs["response"], "response")
            elif isinstance(kwargs["response"], dict):
                kwargs["response"] = str(kwargs["response"])

        # Format the meta prompt template with **kwargs
        try:
            formatted_prompt = template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Error formatting template: {e}")
        
        logger.debug(f"Formatted meta-prompt: {formatted_prompt}")
        """

        logger.debug(f"Model: {model}")
        # Execute using the generator
        try:
            response = self.generator.get_completion(
                prompt_id=prompt_id,  # Using default prompt ID
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                model=model,
                stored_df = None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty,
                seed=seed,
                logprobs=logprobs,
                num_beam=num_beam,
                json_schema=json_schema,
                return_full_response=return_full_response
            )
            return response

        except Exception as e:
            logger.error(f"Error executing meta-prompt: {e}")
            return None




# standardise encoder models
# developing
class Encoder:
    """
    Handles encoding of text using pre-trained models, including bi-encoders, cross-encoders, and rerankers.
    Supports loading models from local directories or using pre-trained models from HuggingFace.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model_type: str = "bi-encoder", local_models_dir: str = None):
        """
        Initialize the Encoder.
        
        Args:
            model_name: Name of the pre-trained model or path to local model
            model_type: Type of encoder model ('bi-encoder', 'cross-encoder', 'reranker')
            local_models_dir: Directory containing local models. If None, uses current working directory / local_models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.local_models_dir = Path(local_models_dir) if local_models_dir else Path.cwd() / "local_models"
        self.model = None
        
        # Load model configuration
        try:
            config_dir = Path.cwd() / "config"
            model_config_path = config_dir / "config_model.json"
            with open(model_config_path, 'r') as f:
                self.model_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model configuration: {e}")
            self.model_config = None
        
        # Load the appropriate model based on model_type
        self._load_model()
    
    def _load_model(self):
        """
        Load the appropriate model based on model_type.
        """
        try:
            # Check if model exists locally
            model_path = self.local_models_dir / self.model_name
            use_local = model_path.exists()
            model_source = model_path if use_local else self.model_name
            
            logger.info(f"Loading {self.model_type} model from {'local path' if use_local else 'HuggingFace'}: {model_source}")
            
            if self.model_type == "bi-encoder":
                self.model = SentenceTransformer(model_source)
            elif self.model_type == "cross-encoder":
                self.model = CrossEncoder(model_source)
            elif self.model_type == "reranker":
                # Rerankers are typically implemented as CrossEncoders
                self.model = CrossEncoder(model_source)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Successfully loaded {self.model_type} model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model: {e}")
            raise
    
    def _validate_model(self, model_name: str, model_type: str) -> bool:
        """
        Validate model name against configuration.

        Parameters:
            model_name (str): Name of the model to validate
            model_type (str): Type of model (bi-encoder, cross-encoder, reranker)

        Returns:
            bool: True if model is valid, False otherwise
        """
        if not self.model_config:
            # If config couldn't be loaded, assume model is valid
            return True
            
        try:
            # Map model_type to config model type
            config_model_type = {
                "bi-encoder": "embedding",
                "cross-encoder": "embedding",
                "reranker": "reranker"
            }.get(model_type)
            
            if not config_model_type or config_model_type not in self.model_config["validation_rules"]["models"]:
                logger.warning(f"Model type {model_type} not found in configuration")
                return False
                
            # Check if model exists in huggingface models list
            huggingface_models = self.model_config["validation_rules"]["models"][config_model_type].get("huggingface", [])
            return model_name in huggingface_models
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode text using the pre-trained bi-encoder model.
        
        Args:
            text: Text to encode
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            np.ndarray: Encoded text
        """
        if self.model_type != "bi-encoder":
            raise ValueError(f"encode method is only available for bi-encoder models, not {self.model_type}")
        return self.model.encode(text, **kwargs)
    
    def predict(self, texts: Union[List[str], List[Tuple[str, str]]], **kwargs) -> np.ndarray:
        """
        Predict similarity scores using cross-encoder or reranker model.
        
        Args:
            texts: For cross-encoders/rerankers, a list of text pairs (sentence1, sentence2)
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            np.ndarray: Similarity scores
        """
        if self.model_type not in ["cross-encoder", "reranker"]:
            raise ValueError(f"predict method is only available for cross-encoder or reranker models, not {self.model_type}")
        return self.model.predict(texts, **kwargs)
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return, if None returns all
            
        Returns:
            List[Dict]: List of dictionaries with document index, score, and text
        """
        if self.model_type != "reranker":
            raise ValueError(f"rerank method is only available for reranker models, not {self.model_type}")
            
        # Create text pairs for the reranker
        text_pairs = [(query, doc) for doc in documents]
        
        # Get similarity scores
        scores = self.model.predict(text_pairs)
        
        # Combine passages with scores
        passage_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        reranked_pairs = sorted(passage_score_pairs, key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            reranked_pairs = reranked_pairs[:top_k]
            
        return reranked_pairs
    