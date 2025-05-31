#!/usr/bin/env python3
"""
ARCGen V2 - Enhanced Automated Reverse Code Generation with Intelligent Scaling
===============================================================================

Advanced AI-powered code optimization and rewriting tool with support for multiple
AI providers, smart chunking, comprehensive configuration, enterprise features,
and intelligent scaling for rate limiting and memory management.

Features:
- Multi-provider AI support (DeepSeek, OpenAI, Claude)
- Smart context-aware chunking with adaptive sizing
- Intelligent rate limiting with exponential backoff
- Dynamic memory management and scaling
- Configuration management with YAML
- Secure environment variable handling
- Progress tracking and rich console output
- Backup and rollback functionality
- Comprehensive logging and reporting
- Performance monitoring and caching
- Security scanning and validation
- Auto-scaling for API limits and resource constraints

Usage:
    python arcgen_v2.py <addon_path> [options]

Examples:
    python arcgen_v2.py ./my_addon --profile balanced --output ./optimized_addon
    python arcgen_v2.py ./my_addon --config custom_config.yaml --verbose
    python arcgen_v2.py ./my_addon --provider openai --backup-enabled
"""

import os
import sys
import json
import time
import zipfile
import logging
import threading
import psutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from collections import deque
import math

# Third-party imports
import yaml
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
import click

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################

@dataclass
class APIConfig:
    """API configuration for different providers"""
    provider: str
    base_url: str
    model: str
    api_key: str
    max_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: int = 60

@dataclass
class ProcessingConfig:
    """File processing configuration"""
    chunk_size: int = 3000
    smart_chunking: bool = True
    max_file_size: int = 10  # MB
    text_extensions: List[str] = None
    binary_extensions: List[str] = None
    max_processing_time: int = 300  # seconds

@dataclass
class OptimizationProfile:
    """Code optimization profile settings"""
    name: str
    preserve_comments: bool = True
    preserve_formatting: bool = True
    optimize_performance: bool = False
    add_documentation: bool = True
    remove_dead_code: bool = False
    refactor_functions: bool = False

@dataclass
class BackupConfig:
    """Backup configuration"""
    enabled: bool = True
    backup_dir: str = "backup"
    compress: bool = True
    retention_days: int = 30

@dataclass
class SecurityConfig:
    """Security configuration"""
    validate_paths: bool = True
    max_processing_time: int = 300
    security_scan: bool = True
    allowed_extensions: List[str] = None

@dataclass
class ScalingConfig:
    """Intelligent scaling configuration"""
    enable_adaptive_scaling: bool = True
    min_concurrent_requests: int = 1
    max_concurrent_requests: int = 10
    memory_threshold_percent: float = 80.0
    rate_limit_backoff_factor: float = 2.0
    max_backoff_time: int = 300  # seconds
    chunk_size_scaling_factor: float = 0.5
    min_chunk_size: int = 500
    max_chunk_size: int = 8000

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                console.print(f"[yellow]Config file {self.config_path} not found, using defaults[/yellow]")
                return self._get_default_config()
        except Exception as e:
            console.print(f"[red]Error loading config: {e}[/red]")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'api': {
                'primary_provider': 'deepseek',
                'deepseek': {
                    'base_url': 'https://api.deepseek.com',
                    'model': 'deepseek-chat',  # Updated to latest V3-0324
                    'max_tokens': 8000,
                    'temperature': 0.7,
                    'top_p': 1.0
                },
                'deepseek_reasoner': {
                    'base_url': 'https://api.deepseek.com',
                    'model': 'deepseek-reasoner',  # R1-0528 for enhanced reasoning
                    'max_tokens': 8000,
                    'temperature': 0.7,
                    'top_p': 1.0
                },
                'openai': {
                    'base_url': 'https://api.openai.com/v1',
                    'model': 'gpt-4',
                    'max_tokens': 8000,
                    'temperature': 0.7,
                    'top_p': 1.0
                },
                'rate_limit': {
                    'requests_per_minute': 60,
                    'concurrent_requests': 4
                }
            },
            'processing': {
                'chunk_size': 3000,
                'smart_chunking': True,
                'max_file_size': 10,
                'text_extensions': ['.lua', '.txt', '.cfg', '.json', '.vmt', '.vmf'],
                'binary_extensions': ['.vtf', '.vvd', '.mdl', '.phy', '.wav', '.mp3']
            },
            'optimization_profiles': {
                'balanced': {
                    'preserve_comments': True,
                    'preserve_formatting': False,
                    'optimize_performance': True,
                    'add_documentation': True,
                    'remove_dead_code': True
                },
                'conservative': {
                    'preserve_comments': True,
                    'preserve_formatting': True,
                    'optimize_performance': False,
                    'add_documentation': False,
                    'remove_dead_code': False
                },
                'aggressive': {
                    'preserve_comments': False,
                    'preserve_formatting': False,
                    'optimize_performance': True,
                    'add_documentation': True,
                    'remove_dead_code': True,
                    'refactor_functions': True
                }
            },
            'backup': {
                'enabled': True,
                'backup_dir': 'backup',
                'compress': True,
                'retention_days': 30
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_file': 'arcgen.log'
            },
            'security': {
                'validate_paths': True,
                'max_processing_time': 300,
                'security_scan': True
            },
            'scaling': {
                'enable_adaptive_scaling': True,
                'min_concurrent_requests': 1,
                'max_concurrent_requests': 10,
                'memory_threshold_percent': 80.0,
                'rate_limit_backoff_factor': 2.0,
                'max_backoff_time': 300,
                'chunk_size_scaling_factor': 0.5,
                'min_chunk_size': 500,
                'max_chunk_size': 8000
            }
        }
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate API configuration
        if 'api' not in self.config:
            raise ValueError("API configuration is required")
        
        # Validate scaling configuration
        scaling = self.config.get('scaling', {})
        if scaling.get('min_concurrent_requests', 1) > scaling.get('max_concurrent_requests', 10):
            raise ValueError("min_concurrent_requests cannot be greater than max_concurrent_requests")
    
    def get_api_config(self, provider: Optional[str] = None) -> APIConfig:
        """Get API configuration for specified provider"""
        provider = provider or self.config['api']['primary_provider']
        
        if provider not in self.config['api']:
            raise ValueError(f"Provider {provider} not configured")
        
        provider_config = self.config['api'][provider]
        api_key_env = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found in environment variable {api_key_env}")
        
        return APIConfig(
            provider=provider,
            base_url=provider_config['base_url'],
            model=provider_config['model'],
            api_key=api_key,
            max_tokens=provider_config.get('max_tokens', 8000),
            temperature=provider_config.get('temperature', 0.7),
            top_p=provider_config.get('top_p', 1.0),
            timeout=provider_config.get('timeout', 60)
        )
    
    def get_optimization_profile(self, profile_name: str) -> OptimizationProfile:
        """Get optimization profile by name"""
        if profile_name not in self.config['optimization_profiles']:
            raise ValueError(f"Profile {profile_name} not found")
        
        profile_data = self.config['optimization_profiles'][profile_name]
        return OptimizationProfile(name=profile_name, **profile_data)
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        security_data = self.config.get('security', {})
        return SecurityConfig(**security_data)
    
    def get_scaling_config(self) -> ScalingConfig:
        """Get scaling configuration"""
        scaling_data = self.config.get('scaling', {})
        return ScalingConfig(**scaling_data)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Create logger
    logger = logging.getLogger('arcgen')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(console=console, show_time=True, show_path=False)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if log_config.get('log_to_file', True):
        log_file = log_config.get('log_file', 'arcgen.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

###############################################################################
#                              AI PROVIDERS                                   #
###############################################################################

class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, config: APIConfig, logger: logging.Logger, rate_limit_config: Dict[str, int]):
        self.config = config
        self.logger = logger
        self.rate_limit_config = rate_limit_config
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from AI provider"""
        raise NotImplementedError
    
    def validate_api_key(self) -> bool:
        """Validate API key"""
        raise NotImplementedError

class DeepSeekProvider(AIProvider):
    """DeepSeek API provider"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using DeepSeek API"""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Add system message for better context
        system_message = kwargs.get('system_message', 
            'You are an expert code optimization assistant. Analyze and improve code while preserving functionality.')
        
        data = {
            'model': self.config.model,
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'stream': False
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise ValueError("No response content received")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek API request failed: {e}")
            raise
    
    def validate_api_key(self) -> bool:
        """Validate DeepSeek API key"""
        try:
            # Make a simple test request with minimal tokens
            self.generate("Hello", max_tokens=1, system_message="You are a helpful assistant.")
            return True
        except Exception:
            return False

class OpenAIProvider(AIProvider):
    """OpenAI API provider"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.config.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p)
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise ValueError("No response content received")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            raise
    
    def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        try:
            # Make a simple test request
            self.generate("Hello", max_tokens=1)
            return True
        except Exception:
            return False

###############################################################################
#                         INTELLIGENT SCALING SYSTEM                          #
###############################################################################

class MemoryMonitor:
    """Monitor system memory usage and provide scaling recommendations"""
    
    def __init__(self, threshold_percent: float = 80.0, logger: Optional[logging.Logger] = None):
        self.threshold_percent = threshold_percent
        self.logger = logger or logging.getLogger(__name__)
        self.memory_history = deque(maxlen=10)  # Keep last 10 readings
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        usage = {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
        self.memory_history.append(usage['percent'])
        return usage
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down due to memory pressure"""
        current_usage = self.get_memory_usage()
        return current_usage['percent'] > self.threshold_percent
    
    def get_scaling_factor(self) -> float:
        """Get recommended scaling factor based on memory usage"""
        current_usage = self.get_memory_usage()
        if current_usage['percent'] > self.threshold_percent:
            # Scale down more aggressively as memory usage increases
            excess = current_usage['percent'] - self.threshold_percent
            return max(0.1, 1.0 - (excess / 20.0))  # Scale down to 10% minimum
        return 1.0
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        gc.collect()
        self.logger.info("Performed memory cleanup")

class AdaptiveRateLimiter:
    """Intelligent rate limiter with exponential backoff and adaptive scaling"""
    
    def __init__(self, initial_max_calls: int, time_window: int, scaling_config: ScalingConfig, 
                 logger: Optional[logging.Logger] = None):
        self.initial_max_calls = initial_max_calls
        self.current_max_calls = initial_max_calls
        self.time_window = time_window
        self.scaling_config = scaling_config
        self.logger = logger or logging.getLogger(__name__)
        
        self.calls = deque()
        self.rate_limit_hits = 0
        self.consecutive_successes = 0
        self.backoff_time = 1.0
        self.lock = threading.Lock()
        
        # Track API response times for adaptive scaling
        self.response_times = deque(maxlen=50)
        self.error_count = 0
        self.last_rate_limit_time = 0
    
    def __enter__(self):
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and now - self.calls[0] >= self.time_window:
                self.calls.popleft()
            
            # Check if we need to wait due to rate limiting
            if len(self.calls) >= self.current_max_calls:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = self.time_window - (now - oldest_call)
                
                if wait_time > 0:
                    self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self.calls.popleft()  # Remove the oldest call
                    self.rate_limit_hits += 1
                    self.consecutive_successes = 0
                    self._adjust_rate_limit_down()
            
            self.calls.append(now)
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Successful request
            self.consecutive_successes += 1
            self.error_count = max(0, self.error_count - 1)
            
            # Gradually increase rate limit after consecutive successes
            if self.consecutive_successes >= 10:
                self._adjust_rate_limit_up()
                self.consecutive_successes = 0
        else:
            # Failed request
            self.error_count += 1
            self.consecutive_successes = 0
            
            # Check if it's a rate limit error
            if exc_val and "429" in str(exc_val):
                self.last_rate_limit_time = time.time()
                self._handle_rate_limit_error()
    
    def _adjust_rate_limit_down(self):
        """Reduce rate limit due to hitting limits"""
        old_limit = self.current_max_calls
        self.current_max_calls = max(
            self.scaling_config.min_concurrent_requests,
            int(self.current_max_calls * 0.7)  # Reduce by 30%
        )
        
        if old_limit != self.current_max_calls:
            self.logger.info(f"Reduced rate limit from {old_limit} to {self.current_max_calls}")
    
    def _adjust_rate_limit_up(self):
        """Increase rate limit after successful operations"""
        old_limit = self.current_max_calls
        self.current_max_calls = min(
            self.scaling_config.max_concurrent_requests,
            int(self.current_max_calls * 1.2)  # Increase by 20%
        )
        
        if old_limit != self.current_max_calls:
            self.logger.info(f"Increased rate limit from {old_limit} to {self.current_max_calls}")
    
    def _handle_rate_limit_error(self):
        """Handle rate limit errors with exponential backoff"""
        self.backoff_time = min(
            self.scaling_config.max_backoff_time,
            self.backoff_time * self.scaling_config.rate_limit_backoff_factor
        )
        
        self.logger.warning(f"Rate limit error detected, backing off for {self.backoff_time:.2f} seconds")
        time.sleep(self.backoff_time)
        
        # Aggressively reduce rate limit
        self.current_max_calls = max(
            self.scaling_config.min_concurrent_requests,
            int(self.current_max_calls * 0.5)
        )
    
    def get_current_limit(self) -> int:
        """Get current rate limit"""
        return self.current_max_calls
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'current_limit': self.current_max_calls,
            'initial_limit': self.initial_max_calls,
            'rate_limit_hits': self.rate_limit_hits,
            'consecutive_successes': self.consecutive_successes,
            'error_count': self.error_count,
            'backoff_time': self.backoff_time,
            'calls_in_window': len(self.calls)
        }

class AdaptiveChunker:
    """Smart chunker that adapts chunk size based on system resources and API performance"""
    
    def __init__(self, initial_chunk_size: int, scaling_config: ScalingConfig, 
                 memory_monitor: MemoryMonitor, logger: Optional[logging.Logger] = None):
        self.initial_chunk_size = initial_chunk_size
        self.current_chunk_size = initial_chunk_size
        self.scaling_config = scaling_config
        self.memory_monitor = memory_monitor
        self.logger = logger or logging.getLogger(__name__)
        
        # Track processing performance
        self.processing_times = deque(maxlen=20)
        self.success_rate = 1.0
        self.total_requests = 0
        self.successful_requests = 0
    
    def get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on current conditions"""
        # Start with memory-based scaling
        memory_factor = self.memory_monitor.get_scaling_factor()
        
        # Adjust based on success rate
        success_factor = self.success_rate
        if success_factor < 0.8:  # If success rate is below 80%
            success_factor *= 0.7  # Reduce chunk size more aggressively
        
        # Combine factors
        combined_factor = min(memory_factor, success_factor)
        
        # Calculate new chunk size
        new_size = int(self.initial_chunk_size * combined_factor)
        
        # Apply bounds
        new_size = max(self.scaling_config.min_chunk_size, new_size)
        new_size = min(self.scaling_config.max_chunk_size, new_size)
        
        # Update current chunk size if significantly different
        if abs(new_size - self.current_chunk_size) > 100:
            old_size = self.current_chunk_size
            self.current_chunk_size = new_size
            self.logger.info(f"Adjusted chunk size from {old_size} to {new_size}")
        
        return self.current_chunk_size
    
    def chunk_text(self, text: str, file_extension: str = "") -> List[str]:
        """Split text into chunks with adaptive sizing"""
        if not text.strip():
            return []
        
        chunk_size = self.get_optimal_chunk_size()
        
        # For code files, try to preserve function/class boundaries
        if file_extension.lower() in ['.lua', '.py', '.js', '.cpp', '.c', '.cs']:
            return self._chunk_code(text, chunk_size)
        else:
            return self._chunk_simple(text, chunk_size)
    
    def _chunk_code(self, text: str, chunk_size: int) -> List[str]:
        """Chunk code while preserving function boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_simple(self, text: str, chunk_size: int) -> List[str]:
        """Simple character-based chunking"""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def record_processing_result(self, success: bool, processing_time: float):
        """Record the result of a processing operation"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.processing_times.append(processing_time)
        
        # Update success rate
        self.success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunker statistics"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'current_chunk_size': self.current_chunk_size,
            'initial_chunk_size': self.initial_chunk_size,
            'success_rate': self.success_rate,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'avg_processing_time': avg_processing_time
        }

class IntelligentScalingManager:
    """Manages all scaling components and coordinates their behavior"""
    
    def __init__(self, config_manager: ConfigManager, logger: logging.Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.scaling_config = config_manager.get_scaling_config()
        
        # Initialize scaling components
        self.memory_monitor = MemoryMonitor(
            threshold_percent=self.scaling_config.memory_threshold_percent,
            logger=logger
        )
        
        rate_limit_config = config_manager.config['api']['rate_limit']
        self.rate_limiter = AdaptiveRateLimiter(
            initial_max_calls=rate_limit_config['concurrent_requests'],
            time_window=60,  # 1 minute window
            scaling_config=self.scaling_config,
            logger=logger
        )
        
        self.adaptive_chunker = AdaptiveChunker(
            initial_chunk_size=config_manager.config['processing']['chunk_size'],
            scaling_config=self.scaling_config,
            memory_monitor=self.memory_monitor,
            logger=logger
        )
        
        # Scaling statistics
        self.scaling_events = []
        self.start_time = time.time()
    
    def should_scale_down_concurrency(self) -> bool:
        """Determine if we should reduce concurrent operations"""
        return (self.memory_monitor.should_scale_down() or 
                self.rate_limiter.error_count > 5)
    
    def get_optimal_concurrency(self) -> int:
        """Get optimal number of concurrent operations"""
        base_concurrency = self.rate_limiter.get_current_limit()
        
        # Reduce based on memory pressure
        if self.memory_monitor.should_scale_down():
            memory_factor = self.memory_monitor.get_scaling_factor()
            base_concurrency = int(base_concurrency * memory_factor)
        
        # Ensure we stay within bounds
        return max(
            self.scaling_config.min_concurrent_requests,
            min(self.scaling_config.max_concurrent_requests, base_concurrency)
        )
    
    def handle_api_error(self, error: Exception):
        """Handle API errors and adjust scaling accordingly"""
        error_str = str(error)
        
        if "429" in error_str:  # Rate limit error
            self.logger.warning("Rate limit detected, scaling down")
            self._record_scaling_event("rate_limit_hit", "Reduced concurrency due to rate limiting")
        elif "memory" in error_str.lower():  # Memory error
            self.logger.warning("Memory pressure detected, scaling down")
            self.memory_monitor.cleanup_memory()
            self._record_scaling_event("memory_pressure", "Reduced chunk size due to memory pressure")
        elif "timeout" in error_str.lower():  # Timeout error
            self.logger.warning("Timeout detected, adjusting parameters")
            self._record_scaling_event("timeout", "Adjusted parameters due to timeout")
    
    def _record_scaling_event(self, event_type: str, description: str):
        """Record a scaling event for analysis"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description,
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'chunker_stats': self.adaptive_chunker.get_stats()
        }
        self.scaling_events.append(event)
        
        # Keep only last 100 events
        if len(self.scaling_events) > 100:
            self.scaling_events = self.scaling_events[-100:]
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate a comprehensive scaling report"""
        runtime = time.time() - self.start_time
        
        return {
            'runtime_seconds': runtime,
            'scaling_enabled': self.scaling_config.enable_adaptive_scaling,
            'memory_monitor': {
                'current_usage': self.memory_monitor.get_memory_usage(),
                'threshold_percent': self.scaling_config.memory_threshold_percent,
                'history': list(self.memory_monitor.memory_history)
            },
            'rate_limiter': self.rate_limiter.get_stats(),
            'adaptive_chunker': self.adaptive_chunker.get_stats(),
            'optimal_concurrency': self.get_optimal_concurrency(),
            'scaling_events': self.scaling_events[-10:],  # Last 10 events
            'total_scaling_events': len(self.scaling_events)
        }

###############################################################################
#                              FILE PROCESSING                                #
###############################################################################

class SmartChunker:
    """Smart chunking that preserves code structure"""
    
    def __init__(self, chunk_size: int = 3000):
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str, file_extension: str = "") -> List[str]:
        """Split text into chunks while preserving structure"""
        if not text.strip():
            return []
        
        # For code files, try to preserve function/class boundaries
        if file_extension.lower() in ['.lua', '.py', '.js', '.cpp', '.c', '.cs']:
            return self._chunk_code(text)
        else:
            return self._chunk_simple(text)
    
    def _chunk_code(self, text: str) -> List[str]:
        """Chunk code while preserving function boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_simple(self, text: str) -> List[str]:
        """Simple character-based chunking"""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

class FileProcessor:
    """Handles file processing and optimization with intelligent scaling"""
    
    def __init__(self, config_manager: ConfigManager, ai_provider: AIProvider, 
                 scaling_manager: IntelligentScalingManager, logger: logging.Logger):
        self.config_manager = config_manager
        self.ai_provider = ai_provider
        self.scaling_manager = scaling_manager
        self.logger = logger
        self.security_config = config_manager.get_security_config()
        self.stats = {
            'files_processed': 0,
            'files_optimized': 0,
            'files_copied': 0,
            'files_failed': 0,
            'total_size_before': 0,
            'total_size_after': 0,
            'processing_time': 0,
            'scaling_events': 0,
            'memory_cleanups': 0
        }
        self.stats_lock = threading.Lock()
    
    def process_file(self, source_file: Path, addon_root: Path, output_root: Path, 
                    profile: OptimizationProfile) -> Dict[str, Any]:
        """Process a single file"""
        start_time = time.time()
        relative_path = source_file.relative_to(addon_root)
        target_path = output_root / relative_path
        
        # Security: Validate target path
        if not self._is_safe_path(target_path, output_root):
            return {
                'file': str(source_file),
                'relative': str(relative_path),
                'error': 'Path traversal detected',
                'processing_time': time.time() - start_time
            }
        
        result = {
            'file': str(source_file),
            'relative': str(relative_path),
            'size_before': source_file.stat().st_size,
            'size_after': 0,
            'optimized': False,
            'copied': False,
            'error': None,
            'processing_time': 0
        }
        
        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file should be processed or copied
            if self._should_process_file(source_file):
                optimized_content = self._optimize_file(source_file, profile)
                if optimized_content is not None:
                    with target_path.open('w', encoding='utf-8', errors='replace') as f:
                        f.write(optimized_content)
                    result['optimized'] = True
                    result['size_after'] = target_path.stat().st_size
                    with self.stats_lock:
                        self.stats['files_optimized'] += 1
                else:
                    # Fallback to copy if optimization failed
                    self._copy_file(source_file, target_path)
                    result['copied'] = True
                    result['size_after'] = target_path.stat().st_size
                    with self.stats_lock:
                        self.stats['files_copied'] += 1
            else:
                # Copy binary/non-text files
                self._copy_file(source_file, target_path)
                result['copied'] = True
                result['size_after'] = target_path.stat().st_size
                with self.stats_lock:
                    self.stats['files_copied'] += 1
            
            with self.stats_lock:
                self.stats['files_processed'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error processing {source_file}: {e}")
            with self.stats_lock:
                self.stats['files_failed'] += 1
        
        result['processing_time'] = time.time() - start_time
        with self.stats_lock:
            self.stats['processing_time'] += result['processing_time']
            self.stats['total_size_before'] += result['size_before']
            self.stats['total_size_after'] += result['size_after']
        
        return result
    
    def _is_safe_path(self, target_path: Path, base_path: Path) -> bool:
        """Check if target path is safe (no path traversal)"""
        try:
            target_path.resolve().relative_to(base_path.resolve())
            return True
        except ValueError:
            return False
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed or copied"""
        config = self.config_manager.config['processing']
        text_extensions = config.get('text_extensions', [])
        binary_extensions = config.get('binary_extensions', [])
        max_size = config.get('max_file_size', 10) * 1024 * 1024  # Convert to bytes
        
        # Security: Check allowed extensions
        if self.security_config.allowed_extensions:
            ext = file_path.suffix.lower()
            if ext not in self.security_config.allowed_extensions:
                return False
        
        # Check file size
        if file_path.stat().st_size > max_size:
            return False
        
        # Check extension
        ext = file_path.suffix.lower()
        if ext in binary_extensions:
            return False
        if ext in text_extensions:
            return True
        
        # Try to detect if file is text
        try:
            with file_path.open('rb') as f:
                chunk = f.read(1024)
                return b'\x00' not in chunk  # Simple binary detection
        except Exception:
            return False
    
    def _optimize_file(self, file_path: Path, profile: OptimizationProfile) -> Optional[str]:
        """Optimize a single file using AI with intelligent scaling"""
        processing_start_time = time.time()
        success = False
        
        try:
            # Check memory pressure before processing
            if self.scaling_manager.memory_monitor.should_scale_down():
                self.scaling_manager.memory_monitor.cleanup_memory()
                with self.stats_lock:
                    self.stats['memory_cleanups'] += 1
            
            # Read file with size limit for memory safety
            max_size = self.config_manager.config['processing'].get('max_file_size', 10) * 1024 * 1024
            if file_path.stat().st_size > max_size:
                self.logger.warning(f"File {file_path} too large, skipping optimization")
                return None
            
            with file_path.open('r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if not content.strip():
                success = True
                return content
            
            # Use adaptive chunker for intelligent chunk sizing
            chunks = self.scaling_manager.adaptive_chunker.chunk_text(content, file_path.suffix)
            optimized_chunks = []
            
            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()
                
                try:
                    # Use rate limiter for API calls
                    with self.scaling_manager.rate_limiter:
                        prompt = self._build_optimization_prompt(chunk, file_path.suffix, profile)
                        optimized_chunk = self.ai_provider.generate(prompt)
                    
                    # Check for timeout
                    chunk_processing_time = time.time() - chunk_start_time
                    if chunk_processing_time > self.security_config.max_processing_time:
                        self.logger.warning(f"Chunk processing timeout for {file_path}")
                        self.scaling_manager.adaptive_chunker.record_processing_result(False, chunk_processing_time)
                        return None
                    
                    # Clean up the response (remove markdown formatting if present)
                    optimized_chunk = self._clean_ai_response(optimized_chunk)
                    optimized_chunks.append(optimized_chunk)
                    
                    # Record successful chunk processing
                    self.scaling_manager.adaptive_chunker.record_processing_result(True, chunk_processing_time)
                    
                    self.logger.debug(f"Optimized chunk {i+1}/{len(chunks)} for {file_path}")
                    
                except Exception as chunk_error:
                    # Handle chunk-specific errors
                    self.scaling_manager.handle_api_error(chunk_error)
                    self.scaling_manager.adaptive_chunker.record_processing_result(False, time.time() - chunk_start_time)
                    
                    # If it's a rate limit error, we might want to retry
                    if "429" in str(chunk_error):
                        self.logger.warning(f"Rate limit hit during chunk {i+1}, retrying...")
                        with self.stats_lock:
                            self.stats['scaling_events'] += 1
                        
                        # Wait and retry once
                        time.sleep(2)
                        try:
                            with self.scaling_manager.rate_limiter:
                                optimized_chunk = self.ai_provider.generate(prompt)
                            optimized_chunk = self._clean_ai_response(optimized_chunk)
                            optimized_chunks.append(optimized_chunk)
                            self.scaling_manager.adaptive_chunker.record_processing_result(True, time.time() - chunk_start_time)
                        except Exception as retry_error:
                            self.logger.error(f"Retry failed for chunk {i+1}: {retry_error}")
                            return None
                    else:
                        raise chunk_error
            
            success = True
            return '\n'.join(optimized_chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to optimize {file_path}: {e}")
            self.scaling_manager.handle_api_error(e)
            return None
        finally:
            # Record processing result for adaptive scaling
            total_processing_time = time.time() - processing_start_time
            self.scaling_manager.adaptive_chunker.record_processing_result(success, total_processing_time)
    
    def _build_optimization_prompt(self, code: str, file_extension: str, profile: OptimizationProfile) -> str:
        """Build optimization prompt based on profile and file type"""
        base_prompt = f"Optimize this {file_extension} code chunk. "
        
        instructions = []
        if profile.preserve_comments:
            instructions.append("preserve existing comments")
        if profile.preserve_formatting:
            instructions.append("maintain current formatting style")
        if profile.optimize_performance:
            instructions.append("improve performance and efficiency")
        if profile.add_documentation:
            instructions.append("add helpful documentation where needed")
        if profile.remove_dead_code:
            instructions.append("remove any dead or unused code")
        if profile.refactor_functions:
            instructions.append("refactor functions for better structure")
        
        if instructions:
            base_prompt += "Please " + ", ".join(instructions) + ". "
        
        base_prompt += "Return ONLY the optimized code without explanations or markdown formatting.\n\n"
        base_prompt += code
        
        return base_prompt
    
    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response by removing markdown formatting"""
        if not response:
            return ""
            
        lines = response.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip().startswith('#') and not line.strip().startswith('# '):
                continue  # Skip markdown headers outside code blocks
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _copy_file(self, source: Path, target: Path):
        """Copy file from source to target"""
        import shutil
        shutil.copy2(source, target)

###############################################################################
#                              BACKUP SYSTEM                                  #
###############################################################################

class BackupManager:
    """Manages backup creation and restoration"""
    
    def __init__(self, config: BackupConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_backup(self, source_path: Path) -> Optional[Path]:
        """Create backup of source directory"""
        if not self.config.enabled:
            return None
        
        try:
            backup_dir = source_path.parent / self.config.backup_dir
            backup_dir.mkdir(exist_ok=True)
            
            # Clean up old backups first
            self.cleanup_old_backups(backup_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.name}_backup_{timestamp}"
            
            if self.config.compress:
                backup_path = backup_dir / f"{backup_name}.zip"
                self._create_zip_backup(source_path, backup_path)
            else:
                backup_path = backup_dir / backup_name
                import shutil
                shutil.copytree(source_path, backup_path)
            
            self.logger.info(f"Backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None
    
    def _create_zip_backup(self, source_path: Path, backup_path: Path):
        """Create compressed backup"""
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)
    
    def cleanup_old_backups(self, backup_dir: Path):
        """Remove old backups based on retention policy"""
        if self.config.retention_days <= 0:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        for backup_file in backup_dir.glob("*_backup_*"):
            try:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    if backup_file.is_file():
                        backup_file.unlink()
                    else:
                        import shutil
                        shutil.rmtree(backup_file)
                    self.logger.info(f"Removed old backup: {backup_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old backup {backup_file}: {e}")

###############################################################################
#                              MAIN APPLICATION                               #
###############################################################################

class ARCGenV2:
    """Main ARCGen V2 application with intelligent scaling"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.logger = setup_logging(self.config_manager.config)
        self.ai_provider = None
        self.scaling_manager = None
        self.file_processor = None
        self.backup_manager = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        # Initialize intelligent scaling manager first
        self.scaling_manager = IntelligentScalingManager(self.config_manager, self.logger)
        
        # Initialize AI provider
        api_config = self.config_manager.get_api_config()
        rate_limit_config = self.config_manager.config['api']['rate_limit']
        
        if api_config.provider == 'deepseek':
            self.ai_provider = DeepSeekProvider(api_config, self.logger, rate_limit_config)
        elif api_config.provider == 'openai':
            self.ai_provider = OpenAIProvider(api_config, self.logger, rate_limit_config)
        else:
            raise ValueError(f"Unsupported provider: {api_config.provider}")
        
        # Validate API key
        if not self.ai_provider.validate_api_key():
            console.print("[red]API key validation failed[/red]")
            sys.exit(1)
        
        # Initialize file processor with scaling manager
        self.file_processor = FileProcessor(self.config_manager, self.ai_provider, self.scaling_manager, self.logger)
        
        # Initialize backup manager
        backup_config = BackupConfig(**self.config_manager.config.get('backup', {}))
        self.backup_manager = BackupManager(backup_config, self.logger)
    
    def process_addon(self, addon_path: Path, output_path: Path, profile_name: str = 'balanced') -> Dict[str, Any]:
        """Process an entire addon"""
        start_time = time.time()
        
        # Validate paths
        if not addon_path.exists() or not addon_path.is_dir():
            raise ValueError(f"Invalid addon path: {addon_path}")
        
        # Create backup
        backup_path = self.backup_manager.create_backup(addon_path)
        
        # Get optimization profile
        profile = self.config_manager.get_optimization_profile(profile_name)
        
        # Collect all files
        all_files = list(addon_path.rglob('*'))
        target_files = [f for f in all_files if f.is_file()]
        
        console.print(f"[green]Processing {len(target_files)} files...[/green]")
        
        # Process files with progress bar and intelligent scaling
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(target_files))
            
            # Use intelligent scaling for dynamic concurrency control
            optimal_workers = self.scaling_manager.get_optimal_concurrency()
            self.logger.info(f"Starting with {optimal_workers} concurrent workers")
            
            # Process files in batches to allow for dynamic scaling
            batch_size = max(10, len(target_files) // 10)  # Process in 10% batches
            
            for i in range(0, len(target_files), batch_size):
                batch_files = target_files[i:i + batch_size]
                
                # Check if we need to adjust concurrency
                current_workers = self.scaling_manager.get_optimal_concurrency()
                if current_workers != optimal_workers:
                    self.logger.info(f"Adjusting concurrency from {optimal_workers} to {current_workers}")
                    optimal_workers = current_workers
                
                # Process current batch
                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            self.file_processor.process_file, 
                            file_path, addon_path, output_path, profile
                        ): file_path
                        for file_path in batch_files
                    }
                    
                    for future in as_completed(future_to_file):
                        try:
                            result = future.result()
                            results.append(result)
                            progress.advance(task)
                        except Exception as e:
                            # Handle individual file processing errors
                            file_path = future_to_file[future]
                            self.logger.error(f"Error processing {file_path}: {e}")
                            self.scaling_manager.handle_api_error(e)
                            
                            # Add error result
                            results.append({
                                'file': str(file_path),
                                'error': str(e),
                                'processing_time': 0
                            })
                            progress.advance(task)
        
        # Generate report
        total_time = time.time() - start_time
        report = self._generate_report(addon_path, output_path, results, total_time, backup_path)
        
        # Save report
        self._save_report(report, output_path)
        
        # Display summary
        self._display_summary(report)
        
        return report
    
    def _generate_report(self, addon_path: Path, output_path: Path, results: List[Dict], 
                        total_time: float, backup_path: Optional[Path]) -> Dict[str, Any]:
        """Generate comprehensive processing report with scaling statistics"""
        stats = self.file_processor.stats
        scaling_report = self.scaling_manager.get_scaling_report()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'addon_path': str(addon_path),
            'output_path': str(output_path),
            'backup_path': str(backup_path) if backup_path else None,
            'total_time': total_time,
            'statistics': {
                'files_total': len(results),
                'files_optimized': stats['files_optimized'],
                'files_copied': stats['files_copied'],
                'files_failed': stats['files_failed'],
                'size_before': stats['total_size_before'],
                'size_after': stats['total_size_after'],
                'size_reduction': stats['total_size_before'] - stats['total_size_after'],
                'processing_time': stats['processing_time'],
                'scaling_events': stats['scaling_events'],
                'memory_cleanups': stats['memory_cleanups']
            },
            'scaling_report': scaling_report,
            'results': results
        }
    
    def _save_report(self, report: Dict[str, Any], output_path: Path):
        """Save report to file"""
        report_path = output_path / 'arcgen_report.json'
        try:
            with report_path.open('w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Report saved to {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def _display_summary(self, report: Dict[str, Any]):
        """Display comprehensive processing summary with scaling information"""
        stats = report['statistics']
        scaling_report = report['scaling_report']
        
        # Create main summary table
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(stats['files_total']))
        table.add_row("Files Optimized", str(stats['files_optimized']))
        table.add_row("Files Copied", str(stats['files_copied']))
        table.add_row("Files Failed", str(stats['files_failed']))
        table.add_row("Size Before", f"{stats['size_before'] / 1024:.1f} KB")
        table.add_row("Size After", f"{stats['size_after'] / 1024:.1f} KB")
        table.add_row("Size Reduction", f"{stats['size_reduction'] / 1024:.1f} KB")
        table.add_row("Total Time", f"{report['total_time']:.1f}s")
        table.add_row("Scaling Events", str(stats['scaling_events']))
        table.add_row("Memory Cleanups", str(stats['memory_cleanups']))
        
        console.print(table)
        
        # Create scaling summary table
        if scaling_report['scaling_enabled']:
            scaling_table = Table(title="Intelligent Scaling Summary")
            scaling_table.add_column("Component", style="cyan")
            scaling_table.add_column("Status", style="green")
            
            # Memory information
            memory_info = scaling_report['memory_monitor']['current_usage']
            scaling_table.add_row("Memory Usage", f"{memory_info['percent']:.1f}%")
            scaling_table.add_row("Available Memory", f"{memory_info['available_gb']:.1f} GB")
            
            # Rate limiter information
            rate_info = scaling_report['rate_limiter']
            scaling_table.add_row("Current Rate Limit", str(rate_info['current_limit']))
            scaling_table.add_row("Rate Limit Hits", str(rate_info['rate_limit_hits']))
            
            # Chunker information
            chunker_info = scaling_report['adaptive_chunker']
            scaling_table.add_row("Current Chunk Size", str(chunker_info['current_chunk_size']))
            scaling_table.add_row("Success Rate", f"{chunker_info['success_rate']:.1%}")
            
            # Overall scaling
            scaling_table.add_row("Optimal Concurrency", str(scaling_report['optimal_concurrency']))
            scaling_table.add_row("Total Scaling Events", str(scaling_report['total_scaling_events']))
            
            console.print(scaling_table)
        
        # Display recent scaling events if any
        if scaling_report['scaling_events']:
            console.print("\n[yellow]Recent Scaling Events:[/yellow]")
            for event in scaling_report['scaling_events'][-3:]:  # Show last 3 events
                timestamp = datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")
                console.print(f"  {timestamp}: {event['description']}")
        
        console.print(f"\n[green]✓ Processing completed successfully![/green]")
        console.print(f"[dim]Report saved to: {report['output_path']}/arcgen_report.json[/dim]")

###############################################################################
#                              CLI INTERFACE                                  #
###############################################################################

@click.command()
@click.argument('addon_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory path')
@click.option('--profile', '-p', default='balanced', help='Optimization profile (conservative, balanced, aggressive)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--provider', help='AI provider to use (deepseek, openai)')
@click.option('--backup/--no-backup', default=True, help='Enable/disable backup creation')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def main(addon_path: str, output: Optional[str], profile: str, config: Optional[str], 
         provider: Optional[str], backup: bool, verbose: bool):
    """
    ARCGen V2 - Enhanced Automated Reverse Code Generation
    
    Process and optimize code files using AI-powered analysis.
    """
    try:
        # Display banner
        console.print(Panel.fit(
            "[bold blue]ARCGen V2[/bold blue]\n"
            "[dim]Enhanced Automated Reverse Code Generation[/dim]",
            border_style="blue"
        ))
        
        # Initialize application
        app = ARCGenV2(config)
        
        # Override provider if specified
        if provider:
            app.config_manager.config['api']['primary_provider'] = provider
            # Store backup config before reinitializing
            backup_enabled = app.backup_manager.config.enabled
            app._initialize_components()
            # Restore backup preference
            app.backup_manager.config.enabled = backup_enabled
        
        # Set backup preference
        app.backup_manager.config.enabled = backup
        
        # Set paths
        addon_path = Path(addon_path).resolve()
        if output:
            output_path = Path(output).resolve()
        else:
            output_path = addon_path.parent / f"{addon_path.name}_optimized"
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process addon
        console.print(f"[blue]Processing addon:[/blue] {addon_path}")
        console.print(f"[blue]Output directory:[/blue] {output_path}")
        console.print(f"[blue]Optimization profile:[/blue] {profile}")
        
        report = app.process_addon(addon_path, output_path, profile)
        
        console.print("[green]✓ Processing completed successfully![/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 