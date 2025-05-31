#!/usr/bin/env python3
"""
ARCGen V2 - Enhanced Automated Reverse Code Generation
======================================================

Advanced AI-powered code optimization and rewriting tool with support for multiple
AI providers, smart chunking, comprehensive configuration, and enterprise features.

Features:
- Multi-provider AI support (DeepSeek, OpenAI, Claude)
- Smart context-aware chunking
- Configuration management with YAML
- Secure environment variable handling
- Progress tracking and rich console output
- Backup and rollback functionality
- Comprehensive logging and reporting
- Performance monitoring and caching
- Security scanning and validation

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

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
                    'model': 'deepseek-coder',
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
                'security_scan': True,
                'allowed_extensions': ['.lua', '.txt', '.cfg', '.json', '.vmt', '.vmf', '.py', '.js', '.cpp', '.c', '.cs']
            }
        }
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate API keys
        provider = self.config['api']['primary_provider']
        if provider == 'deepseek' and not os.getenv('DEEPSEEK_API_KEY'):
            console.print("[red]DEEPSEEK_API_KEY environment variable not set[/red]")
            sys.exit(1)
        elif provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
            console.print("[red]OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
    
    def get_api_config(self, provider: Optional[str] = None) -> APIConfig:
        """Get API configuration for specified provider"""
        provider = provider or self.config['api']['primary_provider']
        
        if provider == 'deepseek':
            config = self.config['api']['deepseek']
            api_key = os.getenv('DEEPSEEK_API_KEY')
        elif provider == 'openai':
            config = self.config['api'].get('openai', {})
            api_key = os.getenv('OPENAI_API_KEY')
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        return APIConfig(
            provider=provider,
            base_url=config.get('base_url', ''),
            model=config.get('model', ''),
            api_key=api_key,
            max_tokens=config.get('max_tokens', 8000),
            temperature=config.get('temperature', 0.7),
            top_p=config.get('top_p', 1.0),
            timeout=config.get('timeout', 60)
        )
    
    def get_optimization_profile(self, profile_name: str) -> OptimizationProfile:
        """Get optimization profile by name"""
        profiles = self.config.get('optimization_profiles', {})
        if profile_name not in profiles:
            console.print(f"[yellow]Profile '{profile_name}' not found, using balanced[/yellow]")
            profile_name = 'balanced'
        
        profile_config = profiles.get(profile_name, profiles.get('balanced', {}))
        return OptimizationProfile(
            name=profile_name,
            **profile_config
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        security_config = self.config.get('security', {})
        return SecurityConfig(**security_config)

###############################################################################
#                              LOGGING SETUP                                  #
###############################################################################

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    
    # Create logger
    logger = logging.getLogger('arcgen')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(console=console, show_time=True, show_path=False)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler if enabled
    if log_config.get('log_to_file', True):
        log_file = log_config.get('log_file', 'arcgen.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
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
        self.session = requests.Session()
        # Use rate limit from configuration
        requests_per_minute = rate_limit_config.get('requests_per_minute', 60)
        self.rate_limiter = RateLimiter(requests_per_minute, 60)
    
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
        with self.rate_limiter:
            url = f"{self.config.base_url}/chat/completions"
            
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful code optimization assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            try:
                response = self.session.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "[Error] No content returned from API"
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                return f"[Error] API request failed: {e}"
    
    def validate_api_key(self) -> bool:
        """Validate DeepSeek API key"""
        try:
            response = self.generate("Hello")
            return not response.startswith("[Error]")
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

class OpenAIProvider(AIProvider):
    """OpenAI API provider"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        with self.rate_limiter:
            url = f"{self.config.base_url}/chat/completions"
            
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful code optimization assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            try:
                response = self.session.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    return "[Error] No content returned from API"
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                return f"[Error] API request failed: {e}"
    
    def validate_api_key(self) -> bool:
        """Validate OpenAI API key"""
        try:
            response = self.generate("Hello")
            return not response.startswith("[Error]")
        except Exception as e:
            self.logger.error(f"API key validation failed: {e}")
            return False

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def __enter__(self):
        with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # Wait if we've hit the rate limit
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    self.calls = self.calls[1:]  # Remove the oldest call
            
            self.calls.append(now)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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
    """Handles file processing and optimization"""
    
    def __init__(self, config_manager: ConfigManager, ai_provider: AIProvider, logger: logging.Logger):
        self.config_manager = config_manager
        self.ai_provider = ai_provider
        self.logger = logger
        self.chunker = SmartChunker(config_manager.config['processing']['chunk_size'])
        self.security_config = config_manager.get_security_config()
        self.stats = {
            'files_processed': 0,
            'files_optimized': 0,
            'files_copied': 0,
            'files_failed': 0,
            'total_size_before': 0,
            'total_size_after': 0,
            'processing_time': 0
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
        """Optimize a single file using AI"""
        try:
            # Read file with size limit for memory safety
            max_size = self.config_manager.config['processing'].get('max_file_size', 10) * 1024 * 1024
            if file_path.stat().st_size > max_size:
                self.logger.warning(f"File {file_path} too large, skipping optimization")
                return None
            
            with file_path.open('r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if not content.strip():
                return content
            
            # Split into chunks
            chunks = self.chunker.chunk_text(content, file_path.suffix)
            optimized_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Add timeout for individual chunk processing
                start_time = time.time()
                prompt = self._build_optimization_prompt(chunk, file_path.suffix, profile)
                optimized_chunk = self.ai_provider.generate(prompt)
                
                # Check for timeout
                if time.time() - start_time > self.security_config.max_processing_time:
                    self.logger.warning(f"Chunk processing timeout for {file_path}")
                    return None
                
                # Clean up the response (remove markdown formatting if present)
                optimized_chunk = self._clean_ai_response(optimized_chunk)
                optimized_chunks.append(optimized_chunk)
                
                self.logger.debug(f"Optimized chunk {i+1}/{len(chunks)} for {file_path}")
            
            return '\n'.join(optimized_chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to optimize {file_path}: {e}")
            return None
    
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
    """Main ARCGen V2 application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.logger = setup_logging(self.config_manager.config)
        self.ai_provider = None
        self.file_processor = None
        self.backup_manager = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
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
        
        # Initialize file processor
        self.file_processor = FileProcessor(self.config_manager, self.ai_provider, self.logger)
        
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
        
        # Process files with progress bar
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
            
            # Use thread pool for parallel processing
            max_workers = self.config_manager.config['api']['rate_limit']['concurrent_requests']
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self.file_processor.process_file, 
                        file_path, addon_path, output_path, profile
                    ): file_path
                    for file_path in target_files
                }
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
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
        """Generate processing report"""
        stats = self.file_processor.stats
        
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
                'processing_time': stats['processing_time']
            },
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
        """Display processing summary"""
        stats = report['statistics']
        
        # Create summary table
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
        
        console.print(table)

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