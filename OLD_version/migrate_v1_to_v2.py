#!/usr/bin/env python3
"""
Migration script from ARCGen V1 to V2
Helps users upgrade their existing setup and configuration
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def detect_v1_installation() -> Optional[Path]:
    """Detect existing ARCGen V1 installation"""
    current_dir = Path.cwd()
    
    # Look for ARCGen.py (V1 main file)
    v1_file = current_dir / "ARCGen.py"
    if v1_file.exists():
        return current_dir
    
    # Check parent directories
    for parent in current_dir.parents:
        v1_file = parent / "ARCGen.py"
        if v1_file.exists():
            return parent
    
    return None

def extract_v1_config(v1_path: Path) -> Dict[str, Any]:
    """Extract configuration from V1 ARCGen.py file"""
    config = {}
    
    try:
        with open(v1_path / "ARCGen.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract hardcoded values from V1
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Extract API configuration
            if line.startswith('DEEPSEEK_API_URL'):
                url = line.split('=')[1].strip().strip('"\'')
                config['deepseek_url'] = url
            elif line.startswith('DEEPSEEK_MODEL_ID'):
                model = line.split('=')[1].strip().strip('"\'')
                config['deepseek_model'] = model
            elif line.startswith('MAX_TOKENS'):
                tokens = int(line.split('=')[1].strip())
                config['max_tokens'] = tokens
            elif line.startswith('CHUNK_SIZE'):
                chunk_size = int(line.split('=')[1].strip())
                config['chunk_size'] = chunk_size
            elif line.startswith('MAX_RETRIES'):
                retries = int(line.split('=')[1].strip())
                config['max_retries'] = retries
    
    except Exception as e:
        print(f"Warning: Could not extract all V1 configuration: {e}")
    
    return config

def create_v2_config(v1_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create V2 configuration based on V1 settings"""
    
    # Default V2 configuration
    v2_config = {
        'api': {
            'primary_provider': 'deepseek',
            'deepseek': {
                'base_url': v1_config.get('deepseek_url', 'https://api.deepseek.com'),
                'model': v1_config.get('deepseek_model', 'deepseek-coder'),
                'max_tokens': v1_config.get('max_tokens', 8000),
                'temperature': 0.7,
                'top_p': 1.0
            },
            'rate_limit': {
                'requests_per_minute': 60,
                'concurrent_requests': 4
            }
        },
        'processing': {
            'chunk_size': v1_config.get('chunk_size', 3000),
            'smart_chunking': True,
            'max_file_size': 10,
            'text_extensions': ['.lua', '.txt', '.cfg', '.json', '.vmt', '.vmf'],
            'binary_extensions': ['.vtf', '.vvd', '.mdl', '.phy', '.wav', '.mp3', '.ogg']
        },
        'optimization_profiles': {
            'conservative': {
                'preserve_comments': True,
                'preserve_formatting': True,
                'optimize_performance': False,
                'add_documentation': True
            },
            'balanced': {
                'preserve_comments': True,
                'preserve_formatting': False,
                'optimize_performance': True,
                'add_documentation': True,
                'remove_dead_code': True
            },
            'aggressive': {
                'preserve_comments': False,
                'preserve_formatting': False,
                'optimize_performance': True,
                'add_documentation': False,
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
            'log_file': 'arcgen.log',
            'rotate_logs': True,
            'max_log_size': 10,
            'backup_count': 5
        },
        'output': {
            'output_suffix': '_optimized',
            'generate_report': True,
            'report_format': 'html',
            'include_stats': True
        },
        'security': {
            'validate_paths': True,
            'max_processing_time': 300,
            'security_scan': True
        },
        'performance': {
            'enable_caching': True,
            'cache_dir': '.arcgen_cache',
            'cache_expiration': 24,
            'memory_limit': 1024
        }
    }
    
    return v2_config

def backup_v1_files(v1_path: Path) -> Path:
    """Create backup of V1 files"""
    backup_dir = v1_path / "v1_backup"
    backup_dir.mkdir(exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "ARCGen.py",
        "README.md",
        "LICENSE"
    ]
    
    for file_name in files_to_backup:
        source_file = v1_path / file_name
        if source_file.exists():
            shutil.copy2(source_file, backup_dir / file_name)
            print(f"Backed up: {file_name}")
    
    return backup_dir

def install_v2_files(target_path: Path):
    """Install V2 files to target directory"""
    current_dir = Path(__file__).parent
    
    # Files to copy
    v2_files = [
        "arcgen_v2.py",
        "requirements.txt",
        "env.example",
        "setup.py",
        "README_V2.md"
    ]
    
    for file_name in v2_files:
        source_file = current_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_path / file_name)
            print(f"Installed: {file_name}")
    
    # Copy directories
    dirs_to_copy = ["tests", "examples"]
    for dir_name in dirs_to_copy:
        source_dir = current_dir / dir_name
        target_dir = target_path / dir_name
        if source_dir.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)
            print(f"Installed directory: {dir_name}")

def migrate_api_key(v1_path: Path, target_path: Path):
    """Migrate API key from V1 to V2 environment file"""
    v1_file = v1_path / "ARCGen.py"
    env_file = target_path / ".env"
    
    api_key = None
    
    # Try to extract API key from V1 file
    try:
        with open(v1_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line in lines:
            if 'DEEPSEEK_API_KEY' in line and '=' in line:
                # Extract the key value
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key_value = parts[1].strip().strip('"\'')
                    if key_value and key_value != "API-KEY-HERE":
                        api_key = key_value
                        break
    
    except Exception as e:
        print(f"Warning: Could not extract API key from V1: {e}")
    
    # Create .env file
    env_content = f"""# ARCGen V2 Environment Variables
# Migrated from ARCGen V1

# DeepSeek API Configuration
DEEPSEEK_API_KEY={api_key or 'your_deepseek_api_key_here'}

# OpenAI API Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Debug mode
DEBUG=false

# Log level
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    if api_key:
        print("✓ API key migrated successfully")
    else:
        print("⚠ API key not found in V1, please update .env file manually")

def create_migration_report(v1_path: Path, target_path: Path, v1_config: Dict[str, Any]):
    """Create migration report"""
    report_content = f"""# ARCGen V1 to V2 Migration Report

## Migration Summary
- **Source (V1)**: {v1_path}
- **Target (V2)**: {target_path}
- **Migration Date**: {os.popen('date').read().strip()}

## V1 Configuration Detected
- **API URL**: {v1_config.get('deepseek_url', 'Not found')}
- **Model**: {v1_config.get('deepseek_model', 'Not found')}
- **Max Tokens**: {v1_config.get('max_tokens', 'Not found')}
- **Chunk Size**: {v1_config.get('chunk_size', 'Not found')}

## Files Migrated
- ✓ ARCGen V2 main application
- ✓ Configuration system
- ✓ Environment variables
- ✓ Requirements and dependencies
- ✓ Documentation
- ✓ Tests and examples

## New Features in V2
- **Multi-provider AI support** (DeepSeek, OpenAI, Claude)
- **Smart context-aware chunking**
- **Configuration management with YAML**
- **Secure environment variable handling**
- **Progress tracking and rich console output**
- **Backup and rollback functionality**
- **Comprehensive logging and reporting**
- **Security scanning and validation**

## Next Steps
1. Install dependencies: `pip install -r requirements.txt`
2. Update API key in `.env` file if needed
3. Review and customize `config.yaml`
4. Test with: `python arcgen_v2.py --help`
5. Run examples: `python examples/example_usage.py`

## Breaking Changes
- Command-line interface has changed (now uses Click)
- Configuration is now in YAML format
- API key must be in environment variable
- Output format and reporting have changed

## Support
- Check README_V2.md for detailed documentation
- Run tests with: `python -m pytest tests/`
- See examples in examples/ directory
"""
    
    report_file = target_path / "MIGRATION_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Migration report created: {report_file}")

def main():
    """Main migration function"""
    print("ARCGen V1 to V2 Migration Tool")
    print("=" * 50)
    
    # Detect V1 installation
    print("1. Detecting ARCGen V1 installation...")
    v1_path = detect_v1_installation()
    
    if not v1_path:
        print("❌ ARCGen V1 installation not found in current directory or parents")
        print("Please run this script from your ARCGen V1 directory")
        sys.exit(1)
    
    print(f"✓ Found ARCGen V1 at: {v1_path}")
    
    # Extract V1 configuration
    print("\n2. Extracting V1 configuration...")
    v1_config = extract_v1_config(v1_path)
    print(f"✓ Extracted configuration: {len(v1_config)} settings")
    
    # Ask user for target directory
    print("\n3. Setting up target directory...")
    target_path = v1_path  # Install in same directory by default
    
    response = input(f"Install V2 in current directory ({target_path})? [Y/n]: ").strip().lower()
    if response in ['n', 'no']:
        target_input = input("Enter target directory path: ").strip()
        target_path = Path(target_input).resolve()
    
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Target directory: {target_path}")
    
    # Create backup
    print("\n4. Creating backup of V1 files...")
    backup_dir = backup_v1_files(v1_path)
    print(f"✓ V1 files backed up to: {backup_dir}")
    
    # Install V2 files
    print("\n5. Installing ARCGen V2 files...")
    install_v2_files(target_path)
    print("✓ V2 files installed")
    
    # Create V2 configuration
    print("\n6. Creating V2 configuration...")
    v2_config = create_v2_config(v1_config)
    config_file = target_path / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(v2_config, f, default_flow_style=False, indent=2)
    print(f"✓ Configuration created: {config_file}")
    
    # Migrate API key
    print("\n7. Migrating API key...")
    migrate_api_key(v1_path, target_path)
    
    # Create migration report
    print("\n8. Creating migration report...")
    create_migration_report(v1_path, target_path, v1_config)
    
    # Final instructions
    print("\n" + "=" * 50)
    print("✅ Migration completed successfully!")
    print("\nNext steps:")
    print("1. cd", target_path)
    print("2. pip install -r requirements.txt")
    print("3. Update .env file with your API key if needed")
    print("4. python arcgen_v2.py --help")
    print("\nSee MIGRATION_REPORT.md for detailed information.")
    print("=" * 50)

if __name__ == "__main__":
    main() 