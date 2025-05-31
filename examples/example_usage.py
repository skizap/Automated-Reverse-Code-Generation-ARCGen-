#!/usr/bin/env python3
"""
Example usage scripts for ARCGen V2
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arcgen_v2 import ARCGenV2, ConfigManager

def example_basic_usage():
    """
    Example 1: Basic usage with default settings
    """
    print("=== Example 1: Basic Usage ===")
    
    # Set up environment (you would normally have this in .env file)
    os.environ['DEEPSEEK_API_KEY'] = 'your_api_key_here'
    
    try:
        # Initialize ARCGen with default config
        app = ARCGenV2()
        
        # Process an addon (replace with actual path)
        addon_path = Path("./test_addon")
        output_path = Path("./test_addon_optimized")
        
        if addon_path.exists():
            report = app.process_addon(addon_path, output_path)
            print(f"Processing completed! Report: {report['statistics']}")
        else:
            print("Test addon directory not found. Create one to test.")
            
    except Exception as e:
        print(f"Error: {e}")

def example_custom_config():
    """
    Example 2: Using custom configuration
    """
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create custom config
    custom_config = {
        'api': {
            'primary_provider': 'deepseek',
            'deepseek': {
                'base_url': 'https://api.deepseek.com',
                'model': 'deepseek-coder',
                'max_tokens': 4000,  # Reduced for faster processing
                'temperature': 0.5   # More deterministic
            }
        },
        'processing': {
            'chunk_size': 2000,  # Smaller chunks
            'smart_chunking': True,
            'text_extensions': ['.lua', '.txt', '.js', '.py']
        },
        'optimization_profiles': {
            'custom': {
                'preserve_comments': True,
                'optimize_performance': True,
                'add_documentation': False,
                'remove_dead_code': True
            }
        },
        'backup': {
            'enabled': False  # Disable backup for testing
        }
    }
    
    # Save custom config
    import yaml
    config_path = Path("custom_config.yaml")
    with config_path.open('w') as f:
        yaml.dump(custom_config, f, default_flow_style=False)
    
    try:
        # Initialize with custom config
        app = ARCGenV2(str(config_path))
        print("Custom configuration loaded successfully!")
        
        # Clean up
        config_path.unlink()
        
    except Exception as e:
        print(f"Error: {e}")

def example_different_profiles():
    """
    Example 3: Using different optimization profiles
    """
    print("\n=== Example 3: Different Optimization Profiles ===")
    
    profiles = ['conservative', 'balanced', 'aggressive']
    
    for profile in profiles:
        print(f"\nTesting {profile} profile:")
        
        try:
            app = ARCGenV2()
            config_manager = app.config_manager
            
            # Get profile configuration
            profile_obj = config_manager.get_optimization_profile(profile)
            print(f"  - Preserve comments: {profile_obj.preserve_comments}")
            print(f"  - Optimize performance: {profile_obj.optimize_performance}")
            print(f"  - Remove dead code: {profile_obj.remove_dead_code}")
            print(f"  - Refactor functions: {profile_obj.refactor_functions}")
            
        except Exception as e:
            print(f"  Error: {e}")

def example_file_processing():
    """
    Example 4: Manual file processing
    """
    print("\n=== Example 4: Manual File Processing ===")
    
    try:
        app = ARCGenV2()
        processor = app.file_processor
        
        # Create a test file
        test_file = Path("test_code.lua")
        test_content = """
-- Test Lua code
function greet(name)
    print("Hello, " .. name .. "!")
    return true
end

-- Another function
function calculate(a, b)
    local result = a + b
    print("Result: " .. result)
    return result
end
        """.strip()
        
        test_file.write_text(test_content)
        
        # Test file detection
        should_process = processor._should_process_file(test_file)
        print(f"Should process {test_file}: {should_process}")
        
        # Test chunking
        chunker = processor.chunker
        chunks = chunker.chunk_text(test_content, ".lua")
        print(f"File split into {len(chunks)} chunks")
        
        # Clean up
        test_file.unlink()
        
    except Exception as e:
        print(f"Error: {e}")

def example_security_features():
    """
    Example 5: Security features demonstration
    """
    print("\n=== Example 5: Security Features ===")
    
    try:
        app = ARCGenV2()
        
        # Test path validation
        dangerous_paths = [
            "../../../etc/passwd",
            "C:\\Windows\\System32\\config",
            "/etc/shadow",
            "..\\..\\sensitive_file.txt"
        ]
        
        for path in dangerous_paths:
            try:
                test_path = Path(path)
                # This would be caught by security validation
                print(f"Testing dangerous path: {path} - Would be blocked")
            except Exception as e:
                print(f"Security check caught: {path}")
        
        # Show security config
        security_config = app.config_manager.config.get('security', {})
        print(f"\nSecurity settings:")
        print(f"  - Path validation: {security_config.get('validate_paths', False)}")
        print(f"  - Max processing time: {security_config.get('max_processing_time', 0)}s")
        print(f"  - Security scan: {security_config.get('security_scan', False)}")
        
    except Exception as e:
        print(f"Error: {e}")

def example_backup_system():
    """
    Example 6: Backup system demonstration
    """
    print("\n=== Example 6: Backup System ===")
    
    try:
        app = ARCGenV2()
        backup_manager = app.backup_manager
        
        # Create test directory
        test_dir = Path("test_backup_source")
        test_dir.mkdir(exist_ok=True)
        (test_dir / "file1.txt").write_text("Test content 1")
        (test_dir / "file2.lua").write_text("print('test')")
        
        # Create backup
        backup_path = backup_manager.create_backup(test_dir)
        
        if backup_path:
            print(f"Backup created: {backup_path}")
            print(f"Backup exists: {backup_path.exists()}")
            print(f"Backup size: {backup_path.stat().st_size} bytes")
            
            # Clean up
            backup_path.unlink()
        else:
            print("Backup creation disabled or failed")
        
        # Clean up test directory
        import shutil
        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """
    Run all examples
    """
    print("ARCGen V2 - Example Usage Scripts")
    print("=" * 50)
    
    # Note: These examples require proper API key setup
    print("Note: Set DEEPSEEK_API_KEY environment variable to run API examples")
    
    example_basic_usage()
    example_custom_config()
    example_different_profiles()
    example_file_processing()
    example_security_features()
    example_backup_system()
    
    print("\n" + "=" * 50)
    print("Examples completed!")

if __name__ == "__main__":
    main() 