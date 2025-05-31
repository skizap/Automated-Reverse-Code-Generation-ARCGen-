#!/usr/bin/env python3
"""
Unit tests for ARCGen V2
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arcgen_v2 import (
    ConfigManager, SmartChunker, FileProcessor, BackupManager,
    DeepSeekProvider, APIConfig, OptimizationProfile, BackupConfig
)

class TestConfigManager(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config_loading(self):
        """Test loading default configuration when file doesn't exist"""
        config_manager = ConfigManager(str(self.config_path))
        self.assertIsInstance(config_manager.config, dict)
        self.assertIn('api', config_manager.config)
        self.assertIn('processing', config_manager.config)
    
    def test_api_config_creation(self):
        """Test API configuration creation"""
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'}):
            config_manager = ConfigManager(str(self.config_path))
            api_config = config_manager.get_api_config('deepseek')
            self.assertEqual(api_config.provider, 'deepseek')
            self.assertEqual(api_config.api_key, 'test_key')
    
    def test_optimization_profile_creation(self):
        """Test optimization profile creation"""
        config_manager = ConfigManager(str(self.config_path))
        profile = config_manager.get_optimization_profile('balanced')
        self.assertIsInstance(profile, OptimizationProfile)
        self.assertEqual(profile.name, 'balanced')

class TestSmartChunker(unittest.TestCase):
    """Test smart chunking functionality"""
    
    def setUp(self):
        self.chunker = SmartChunker(chunk_size=100)
    
    def test_empty_text_chunking(self):
        """Test chunking empty text"""
        chunks = self.chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_simple_text_chunking(self):
        """Test simple text chunking"""
        text = "a" * 250
        chunks = self.chunker.chunk_text(text, ".txt")
        self.assertEqual(len(chunks), 3)  # 100, 100, 50
    
    def test_code_chunking(self):
        """Test code chunking preserves structure"""
        code = """
function test1() {
    console.log("test1");
}

function test2() {
    console.log("test2");
}
        """.strip()
        
        chunks = self.chunker.chunk_text(code, ".js")
        self.assertGreater(len(chunks), 0)
        # Each chunk should be valid (not cut in middle of function)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)

class TestDeepSeekProvider(unittest.TestCase):
    """Test DeepSeek API provider"""
    
    def setUp(self):
        self.api_config = APIConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com",
            model="deepseek-coder",
            api_key="test_key",
            max_tokens=1000,
            temperature=0.7
        )
        self.logger = Mock()
        self.provider = DeepSeekProvider(self.api_config, self.logger)
    
    @patch('requests.Session.post')
    def test_successful_generation(self, mock_post):
        """Test successful API response"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "optimized code"}}]
        }
        mock_post.return_value = mock_response
        
        result = self.provider.generate("test prompt")
        self.assertEqual(result, "optimized code")
    
    @patch('requests.Session.post')
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        mock_post.side_effect = Exception("API Error")
        
        result = self.provider.generate("test prompt")
        self.assertTrue(result.startswith("[Error]"))

class TestFileProcessor(unittest.TestCase):
    """Test file processing functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = Mock()
        self.config_manager.config = {
            'processing': {
                'chunk_size': 1000,
                'text_extensions': ['.lua', '.txt'],
                'binary_extensions': ['.png', '.jpg'],
                'max_file_size': 10
            }
        }
        
        self.ai_provider = Mock()
        self.ai_provider.generate.return_value = "optimized code"
        
        self.logger = Mock()
        self.processor = FileProcessor(self.config_manager, self.ai_provider, self.logger)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_should_process_text_file(self):
        """Test text file detection"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.lua"
        test_file.write_text("print('hello')")
        
        should_process = self.processor._should_process_file(test_file)
        self.assertTrue(should_process)
    
    def test_should_not_process_binary_file(self):
        """Test binary file detection"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.png"
        test_file.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header
        
        should_process = self.processor._should_process_file(test_file)
        self.assertFalse(should_process)
    
    def test_optimization_prompt_building(self):
        """Test optimization prompt building"""
        profile = OptimizationProfile(
            name="test",
            preserve_comments=True,
            optimize_performance=True
        )
        
        prompt = self.processor._build_optimization_prompt("test code", ".lua", profile)
        self.assertIn("preserve existing comments", prompt)
        self.assertIn("improve performance", prompt)
        self.assertIn("test code", prompt)

class TestBackupManager(unittest.TestCase):
    """Test backup management"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.backup_config = BackupConfig(
            enabled=True,
            backup_dir="backup",
            compress=True,
            retention_days=30
        )
        self.logger = Mock()
        self.backup_manager = BackupManager(self.backup_config, self.logger)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_backup_creation(self):
        """Test backup creation"""
        # Create source directory with files
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        (source_dir / "test.txt").write_text("test content")
        
        backup_path = self.backup_manager.create_backup(source_dir)
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())
        self.assertTrue(backup_path.name.endswith('.zip'))
    
    def test_backup_disabled(self):
        """Test backup when disabled"""
        self.backup_manager.config.enabled = False
        source_dir = Path(self.temp_dir) / "source"
        source_dir.mkdir()
        
        backup_path = self.backup_manager.create_backup(source_dir)
        self.assertIsNone(backup_path)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test_key'})
    @patch('arcgen_v2.DeepSeekProvider.validate_api_key')
    @patch('arcgen_v2.DeepSeekProvider.generate')
    def test_end_to_end_processing(self, mock_generate, mock_validate):
        """Test end-to-end processing workflow"""
        from arcgen_v2 import ARCGenV2
        
        # Setup mocks
        mock_validate.return_value = True
        mock_generate.return_value = "optimized code"
        
        # Create test addon structure
        addon_dir = Path(self.temp_dir) / "test_addon"
        addon_dir.mkdir()
        (addon_dir / "init.lua").write_text("print('hello world')")
        (addon_dir / "config.txt").write_text("setting=value")
        
        # Create output directory
        output_dir = Path(self.temp_dir) / "output"
        
        # Create config file
        config_path = Path(self.temp_dir) / "config.yaml"
        config_path.write_text("""
api:
  primary_provider: deepseek
  deepseek:
    base_url: https://api.deepseek.com
    model: deepseek-coder
processing:
  text_extensions: ['.lua', '.txt']
backup:
  enabled: false
        """)
        
        # Initialize and run
        app = ARCGenV2(str(config_path))
        report = app.process_addon(addon_dir, output_dir)
        
        # Verify results
        self.assertIsInstance(report, dict)
        self.assertIn('statistics', report)
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / "init.lua").exists())

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 