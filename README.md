# ARCGen V2 - Enhanced Automated Reverse Code Generation

ARCGen V2 is a powerful, enterprise-grade AI-powered code optimization and rewriting tool designed for code analysis and optimization. It provides advanced features for analyzing, optimizing, and transforming code while maintaining security best practices.

## 🚀 Key Features

### Core Capabilities
- **Multi-Provider AI Support**: DeepSeek, OpenAI, Claude integration
- **Smart Context-Aware Chunking**: Preserves function/class boundaries
- **Advanced Configuration Management**: YAML-based configuration with profiles
- **Secure Environment Handling**: Environment variables for API keys
- **Enterprise-Grade Logging**: Structured logging with multiple levels
- **Comprehensive Backup System**: Automatic backups with compression and retention

### Security & Compliance
- **Path Validation**: Prevents directory traversal attacks
- **Security Scanning**: Detects potentially malicious code patterns
- **Rate Limiting**: Built-in API rate limiting and throttling
- **Audit Trail**: Complete processing logs and reports
- **Timeout Protection**: Prevents infinite processing loops

### Performance & Reliability
- **Parallel Processing**: Multi-threaded file processing
- **Progress Tracking**: Real-time progress bars and ETA
- **Error Recovery**: Graceful error handling and fallback mechanisms
- **Caching System**: Optional response caching for efficiency
- **Memory Management**: Configurable memory limits

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows 10/11, Linux, or macOS
- Minimum 4GB RAM (8GB recommended)
- 1GB free disk space

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `requests>=2.31.0` - HTTP client for API calls
- `pyyaml>=6.0.1` - YAML configuration parsing
- `python-dotenv>=1.0.0` - Environment variable management
- `rich>=13.7.0` - Rich console output and progress bars
- `click>=8.1.7` - Command-line interface framework
- `tqdm>=4.66.0` - Progress bars
- `colorama>=0.4.6` - Cross-platform colored terminal text

## 🛠️ Installation

### Quick Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ARCGen-V2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Set up configuration**:
   ```bash
   # config.yaml is created automatically with defaults
   # Customize as needed
   ```

### API Key Setup

#### DeepSeek API
1. Sign up at [DeepSeek Platform](https://platform.deepseek.com)
2. Generate an API key
3. Add to `.env`: `DEEPSEEK_API_KEY=your_key_here`

#### OpenAI API (Optional)
1. Sign up at [OpenAI Platform](https://platform.openai.com)
2. Generate an API key
3. Add to `.env`: `OPENAI_API_KEY=your_key_here`

## 🎯 Usage

### Basic Usage
```bash
# Process an addon with default settings
python arcgen_v2.py /path/to/addon

# Specify output directory
python arcgen_v2.py /path/to/addon --output /path/to/output

# Use specific optimization profile
python arcgen_v2.py /path/to/addon --profile aggressive

# Enable verbose logging
python arcgen_v2.py /path/to/addon --verbose
```

### Advanced Usage
```bash
# Use custom configuration file
python arcgen_v2.py /path/to/addon --config custom_config.yaml

# Use different AI provider
python arcgen_v2.py /path/to/addon --provider openai

# Disable backup creation
python arcgen_v2.py /path/to/addon --no-backup

# Combine multiple options
python arcgen_v2.py /path/to/addon \
  --output ./optimized \
  --profile balanced \
  --provider deepseek \
  --verbose
```

### Command-Line Options
- `addon_path`: Path to the addon directory to process (required)
- `--output, -o`: Output directory path
- `--profile, -p`: Optimization profile (conservative, balanced, aggressive)
- `--config, -c`: Custom configuration file path
- `--provider`: AI provider to use (deepseek, openai)
- `--backup/--no-backup`: Enable/disable backup creation
- `--verbose, -v`: Enable verbose output

## ⚙️ Configuration

### Configuration File Structure
The `config.yaml` file contains all configuration options:

```yaml
# API Configuration
api:
  primary_provider: "deepseek"
  deepseek:
    base_url: "https://api.deepseek.com"
    model: "deepseek-coder"
    max_tokens: 8000
    temperature: 0.7

# Processing Configuration
processing:
  chunk_size: 3000
  smart_chunking: true
  max_file_size: 10  # MB
  text_extensions:
    - ".lua"
    - ".txt"
    - ".cfg"
    - ".json"

# Optimization Profiles
optimization_profiles:
  conservative:
    preserve_comments: true
    preserve_formatting: true
    optimize_performance: false
  
  balanced:
    preserve_comments: true
    optimize_performance: true
    remove_dead_code: true
  
  aggressive:
    preserve_comments: false
    optimize_performance: true
    refactor_functions: true

# Backup Configuration
backup:
  enabled: true
  backup_dir: "backup"
  compress: true
  retention_days: 30

# Security Configuration
security:
  validate_paths: true
  max_processing_time: 300
  security_scan: true
```

### Optimization Profiles

#### Conservative Profile
- **Purpose**: Minimal changes, focus on readability
- **Features**: Preserves comments and formatting, adds documentation
- **Use Case**: Legacy code that needs minimal modification

#### Balanced Profile (Default)
- **Purpose**: Moderate optimization with safety
- **Features**: Performance improvements while preserving structure
- **Use Case**: Most general-purpose optimization tasks

#### Aggressive Profile
- **Purpose**: Maximum optimization and refactoring
- **Features**: Extensive code restructuring and optimization
- **Use Case**: Complete code overhaul and modernization

## 📊 Output and Reporting

### Processing Report
ARCGen V2 generates comprehensive reports in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "addon_path": "/path/to/addon",
  "output_path": "/path/to/output",
  "backup_path": "/path/to/backup.zip",
  "total_time": 45.2,
  "statistics": {
    "files_total": 150,
    "files_optimized": 120,
    "files_copied": 25,
    "files_failed": 5,
    "size_before": 2048576,
    "size_after": 1843200,
    "size_reduction": 205376
  }
}
```

### Console Output
Rich console output provides:
- Real-time progress bars
- Color-coded status messages
- Processing statistics table
- Error and warning highlights

## 🚀 Intelligent Scaling System

ARCGen V2 features an advanced intelligent scaling system that automatically adapts to various constraints and optimizes performance in real-time.

### Adaptive Rate Limiting
- **Exponential Backoff**: Automatically handles rate limit errors with intelligent backoff strategies
- **Dynamic Adjustment**: Increases or decreases request rates based on API response patterns
- **Concurrent Request Optimization**: Dynamically adjusts the number of concurrent requests
- **Success Rate Monitoring**: Tracks API success rates and adjusts accordingly

### Memory Management
- **Real-time Monitoring**: Continuously monitors system memory usage
- **Automatic Cleanup**: Triggers garbage collection when memory pressure is detected
- **Adaptive Scaling**: Reduces processing intensity when memory usage exceeds thresholds
- **Memory-based Concurrency**: Adjusts concurrent operations based on available memory

### Adaptive Chunking
- **Dynamic Chunk Sizing**: Automatically adjusts chunk sizes based on system performance
- **Success Rate Optimization**: Reduces chunk sizes when processing failures occur
- **Memory-aware Chunking**: Considers memory usage when determining optimal chunk sizes
- **Performance Tracking**: Monitors processing times to optimize chunk sizes

### Intelligent Concurrency Control
- **Dynamic Worker Adjustment**: Automatically scales the number of worker threads
- **Resource-based Scaling**: Considers both memory and API limits when determining concurrency
- **Batch Processing**: Processes files in batches to allow for dynamic scaling adjustments
- **Error Recovery**: Automatically reduces concurrency when errors are detected

### Scaling Configuration
```yaml
scaling:
  enable_adaptive_scaling: true
  min_concurrent_requests: 1
  max_concurrent_requests: 10
  memory_threshold_percent: 80.0
  rate_limit_backoff_factor: 2.0
  max_backoff_time: 300
  chunk_size_scaling_factor: 0.5
  min_chunk_size: 500
  max_chunk_size: 8000
```

### Scaling Demo
Test the intelligent scaling system:
```bash
# Demonstrate rate limiting adaptation
python scaling_demo.py --scenario rate_limit --duration 30

# Demonstrate memory pressure handling
python scaling_demo.py --scenario memory_pressure --duration 20

# Demonstrate mixed scenarios (default)
python scaling_demo.py --scenario mixed --duration 60
```

## 🔒 Security Features

### Path Validation
- Prevents directory traversal attacks
- Validates all input and output paths
- Restricts access to authorized directories only

### Security Scanning
- Detects potentially malicious code patterns
- Identifies suspicious function calls
- Flags dangerous operations

### Rate Limiting
- Built-in API rate limiting with intelligent scaling
- Configurable request throttling with adaptive adjustment
- Prevents API abuse and quota exhaustion
- Automatic recovery from rate limit errors

### Audit Trail
- Complete processing logs
- Detailed error reporting
- Backup creation tracking
- Scaling event monitoring

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_arcgen.py

# Run with coverage
python -m pytest tests/ --cov=arcgen_v2 --cov-report=html
```

### Test Coverage
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Mock testing for external API calls
- Error condition testing

## 🔧 Development

### Project Structure
```
ARCGen-V2/
├── arcgen_v2.py          # Main application
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
├── env.example          # Environment variables template
├── tests/               # Test suite
│   └── test_arcgen.py   # Unit tests
├── docs/                # Documentation
└── examples/            # Usage examples
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Maintain test coverage above 80%

## 📝 Changelog

### Version 2.0.0
- **New**: Multi-provider AI support
- **New**: Smart context-aware chunking
- **New**: Configuration management system
- **New**: Comprehensive backup system
- **New**: Security scanning and validation
- **Enhanced**: Error handling and recovery
- **Enhanced**: Progress tracking and reporting
- **Enhanced**: Logging and audit trail

### Version 1.0.0
- Initial release with basic functionality
- DeepSeek API integration
- Simple file processing
- Basic chunking strategy

## 🆘 Troubleshooting

### Common Issues

#### API Key Errors
```
Error: DEEPSEEK_API_KEY environment variable not set
```
**Solution**: Ensure your `.env` file contains the correct API key.

#### Rate Limiting
```
Error: API request failed: 429 Too Many Requests
```
**Solution**: The intelligent scaling system now automatically handles rate limiting with exponential backoff and adaptive concurrency control. No manual intervention required.

#### Memory Issues
```
Error: Memory limit exceeded
```
**Solution**: The intelligent scaling system automatically monitors memory usage and scales down operations when memory pressure is detected. It also performs automatic memory cleanup.

#### File Processing Errors
```
Error: Failed to process file
```
**Solution**: Check file permissions and ensure file is not corrupted.

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
export LOG_LEVEL=DEBUG
python arcgen_v2.py /path/to/addon --verbose
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original ARCGen V1 by Tammy
- DeepSeek team for the excellent API
- Rich library for beautiful console output
- Click framework for CLI interface
- All contributors and testers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

---

**Remember**: Always ensure you have proper authorization before using this tool on any systems or code that you do not own. 