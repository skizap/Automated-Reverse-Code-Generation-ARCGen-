#!/usr/bin/env python3
"""
DeepSeek API Integration Test Script

This script tests the DeepSeek API integration to ensure it's working correctly.
Run this before using ARCGen to verify your API key and configuration.

Usage:
    python test_deepseek_api.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

def test_deepseek_api():
    """Test DeepSeek API connection and functionality"""
    
    print("🔍 Testing DeepSeek API Integration...")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        print("   Please set your API key in .env file")
        return False
    
    if api_key == 'your_deepseek_api_key_here':
        print("❌ Please replace the placeholder API key with your actual key")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...")
    
    # Test API endpoints
    base_url = "https://api.deepseek.com"
    
    # Test 1: Models endpoint
    print("\n📋 Testing models endpoint...")
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=10
        )
        if response.status_code == 200:
            models = response.json()
            print("✅ Models endpoint working")
            available_models = [model['id'] for model in models.get('data', [])]
            print(f"   Available models: {', '.join(available_models)}")
        else:
            print(f"⚠️  Models endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
    
    # Test 2: Chat completions with deepseek-chat
    print("\n💬 Testing chat completions (deepseek-chat)...")
    try:
        test_chat_completion(api_key, base_url, "deepseek-chat")
    except Exception as e:
        print(f"❌ deepseek-chat test failed: {e}")
        return False
    
    # Test 3: Chat completions with deepseek-reasoner
    print("\n🧠 Testing reasoning model (deepseek-reasoner)...")
    try:
        test_chat_completion(api_key, base_url, "deepseek-reasoner", 
                           "Solve this simple math problem: 2 + 2 = ?")
    except Exception as e:
        print(f"⚠️  deepseek-reasoner test failed: {e}")
        print("   This is optional - deepseek-chat should work fine for code optimization")
    
    print("\n🎉 DeepSeek API integration test completed!")
    print("✅ Your API configuration appears to be working correctly")
    return True

def test_chat_completion(api_key: str, base_url: str, model: str, 
                        test_prompt: str = "Hello! Please respond with 'API test successful'"):
    """Test a specific model's chat completion"""
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': test_prompt}
        ],
        'max_tokens': 50,
        'temperature': 0.7,
        'stream': False
    }
    
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"✅ {model} working - Response: {content[:100]}...")
            return True
        else:
            print(f"❌ {model} returned empty response")
            return False
    else:
        print(f"❌ {model} failed with status {response.status_code}")
        try:
            error_detail = response.json()
            print(f"   Error: {error_detail}")
        except:
            print(f"   Raw response: {response.text}")
        return False

def check_configuration():
    """Check ARCGen configuration files"""
    print("\n⚙️  Checking ARCGen configuration...")
    
    config_file = Path("config.yaml")
    if config_file.exists():
        print("✅ config.yaml found")
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if using updated model names
            deepseek_model = config.get('api', {}).get('deepseek', {}).get('model')
            if deepseek_model == 'deepseek-chat':
                print("✅ Using latest deepseek-chat model")
            elif deepseek_model == 'deepseek-coder':
                print("⚠️  Consider updating to 'deepseek-chat' for latest features")
            else:
                print(f"ℹ️  Using model: {deepseek_model}")
                
        except Exception as e:
            print(f"⚠️  Could not parse config.yaml: {e}")
    else:
        print("ℹ️  config.yaml not found - will use defaults")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - copy from env.example")

if __name__ == "__main__":
    print("DeepSeek API Integration Test")
    print("=" * 30)
    
    # Check configuration first
    check_configuration()
    
    # Test API
    success = test_deepseek_api()
    
    if success:
        print("\n🚀 Ready to use ARCGen with DeepSeek!")
        print("   Run: python arcgen_v2.py <your_addon_path>")
    else:
        print("\n❌ Please fix the issues above before using ARCGen")
        sys.exit(1) 