#!/usr/bin/env python3
"""
Health check script for Naive-Ollma Docker setup
"""

import requests
import json
import sys
import os
from requests.auth import HTTPBasicAuth

def check_ollama():
    """Check if Ollama is running and FinGPT model is available"""
    print("🔍 Checking Ollama...")
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running with {len(models)} models")
            
            # Check for local Ollama models
            local_models = [m for m in models if "qwen3.5" in m.get("name", "").lower() or "qwen3-coder" in m.get("name", "").lower() or "deepseek-coder" in m.get("name", "").lower()]
            if local_models:
                print(f"✅ Local Ollama model found: {local_models[0]['name']}")
                return True
            else:
                print("⚠️  Preferred Ollama model not found. You may need to pull one:")
                print("   ollama pull qwen3.5:35b")
                return False
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_worldquant_credentials():
    """Check if WorldQuant Brain credentials are valid"""
    print("🔍 Checking WorldQuant Brain credentials...")
    
    try:
        from core.config import load_credentials, fix_session_proxy
        username, password = load_credentials()
    except Exception as e:
        print(f"❌ Error loading credentials from .env: {e}")
        return False
    
    try:
        session = requests.Session()
        fix_session_proxy(session)
        session.auth = HTTPBasicAuth(username, password)
        
        response = session.post('https://api.worldquantbrain.com/authentication', timeout=30)
        
        if response.status_code == 201:
            print("✅ WorldQuant Brain authentication successful")
            return True
        else:
            print(f"❌ WorldQuant Brain authentication failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def check_docker_services():
    """Check if Docker services are running"""
    print("🔍 Checking Docker services...")
    
    try:
        import subprocess
        result = subprocess.run(
            ["docker-compose", "ps"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            if "Up" in result.stdout:
                print("✅ Docker services are running")
                return True
            else:
                print("⚠️  Docker services are not running")
                return False
        else:
            print(f"❌ Error checking Docker services: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ docker-compose not found. Is Docker installed?")
        return False
    except Exception as e:
        print(f"❌ Error checking Docker services: {e}")
        return False

def test_ollama_generation():
    """Test if Ollama can generate responses"""
    print("🔍 Testing Ollama generation...")
    
    try:
        test_prompt = "Generate a simple alpha factor expression: ts_mean(returns, 20)"
        
        data = {
            'model': 'qwen3.5:35b',
            'prompt': test_prompt,
            'stream': False,
            'temperature': 0.1,
            'num_predict': 100  # Use num_predict instead of max_tokens for Ollama
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print("✅ Ollama generation test successful")
                return True
            else:
                print("❌ Unexpected Ollama response format")
                return False
        else:
            print(f"❌ Ollama generation test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama generation: {e}")
        return False

def main():
    """Run all health checks"""
    print("🏥 Naive-Ollma Health Check")
    print("=" * 40)
    
    checks = [
        ("Docker Services", check_docker_services),
        ("Ollama", check_ollama),
        ("Ollama Generation", test_ollama_generation),
        ("WorldQuant Credentials", check_worldquant_credentials),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error in {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 40)
    print("📊 Health Check Summary:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! Your setup is ready to go.")
        print("   Run: docker-compose up -d")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print("   Check the README_Docker.md for troubleshooting tips.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
