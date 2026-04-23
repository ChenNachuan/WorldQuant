#!/usr/bin/env python3

import json
import subprocess
import time
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

from core.log_manager import setup_logger
logger = setup_logger(__name__, "infrastructure")

@dataclass
class ModelInfo:
    """Information about a model in the fleet."""
    name: str
    size_mb: int
    priority: int  # Lower number = higher priority (used first)
    description: str

class ModelFleetManager:
    """Manages a fleet of models with automatic downgrading on VRAM issues."""
    
    def __init__(self, container_name: str = "naive-ollma-gpu"):
        self.container_name = container_name
        self.current_model_index = 0
        self.vram_error_count = 0
        self.max_vram_errors = 3  # Number of VRAM errors before downgrading
        
        # Model fleet ordered by priority (largest to smallest)
        self.model_fleet = [
            ModelInfo("qwen3.5:35b", 23869, 1, "Qwen3.5 35B - Preferred default model"),
            ModelInfo("qwen3-coder:latest", 18557, 2, "Qwen3-Coder 30.5B - Code-focused fallback"),
            ModelInfo("deepseek-coder-v2:16b", 8905, 3, "DeepSeek-Coder-V2 16B - Lightweight fallback"),
        ]
        
        # State file to persist current model selection
        self.state_file = os.path.join("data", "model_fleet_state.json")
        self.load_state()
        
    def load_state(self):
        """Load the current model state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_model_index = state.get('current_model_index', 0)
                    self.vram_error_count = state.get('vram_error_count', 0)
                    logger.info(f"Loaded state: model_index={self.current_model_index}, vram_errors={self.vram_error_count}")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            self.current_model_index = 0
            self.vram_error_count = 0
    
    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                'current_model_index': self.current_model_index,
                'vram_error_count': self.vram_error_count,
                'current_model': self.get_current_model().name,
                'timestamp': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved state: {state}")
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_current_model(self) -> ModelInfo:
        """Get the current model in use."""
        if self.current_model_index >= len(self.model_fleet):
            self.current_model_index = len(self.model_fleet) - 1
        return self.model_fleet[self.current_model_index]
    
    def get_available_models(self) -> List[str]:
        """Get list of available models in the container."""
        try:
            result = subprocess.run([
                'docker', 'exec', self.container_name, 'ollama', 'list'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
            else:
                logger.error(f"Failed to get available models: {result.stderr}")
                return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a specific model is available, download if needed."""
        available_models = self.get_available_models()
        
        if model_name in available_models:
            logger.info(f"Model {model_name} is already available")
            return True
        
        logger.info(f"Model {model_name} not found, downloading...")
        try:
            result = subprocess.run([
                'docker', 'exec', self.container_name, 'ollama', 'pull', model_name
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded model {model_name}")
                return True
            else:
                logger.error(f"Failed to download model {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    def detect_vram_error(self, log_line: str) -> bool:
        """Detect VRAM recovery timeout errors in log lines."""
        vram_error_indicators = [
            "gpu VRAM usage didn't recover within timeout",
            "VRAM usage didn't recover",
            "gpu memory exhausted",
            "CUDA out of memory",
            "GPU memory allocation failed"
        ]
        
        return any(indicator.lower() in log_line.lower() for indicator in vram_error_indicators)
    
    def handle_vram_error(self) -> bool:
        """Handle VRAM error by downgrading to a smaller model."""
        self.vram_error_count += 1
        logger.warning(f"VRAM error detected! Count: {self.vram_error_count}/{self.max_vram_errors}")
        
        if self.vram_error_count >= self.max_vram_errors:
            return self.downgrade_model()
        
        self.save_state()
        return False
    
    def downgrade_model(self) -> bool:
        """Downgrade to the next smaller model in the fleet."""
        if self.current_model_index >= len(self.model_fleet) - 1:
            logger.error("Already using the smallest model in the fleet!")
            return False
        
        old_model = self.get_current_model()
        self.current_model_index += 1
        new_model = self.get_current_model()
        
        logger.warning(f"Downgrading model: {old_model.name} -> {new_model.name}")
        
        # Ensure the new model is available
        if not self.ensure_model_available(new_model.name):
            logger.error(f"Failed to ensure model {new_model.name} is available")
            self.current_model_index -= 1  # Revert
            return False
        
        # Reset VRAM error count
        self.vram_error_count = 0
        
        # Save state
        self.save_state()
        
        # Restart the application with new model
        return self.restart_with_new_model(new_model.name)
    
    def restart_with_new_model(self, model_name: str) -> bool:
        """Restart the application with the new model."""
        logger.info(f"Restarting application with model: {model_name}")
        
        try:
            # Update the alpha generator configuration
            self.update_alpha_generator_config(model_name)
            
            # Restart the Docker container
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.gpu.yml', 'restart', 'naive-ollma'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Successfully restarted with model {model_name}")
                return True
            else:
                logger.error(f"Failed to restart container: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error restarting with new model: {e}")
            return False
    
    def update_alpha_generator_config(self, model_name: str):
        """Update the alpha generator configuration to use the new model."""
        try:
            # Update the default model in alpha_generator_ollama.py
            with open('core/alpha_generator_ollama.py', 'r') as f:
                content = f.read()
            
            # Replace the default model
            content = content.replace(
                "default='qwen3.5:35b'",
                f"default='{model_name}'"
            )
            content = content.replace(
                "getattr(self, 'model_name', 'qwen3.5:35b')",
                f"getattr(self, 'model_name', '{model_name}')"
            )
            
            with open('core/alpha_generator_ollama.py', 'w') as f:
                f.write(content)
            
            logger.info(f"Updated alpha generator config to use {model_name}")
        except Exception as e:
            logger.error(f"Error updating alpha generator config: {e}")
    
    def monitor_logs(self):
        """Monitor Docker logs for VRAM errors."""
        logger.info("Starting VRAM error monitoring...")
        
        try:
            process = subprocess.Popen([
                'docker-compose', '-f', 'docker-compose.gpu.yml', 'logs', '-f', 'naive-ollma'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            for line in process.stdout:
                if self.detect_vram_error(line):
                    logger.warning(f"VRAM error detected in logs: {line.strip()}")
                    if self.handle_vram_error():
                        logger.info("Model downgraded and application restarted")
                        break  # Exit monitoring after restart
                
                # Also check for successful operations to reset error count
                if "successful" in line.lower() or "completed" in line.lower():
                    if self.vram_error_count > 0:
                        logger.info("Successful operation detected, resetting VRAM error count")
                        self.vram_error_count = 0
                        self.save_state()
        
        except KeyboardInterrupt:
            logger.info("VRAM monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error monitoring logs: {e}")
    
    def get_fleet_status(self) -> Dict:
        """Get the current status of the model fleet."""
        current_model = self.get_current_model()
        available_models = self.get_available_models()
        
        return {
            'current_model': {
                'name': current_model.name,
                'size_mb': current_model.size_mb,
                'description': current_model.description,
                'index': self.current_model_index
            },
            'vram_error_count': self.vram_error_count,
            'max_vram_errors': self.max_vram_errors,
            'available_models': available_models,
            'fleet_size': len(self.model_fleet),
            'can_downgrade': self.current_model_index < len(self.model_fleet) - 1
        }
    
    def reset_to_largest_model(self):
        """Reset to the largest model in the fleet."""
        self.current_model_index = 0
        self.vram_error_count = 0
        self.save_state()
        logger.info("Reset to largest model in fleet")
        return self.restart_with_new_model(self.get_current_model().name)

def main():
    """Main function for testing the model fleet manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Fleet Manager')
    parser.add_argument('--monitor', action='store_true', help='Start VRAM monitoring')
    parser.add_argument('--status', action='store_true', help='Show fleet status')
    parser.add_argument('--reset', action='store_true', help='Reset to largest model')
    parser.add_argument('--downgrade', action='store_true', help='Force downgrade to next model')
    
    args = parser.parse_args()
    
    manager = ModelFleetManager()
    
    if args.status:
        status = manager.get_fleet_status()
        print(json.dumps(status, indent=2))
    elif args.reset:
        manager.reset_to_largest_model()
    elif args.downgrade:
        manager.downgrade_model()
    elif args.monitor:
        manager.monitor_logs()
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()
