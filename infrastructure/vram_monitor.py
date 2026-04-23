#!/usr/bin/env python3

import subprocess
import time
import json
import os
from typing import Dict, List

from core.log_manager import setup_logger
logger = setup_logger(__name__, "infrastructure")

class VRAMMonitor:
    def __init__(self, vram_threshold: float = 0.9, check_interval: int = 60):
        """
        Initialize VRAM monitor.
        
        Args:
            vram_threshold: Percentage of VRAM usage that triggers cleanup (0.0-1.0)
            check_interval: How often to check VRAM usage in seconds
        """
        self.vram_threshold = vram_threshold
        self.check_interval = check_interval
        self.restart_count = 0
        self.max_restarts = 3
        
    def get_gpu_info(self) -> List[Dict]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return []
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'utilization': int(parts[4])
                        })
            
            return gpu_info
            
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi command timed out")
            return []
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return []
    
    def check_vram_usage(self) -> bool:
        """Check if VRAM usage is above threshold."""
        gpu_info = self.get_gpu_info()
        
        for gpu in gpu_info:
            memory_usage = gpu['memory_used'] / gpu['memory_total']
            logger.info(f"GPU {gpu['index']} ({gpu['name']}): "
                       f"{gpu['memory_used']}MB/{gpu['memory_total']}MB "
                       f"({memory_usage:.1%}) - Utilization: {gpu['utilization']}%")
            
            if memory_usage > self.vram_threshold:
                logger.warning(f"GPU {gpu['index']} VRAM usage ({memory_usage:.1%}) "
                             f"exceeds threshold ({self.vram_threshold:.1%})")
                return True
        
        return False
    
    def restart_ollama_service(self) -> bool:
        """Restart the Ollama service to free VRAM."""
        try:
            logger.info("Restarting Ollama service...")
            
            # Stop Ollama
            subprocess.run(['docker-compose', '-f', 'docker-compose.gpu.yml', 'stop', 'naive-ollma'], 
                         timeout=30)
            
            # Wait a bit for cleanup
            time.sleep(10)
            
            # Start Ollama
            subprocess.run(['docker-compose', '-f', 'docker-compose.gpu.yml', 'start', 'naive-ollma'], 
                         timeout=30)
            
            # Wait for service to be ready
            time.sleep(30)
            
            logger.info("Ollama service restarted successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Timeout while restarting Ollama service")
            return False
        except Exception as e:
            logger.error(f"Error restarting Ollama service: {e}")
            return False
    
    def cleanup_vram(self) -> bool:
        """Attempt to clean up VRAM without restarting."""
        try:
            logger.info("Attempting VRAM cleanup...")
            
            # Try to restart just the Ollama container
            result = subprocess.run([
                'docker', 'exec', 'naive-ollma-gpu', 
                'sh', '-c', 'pkill -f ollama && sleep 5 && ollama serve'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("VRAM cleanup completed")
                return True
            else:
                logger.warning(f"VRAM cleanup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during VRAM cleanup: {e}")
            return False
    
    def run(self):
        """Main monitoring loop."""
        logger.info(f"Starting VRAM monitor with threshold {self.vram_threshold:.1%}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        while True:
            try:
                if self.check_vram_usage():
                    logger.warning("High VRAM usage detected!")
                    
                    # Try cleanup first
                    if self.cleanup_vram():
                        logger.info("VRAM cleanup successful")
                        time.sleep(30)  # Wait and check again
                        continue
                    
                    # If cleanup failed and we haven't exceeded max restarts
                    if self.restart_count < self.max_restarts:
                        logger.warning(f"Attempting service restart ({self.restart_count + 1}/{self.max_restarts})")
                        if self.restart_ollama_service():
                            self.restart_count += 1
                            time.sleep(60)  # Wait longer after restart
                            continue
                    else:
                        logger.error("Maximum restart attempts exceeded. Manual intervention required.")
                        break
                
                # Reset restart count if VRAM usage is normal
                if self.restart_count > 0:
                    logger.info("VRAM usage normal, resetting restart count")
                    self.restart_count = 0
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("VRAM monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor VRAM usage and manage GPU resources')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='VRAM usage threshold (0.0-1.0, default: 0.9)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    monitor = VRAMMonitor(
        vram_threshold=args.threshold,
        check_interval=args.interval
    )
    
    monitor.run()

if __name__ == "__main__":
    main()
