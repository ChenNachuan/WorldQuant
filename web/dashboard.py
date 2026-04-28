from flask import Flask, render_template, jsonify, request
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, List

from core.log_manager import setup_logger, LOG_DIR

app = Flask(__name__)

logger = setup_logger(__name__, "dashboard")

class AlphaDashboard:
    def __init__(self):
        self.status_file = "dashboard_status.json"
        self.log_file = os.path.join(LOG_DIR, "orchestrator", f"{datetime.now().strftime('%Y-%m-%d')}.log")
        self.submission_log_file = os.path.join("data", "submission_log.json")
        self.results_dir = "results"
        self.logs_dir = LOG_DIR
        
    def get_system_status(self) -> Dict:
        """Get overall system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self.get_gpu_status(),
            "ollama": self.get_ollama_status(),
            "orchestrator": self.get_orchestrator_status(),
            "worldquant": self.get_worldquant_status(),
            "recent_activity": self.get_recent_activity(),
            "statistics": self.get_statistics()
        }
        return status
    
    def get_gpu_status(self) -> Dict:
        """Get GPU status and utilization."""
        try:
            # Try to get GPU info from nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(', ')
                    if len(parts) >= 5:
                        return {
                            "status": "active",
                            "name": parts[0],
                            "memory_used_mb": int(parts[1]),
                            "memory_total_mb": int(parts[2]),
                            "utilization_percent": int(parts[3]),
                            "temperature_c": int(parts[4]),
                            "memory_percent": round((int(parts[1]) / int(parts[2])) * 100, 1)
                        }
        except Exception as e:
            logger.warning(f"Could not get GPU status: {e}")
        
        return {"status": "unknown", "error": "GPU information not available"}
    
    def get_ollama_status(self) -> Dict:
        """Get Ollama service status."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "status": "running",
                    "models": [model.get("name", "") for model in models],
                    "model_count": len(models)
                }
        except Exception as e:
            logger.warning(f"Could not get Ollama status: {e}")
        
        return {"status": "not_responding", "error": "Ollama service not available"}
    
    def get_orchestrator_status(self) -> Dict:
        """Get orchestrator status from Docker container logs."""
        status = {
            "status": "unknown",
            "last_activity": None,
            "current_mode": "continuous",
            "next_mining": None,
            "next_submission": None
        }
        
        try:
            # Try to get Docker container logs
            result = subprocess.run([
                "docker", "logs", "--tail", "50", "naive-ollma-gpu"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    last_line = lines[-1].strip()
                    status["last_activity"] = last_line
                    
                    # Check for recent activity in Docker logs
                    for line in reversed(lines[-50:]):
                        if any(keyword in line for keyword in ["alpha generator", "generating alpha", "Running alpha", "alpha idea"]):
                            status["status"] = "active"
                            break
                        elif any(keyword in line for keyword in ["Error", "Failed", "Exception"]):
                            status["status"] = "error"
                            break
                        elif "ollama" in line.lower() and "started" in line.lower():
                            status["status"] = "active"
                            break
                
                # Check submission schedule
                if os.path.exists(self.submission_log_file):
                    with open(self.submission_log_file, 'r') as f:
                        data = json.load(f)
                        last_submission = data.get("last_submission_date")
                        if last_submission:
                            last_date = datetime.fromisoformat(last_submission)
                            next_submission = last_date + timedelta(days=1)
                            next_submission = next_submission.replace(hour=14, minute=0, second=0, microsecond=0)
                            status["next_submission"] = next_submission.isoformat()
                
                # Calculate next mining time (every 6 hours)
                now = datetime.now()
                hours_since_midnight = now.hour + now.minute / 60
                next_mining_hour = ((int(hours_since_midnight // 6) + 1) * 6) % 24
                next_mining = now.replace(hour=int(next_mining_hour), minute=0, second=0, microsecond=0)
                if next_mining <= now:
                    next_mining += timedelta(days=1)
                status["next_mining"] = next_mining.isoformat()
                
        except Exception as e:
            logger.warning(f"Could not get orchestrator status: {e}")
        
        return status
    
    def get_worldquant_status(self) -> Dict:
        """Check WorldQuant Brain API status."""
        try:
            from core.api_session import get_session_manager
            session = get_session_manager().session
            response = session.post('https://api.worldquantbrain.com/authentication', timeout=10)
            
            if response.status_code == 201:
                return {"status": "connected", "message": "Authentication successful"}
            else:
                return {"status": "auth_failed", "message": f"Status: {response.status_code}"}
        except Exception as e:
            logger.warning(f"Could not check WorldQuant status: {e}")
        
        return {"status": "unknown", "message": "Could not verify connection"}
    
    def get_recent_activity(self) -> List[Dict]:
        """Get recent activity from Docker container logs."""
        activities = []
        
        try:
            # Get logs from Docker container
            result = subprocess.run([
                "docker", "logs", "--tail", "20", "naive-ollma-gpu"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Get last 20 lines
                for line in lines[-20:]:
                    line = line.strip()
                    if line and not line.startswith('---'):
                        # Parse timestamp and message
                        try:
                            # Try to parse Docker log format
                            if 'time=' in line:
                                # Docker log format: time=2025-08-10T21:48:18.314Z level=INFO source=server.go:637 msg="..."
                                parts = line.split('msg="')
                                if len(parts) > 1:
                                    timestamp_part = parts[0].split('time=')[1].split(' ')[0]
                                    message = parts[1].rstrip('"')
                                    timestamp = datetime.fromisoformat(timestamp_part.replace('Z', '+00:00'))
                                    activities.append({
                                        "timestamp": timestamp.isoformat(),
                                        "message": message,
                                        "type": "info" if "INFO" in line else "error" if "ERROR" in line else "warning" if "WARNING" in line else "debug"
                                    })
                                else:
                                    activities.append({
                                        "timestamp": datetime.now().isoformat(),
                                        "message": line,
                                        "type": "unknown"
                                    })
                            elif ' - ' in line:
                                # Standard log format
                                timestamp_str, message = line.split(' - ', 1)
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                activities.append({
                                    "timestamp": timestamp.isoformat(),
                                    "message": message,
                                    "type": "info" if "INFO" in message else "error" if "ERROR" in message else "warning" if "WARNING" in message else "debug"
                                })
                            else:
                                activities.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "message": line,
                                    "type": "unknown"
                                })
                        except:
                            activities.append({
                                "timestamp": datetime.now().isoformat(),
                                "message": line,
                                "type": "unknown"
                            })
            else:
                # Fallback to local log file
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-20:]:
                            line = line.strip()
                            if line and not line.startswith('---'):
                                try:
                                    if ' - ' in line:
                                        timestamp_str, message = line.split(' - ', 1)
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        activities.append({
                                            "timestamp": timestamp.isoformat(),
                                            "message": message,
                                            "type": "info" if "INFO" in message else "error" if "ERROR" in message else "warning" if "WARNING" in message else "debug"
                                        })
                                except:
                                    activities.append({
                                        "timestamp": datetime.now().isoformat(),
                                        "message": line,
                                        "type": "unknown"
                                    })
        except Exception as e:
            logger.warning(f"Could not read recent activity: {e}")
        
        return activities[-10:]  # Return last 10 activities
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated alphas and results."""
        stats = {
            "total_alphas_generated": 0,
            "successful_alphas": 0,
            "failed_alphas": 0,
            "last_24h_generated": 0,
            "last_24h_successful": 0
        }
        
        try:
            # Count files in results directory
            if os.path.exists(self.results_dir):
                result_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
                stats["total_alphas_generated"] = len(result_files)
                
                # Count successful alphas (files with content)
                for file in result_files:
                    file_path = os.path.join(self.results_dir, file)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            if data and len(data) > 0:
                                stats["successful_alphas"] += 1
                    except:
                        stats["failed_alphas"] += 1
                
                # Count last 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                for file in result_files:
                    file_path = os.path.join(self.results_dir, file)
                    if os.path.getmtime(file_path) > cutoff_time.timestamp():
                        stats["last_24h_generated"] += 1
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                if data and len(data) > 0:
                                    stats["last_24h_successful"] += 1
                        except:
                            pass
                            
        except Exception as e:
            logger.warning(f"Could not get statistics: {e}")
        
        return stats
    
    def get_logs(self, lines: int = 50) -> List[str]:
        """Get recent logs from Docker container."""
        logs = []
        try:
            # Get logs from Docker container
            result = subprocess.run([
                "docker", "logs", "--tail", str(lines), "naive-ollma-gpu"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logs = result.stdout.strip().split('\n')
            else:
                # Fallback to local log file if Docker fails
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        lines_list = f.readlines()
                        logs = lines_list[-lines:] if len(lines_list) > lines else lines_list
        except Exception as e:
            logger.warning(f"Could not read logs: {e}")
            # Fallback to local log file
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        lines_list = f.readlines()
                        logs = lines_list[-lines:] if len(lines_list) > lines else lines_list
            except:
                pass
        
        return [line.strip() for line in logs if line.strip()]
    
    def get_alpha_generator_logs(self, lines: int = 50) -> List[str]:
        """Get alpha generator specific logs."""
        logs = []
        try:
            # Get logs from Docker container and filter for alpha generator content
            result = subprocess.run([
                "docker", "logs", "--tail", str(lines * 2), "naive-ollma-gpu"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                all_logs = result.stdout.strip().split('\n')
                # Filter for alpha generator related logs
                alpha_logs = []
                for line in all_logs:
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in [
                        'alpha', 'generator', 'generating', 'ollama', 'model', 'prompt',
                        'response', 'idea', 'factor', 'worldquant', 'submission'
                    ]):
                        alpha_logs.append(line)
                logs = alpha_logs[-lines:] if len(alpha_logs) > lines else alpha_logs
            else:
                # Fallback to local log file
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        lines_list = f.readlines()
                        logs = lines_list[-lines:] if len(lines_list) > lines else lines_list
        except Exception as e:
            logger.warning(f"Could not read alpha generator logs: {e}")
        
        return [line.strip() for line in logs if line.strip()]
    
    def trigger_mining(self) -> Dict:
        """Trigger manual alpha expression mining."""
        try:
            result = subprocess.run([
                "python", "-m", "core.alpha_orchestrator", 
                "--mode", "miner"
            ], capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def trigger_submission(self) -> Dict:
        """Trigger manual alpha submission."""
        try:
            result = subprocess.run([
                "python", "-m", "core.alpha_orchestrator",
                "--mode", "submitter",
                "--batch-size", "3"
            ], capture_output=True, text=True, timeout=600)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def trigger_alpha_generation(self) -> Dict:
        """Trigger manual alpha generation."""
        try:
            result = subprocess.run([
                "python", "-m", "core.alpha_orchestrator",
                "--mode", "generator",
                "--batch-size", "1"
            ], capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global dashboard instance
dashboard = AlphaDashboard()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    return jsonify(dashboard.get_system_status())

@app.route('/api/logs')
def api_logs():
    """API endpoint for logs."""
    lines = request.args.get('lines', 50, type=int)
    return jsonify({"logs": dashboard.get_logs(lines)})

@app.route('/api/alpha_logs')
def api_alpha_logs():
    """API endpoint for alpha generator specific logs."""
    lines = request.args.get('lines', 50, type=int)
    return jsonify({"logs": dashboard.get_alpha_generator_logs(lines)})

@app.route('/api/trigger_mining', methods=['POST'])
def api_trigger_mining():
    """API endpoint to trigger manual mining."""
    result = dashboard.trigger_mining()
    return jsonify(result)

@app.route('/api/trigger_submission', methods=['POST'])
def api_trigger_submission():
    """API endpoint to trigger manual submission."""
    result = dashboard.trigger_submission()
    return jsonify(result)

@app.route('/api/trigger_alpha_generation', methods=['POST'])
def api_trigger_alpha_generation():
    """API endpoint to trigger manual alpha generation."""
    result = dashboard.trigger_alpha_generation()
    return jsonify(result)

@app.route('/api/refresh')
def api_refresh():
    """API endpoint to refresh status."""
    return jsonify(dashboard.get_system_status())

if __name__ == '__main__':
    template_dir = Path(__file__).resolve().parent / 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    print("Starting Alpha Generator Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Ollama WebUI: http://localhost:3000")
    print("Ollama API: http://localhost:11434")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
