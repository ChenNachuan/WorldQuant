import argparse
import logging
import requests
import json
import os
import re
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Tuple
import time

from core.log_manager import setup_logger
logger = setup_logger(__name__, "miner")

class AlphaExpressionMiner:
    def __init__(self):
        logger.info("Initializing AlphaExpressionMiner")
        self.sess = requests.Session()
        from core.config import fix_session_proxy
        fix_session_proxy(self.sess)
        self.setup_auth()
        
    def setup_auth(self) -> None:
        from core.config import load_credentials
        logger.info("Loading credentials from .env")
        username, password = load_credentials()
        self.sess.auth = HTTPBasicAuth(username, password)
        
        logger.info("Authenticating with WorldQuant Brain...")
        response = self.sess.post('https://api.worldquantbrain.com/authentication')
        logger.info(f"Authentication response status: {response.status_code}")
        
        if response.status_code != 201:
            logger.error(f"Authentication failed: {response.text}")
            raise Exception(f"Authentication failed: {response.text}")
        logger.info("Authentication successful")

    def remove_alpha_from_hopeful(self, expression: str) -> bool:
        from core.alpha_store import remove_alpha_by_expression
        return remove_alpha_by_expression(expression)

    def parse_expression(self, expression: str) -> List[Dict]:
        """Parse the alpha expression to find numeric parameters and their positions."""
        logger.info(f"Parsing expression: {expression}")
        parameters = []
        # Match numbers that:
        # 1. Are preceded by '(' or ',' or space
        # 2. Are not part of a variable name (not preceded/followed by letters)
        # 3. Can be integers or decimals
        for match in re.finditer(r'(?<=[,()\s])(-?\d*\.?\d+)(?![a-zA-Z])', expression):
            number_str = match.group()
            try:
                number = float(number_str)
            except ValueError:
                continue
            start_pos = match.start()
            end_pos = match.end()
            parameters.append({
                'value': number,
                'start': start_pos,
                'end': end_pos,
                'context': expression[max(0, start_pos-20):min(len(expression), end_pos+20)],
                'is_integer': number.is_integer()
            })
            logger.debug(f"Found parameter: {number} at position {start_pos}-{end_pos}")
        
        logger.info(f"Found {len(parameters)} parameters to vary")
        return parameters

    def get_user_parameter_selection(self, parameters: List[Dict]) -> List[Dict]:
        """Interactively get user selection for parameters to vary."""
        if not parameters:
            logger.info("No parameters found in expression")
            return []

        print("\nFound the following parameters in the expression:")
        for i, param in enumerate(parameters, 1):
            print(f"{i}. Value: {param['value']} | Context: ...{param['context']}...")

        while True:
            try:
                selection = input("\nEnter the numbers of parameters to vary (comma-separated, or 'all'): ")
                if selection.lower() == 'all':
                    selected_indices = list(range(len(parameters)))
                else:
                    selected_indices = [int(x.strip())-1 for x in selection.split(',')]
                    if not all(0 <= i < len(parameters) for i in selected_indices):
                        raise ValueError("Invalid parameter number")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        selected_params = [parameters[i] for i in selected_indices]
        return selected_params

    def get_parameter_ranges(self, parameters: List[Dict], auto_mode: bool = False) -> List[Dict]:
        """Get range and step size for each selected parameter."""
        for param in parameters:
            if auto_mode:
                # Use default ranges for automated mode
                original_value = param['value']
                if param['is_integer']:
                    # For integers, use ±20% range with step of 1
                    range_val = max(1, abs(original_value) * 0.2)
                    min_val = max(1, original_value - range_val)
                    max_val = original_value + range_val
                    step = 1
                else:
                    # For floats, use ±10% range with step of 10% of the range
                    range_val = abs(original_value) * 0.1
                    min_val = original_value - range_val
                    max_val = original_value + range_val
                    step = range_val / 5  # 5 steps across the range
                
                logger.info(f"Auto mode: Parameter {param['value']} -> range [{min_val:.2f}, {max_val:.2f}], step {step:.2f}")
            else:
                # Interactive mode - get user input
                while True:
                    try:
                        print(f"\nParameter: {param['value']} | Context: ...{param['context']}...")
                        range_input = input("Enter range (e.g., '10' for ±10, or '5,15' for 5 to 15): ")
                        if ',' in range_input:
                            min_val, max_val = map(float, range_input.split(','))
                        else:
                            range_val = float(range_input)
                            min_val = param['value'] - range_val
                            max_val = param['value'] + range_val

                        step = float(input("Enter step size: "))
                        if step <= 0:
                            raise ValueError("Step size must be positive")
                        if min_val >= max_val:
                            raise ValueError("Min value must be less than max value")

                        # If the original value was an integer, ensure step is also an integer
                        if param['is_integer'] and not step.is_integer():
                            print("Warning: Original value is integer, rounding step to nearest integer")
                            step = round(step)

                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}. Please try again.")

            param['min'] = min_val
            param['max'] = max_val
            param['step'] = step

        return parameters

    def generate_variations(self, expression: str, parameters: List[Dict]) -> List[str]:
        """Generate variations of the expression based on user-selected parameters and ranges."""
        logger.info("Generating variations based on selected parameters")
        variations = []
        
        # Sort parameters in reverse order to modify from end to start
        parameters.sort(reverse=True, key=lambda x: x['start'])
        
        # Generate all combinations of parameter values
        param_values = []
        for param in parameters:
            values = []
            current = param['min']
            while current <= param['max']:
                # Format the number appropriately based on whether it's an integer
                if param['is_integer']:
                    value = str(int(round(current)))
                else:
                    # Format to remove trailing zeros and unnecessary decimal points
                    value = f"{current:.10f}".rstrip('0').rstrip('.')
                values.append(value)
                current += param['step']
            
            # Add original value if not already included
            original_value = str(int(param['value'])) if param['is_integer'] else f"{param['value']:.10f}".rstrip('0').rstrip('.')
            if original_value not in values:
                values.append(original_value)
            
            param_values.append(values)
        
        # Generate all combinations
        from itertools import product
        for value_combination in product(*param_values):
            new_expr = expression
            for param, value in zip(parameters, value_combination):
                new_expr = new_expr[:param['start']] + value + new_expr[param['end']:]
            variations.append(new_expr)
            logger.debug(f"Generated variation: {new_expr}")
        
        logger.info(f"Generated {len(variations)} total variations")
        return variations

    def test_alpha(self, alpha_expression: str) -> Dict:
        """Test an alpha expression using WorldQuant Brain simulation."""
        logger.info(f"Testing alpha: {alpha_expression}")
        
        simulation_data = {
            'type': 'REGULAR',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': 'USA',
                'universe': 'TOP3000',
                'delay': 1,
                'decay': 0,
                'neutralization': 'INDUSTRY',
                'truncation': 0.01,
                'pasteurization': 'ON',
                'unitHandling': 'VERIFY',
                'nanHandling': 'OFF',
                'language': 'FASTEXPR',
                'visualization': False,
            },
            'regular': alpha_expression
        }

        sim_resp = self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data)
        logger.info(f"Simulation creation response: {sim_resp.status_code}")
        
        if sim_resp.status_code != 201:
            logger.error(f"Simulation creation failed: {sim_resp.text}")
            return {"status": "error", "message": sim_resp.text}

        sim_progress_url = sim_resp.headers.get('location')
        if not sim_progress_url:
            logger.error("No simulation ID received in response headers")
            return {"status": "error", "message": "No simulation ID received"}
        
        logger.info(f"Monitoring simulation at: {sim_progress_url}")
        
        # Monitor simulation progress
        retry_count = 0
        max_retries = 3
        while True:
            try:
                sim_progress_resp = self.sess.get(sim_progress_url)
                
                # Handle empty response
                if not sim_progress_resp.text.strip():
                    logger.debug("Empty response, simulation still initializing...")
                    sleep(10)
                    continue
                
                # Try to parse JSON response
                try:
                    progress_data = sim_progress_resp.json()
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON response: {sim_progress_resp.text}")
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error("Max retries exceeded for JSON decode")
                        return {"status": "error", "message": "Failed to decode simulation response"}
                    sleep(10)
                    continue
                
                status = progress_data.get("status")
                logger.info(f"Simulation status: {status}")
                
                if status == "COMPLETE" or status == "WARNING":
                    logger.info("Simulation completed successfully")
                    return {"status": "success", "result": progress_data}
                elif status in ["FAILED", "ERROR"]:
                    logger.error(f"Simulation failed: {progress_data}")
                    return {"status": "error", "message": progress_data}
                
                sleep(10)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                retry_count += 1
                if retry_count > max_retries:
                    return {"status": "error", "message": f"Request failed after {max_retries} retries"}
                sleep(10)

def main():
    parser = argparse.ArgumentParser(description='Mine alpha expression variations')
    parser.add_argument('--expression', type=str, required=True,
                      help='Base alpha expression to mine variations from')
    parser.add_argument('--output', type=str, default='mined_expressions.json',
                      help='Output file for results (default: mined_expressions.json)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--auto-mode', action='store_true',
                      help='Run in automated mode without user interaction')
    parser.add_argument('--output-file', type=str, default='mined_expressions.json',
                      help='Output file for results (default: mined_expressions.json)')
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting alpha expression mining with parameters:")
    logger.info(f"Expression: {args.expression}")
    logger.info(f"Output file: {args.output}")
    
    miner = AlphaExpressionMiner()
    
    # Parse expression and get parameters
    parameters = miner.parse_expression(args.expression)
    
    # Get parameter selection (automated or interactive)
    if args.auto_mode:
        # In auto mode, select all parameters
        selected_params = parameters
        logger.info(f"Auto mode: selected all {len(selected_params)} parameters")
    else:
        # Get user selection for parameters to vary
        selected_params = miner.get_user_parameter_selection(parameters)
    
    if not selected_params:
        logger.info("No parameters selected for variation")
        # Still remove the alpha from hopeful_alphas.json even if no parameters found
        logger.info("Mining completed (no parameters to vary), removing alpha from hopeful_alphas.json")
        removed = miner.remove_alpha_from_hopeful(args.expression)
        if removed:
            logger.info(f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json")
        else:
            logger.warning(f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)")
        return
    
    # Get ranges and steps for selected parameters
    selected_params = miner.get_parameter_ranges(selected_params, auto_mode=args.auto_mode)
    
    # Generate variations
    variations = miner.generate_variations(args.expression, selected_params)
    
    # Test variations
    results = []
    total = len(variations)
    for i, var in enumerate(variations, 1):
        logger.info(f"Testing variation {i}/{total}: {var}")
        result = miner.test_alpha(var)
        if result["status"] == "success":
            logger.info(f"Successful test for: {var}")
            results.append({
                "expression": var,
                "result": result["result"]
            })
        else:
            logger.error(f"Failed to test variation: {var}")
            logger.error(f"Error: {result['message']}")
    
    # Save results
    output_file = args.output_file if hasattr(args, 'output_file') else args.output
    logger.info(f"Saving {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Always remove the mined alpha from hopeful_alphas.json after completion
    # This prevents the same alpha from being processed again
    logger.info("Mining completed, removing alpha from hopeful_alphas.json")
    removed = miner.remove_alpha_from_hopeful(args.expression)
    if removed:
        logger.info(f"Successfully removed alpha '{args.expression}' from hopeful_alphas.json")
    else:
        logger.warning(f"Could not remove alpha '{args.expression}' from hopeful_alphas.json (may not exist)")
    
    logger.info("Mining complete")

if __name__ == "__main__":
    main()
