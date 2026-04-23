from core.machine_lib import WorldQuantBrain
from core.log_manager import setup_logger
from time import sleep
import time
import json
import os
from itertools import product
import requests
import argparse

logger = setup_logger(__name__, "miner")

class MachineMiner:
    def __init__(self, username: str, password: str):
        self.brain = WorldQuantBrain(username, password)
        self.alpha_bag = []
        self.gold_bag = []
        
    def mine_alphas(self, region="USA", universe="TOP3000"):
        logger.info(f"Starting machine alpha mining for region: {region}, universe: {universe}")
        
        while True:
            try:
                # Get data fields
                logger.info("Fetching data fields...")
                fields_df = self.brain.get_datafields(region=region, universe=universe)
                logger.info(f"Got {len(fields_df)} data fields")
                
                matrix_fields = self.brain.process_datafields(fields_df, "matrix")
                vector_fields = self.brain.process_datafields(fields_df, "vector")
                logger.info(f"Processed {len(matrix_fields)} matrix fields and {len(vector_fields)} vector fields")
                
                # Generate first order alphas
                logger.info("Generating first order alphas...")
                first_order = self.brain.get_first_order(vector_fields + matrix_fields, self.brain.ops_set)
                logger.info(f"Generated {len(first_order)} first order alphas")
                logger.info(f"Sample alphas: {first_order[:3]}")
                
                # Process alphas one at a time
                for i, alpha in enumerate(first_order):
                    logger.info(f"Processing alpha {i+1}/{len(first_order)}: {alpha}")
                    
                    # Create alpha data with decay 0
                    alpha_data = [(alpha, 0)]
                    
                    try:
                        # Run single simulation
                        results = self.brain.single_simulate(
                            alpha_data=alpha_data,
                            neut="INDUSTRY",
                            region=region,
                            universe=universe
                        )
                        
                        # Process results
                        if results:
                            for result in results:
                                if self._process_result(result, alpha):
                                    logger.info(f"Found promising alpha: {alpha}")
                                    
                        # Add small delay between simulations
                        sleep(1)
                        
                        # Reauth periodically
                        if i % 200 == 0:
                            self.brain.login()
                            
                    except Exception as e:
                        logger.error(f"Error processing alpha {alpha}: {str(e)}")
                        sleep(60)  # Longer sleep on error
                        self.brain.login()
                        continue
                    
            except Exception as e:
                logger.error(f"Error in mining loop: {str(e)}")
                sleep(600)
                self.brain.login()
                continue

    def _process_result(self, result: dict, alpha: str) -> bool:
        """Process a single simulation result."""
        try:
            if not result.get("is"):
                return False
                
            is_data = result["is"]
            
            # Check criteria
            if (is_data["sharpe"] > 1.25 and
                is_data["turnover"] > 0.01 and
                is_data["turnover"] < 0.7 and
                is_data["fitness"] >= 1.0):
                
                # Store successful alpha
                self.alpha_bag.append(alpha)
                
                # Save to file
                self._save_result(alpha, result)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            return False

    def _save_result(self, alpha: str, result: dict):
        """Save successful alpha result to file."""
        timestamp = int(time.time())
        output = {
            "timestamp": timestamp,
            "alpha": alpha,
            "result": result
        }
        
        filename = f'successful_alpha_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Saved successful alpha to {filename}")

def main():
    from core.config import load_credentials
    parser = argparse.ArgumentParser(description='Mine alphas using WorldQuant Brain')
    parser.add_argument('--region', default='USA', help='Region to mine alphas for')
    parser.add_argument('--universe', default='TOP3000', help='Universe to mine alphas for')
    
    args = parser.parse_args()
    
    try:
        username, password = load_credentials()
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        return 1
    
    miner = MachineMiner(username, password)
    miner.mine_alphas(region=args.region, universe=args.universe)

if __name__ == "__main__":
    main() 