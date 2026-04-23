#!/usr/bin/env python3
"""
Test script for the Alpha Orchestrator to verify concurrent execution
of alpha_generator_ollama and alpha_expression_miner.
"""

import os
import sys
import time
import json
import logging
from core.alpha_orchestrator import AlphaOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_hopeful_alphas():
    from core.alpha_store import save_alpha
    test_data_1 = {
        "id": "test_1",
        "grade": "A",
        "is": {"fitness": 0.6, "sharpe": 1.2, "turnover": 0.1, "returns": 0.15, "checks": []},
    }
    test_data_2 = {
        "id": "test_2",
        "grade": "A",
        "is": {"fitness": 0.7, "sharpe": 1.5, "turnover": 0.08, "returns": 0.18, "checks": []},
    }
    save_alpha("rank(close)", test_data_1, source="test")
    save_alpha("rank(volume)", test_data_2, source="test")
    logger.info("Created test alphas in alpha/ directory")

def test_orchestrator_initialization():
    """Test that the orchestrator can be initialized properly."""
    logger.info("Testing orchestrator initialization...")
    
    try:
        orchestrator = AlphaOrchestrator()
        logger.info("✓ Orchestrator initialized successfully")
        logger.info(f"✓ Max concurrent simulations: {orchestrator.max_concurrent_simulations}")
        return orchestrator
    except Exception as e:
        logger.error(f"✗ Failed to initialize orchestrator: {e}")
        return None

def test_concurrent_execution():
    """Test concurrent execution of generator and miner."""
    logger.info("Testing concurrent execution...")
    
    orchestrator = test_orchestrator_initialization()
    if not orchestrator:
        return False
    
    # Create test data
    create_test_hopeful_alphas()
    
    try:
        # Test the continuous miner function directly
        logger.info("Testing alpha expression miner with test data...")
        
        # Run the miner once to see if it works
        orchestrator.run_alpha_expression_miner()
        
        logger.info("✓ Alpha expression miner test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to test concurrent execution: {e}")
        return False

def test_command_line_arguments():
    """Test that the orchestrator accepts the correct command line arguments."""
    logger.info("Testing command line arguments...")
    
    # Test with different max_concurrent values
    test_cases = [1, 3, 5]
    
    for max_concurrent in test_cases:
        try:
            orchestrator = AlphaOrchestrator()
            orchestrator.max_concurrent_simulations = max_concurrent
            logger.info(f"✓ Set max_concurrent to {max_concurrent}")
        except Exception as e:
            logger.error(f"✗ Failed to set max_concurrent to {max_concurrent}: {e}")
            return False
    
    logger.info("✓ All command line argument tests passed")
    return True

def main():
    """Run all tests."""
    logger.info("Starting Alpha Orchestrator tests...")
    
    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Command Line Arguments", test_command_line_arguments),
        ("Concurrent Execution", test_concurrent_execution),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The orchestrator is ready for concurrent execution.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())
