import requests
import json
import time
import os

from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime, timedelta

from .log_manager import setup_logger
logger = setup_logger(__name__, "submitter")

class ImprovedAlphaSubmitter:
    def __init__(self, daily_limit: int = 10):
        from .api_session import get_session_manager
        self._api = get_session_manager()
        self.sess = self._api.session
        self.sess.timeout = (30, 300)

    def _reconnect(self) -> None:
        logger.warning("Reconnecting session due to connection error...")
        self._api._apply_proxy()
        self._api._authenticate()

    def _fix_proxy(self) -> None:
        self._api._apply_proxy()

    def setup_auth(self) -> None:
        self._api.ensure_authenticated()

    def check_hopeful_alphas_count(self, min_count: int = 50) -> bool:
        from .alpha_db import get_alpha_db
        db = get_alpha_db()
        count = db.count_alphas()
        logger.info(f"Found {count} hopeful alphas in alpha/ directory")
        if count >= min_count:
            logger.info(f"Sufficient hopeful alphas ({count} >= {min_count}), proceeding with submission")
            return True
        else:
            logger.info(f"Insufficient hopeful alphas ({count} < {min_count}), skipping submission")
            return False

    def load_hopeful_alphas(self) -> List[Dict]:
        from .alpha_db import get_alpha_db
        try:
            db = get_alpha_db()
            alphas = db.get_all_alphas()
            logger.info(f"Loaded {len(alphas)} hopeful alphas from SQLite")
            return alphas
        except Exception as e:
            logger.error(f"Error loading hopeful alphas: {str(e)}")
            return []

    def fetch_successful_alphas(self, offset: int = 0, limit: int = 10) -> Dict:
        """Fetch successful unsubmitted alphas with good performance metrics."""
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": limit,
            "offset": offset,
            "status": "UNSUBMITTED",
            "is.fitness>": 1,
            "is.sharpe>": 1.25,
            "order": "-dateCreated",
            "hidden": "false"
        }
        
        logger.info(f"Fetching alphas with params: {params}")
        full_url = f"{url}?{'&'.join(f'{k}={v}' for k,v in params.items())}"
        logger.info(f"Request URL: {full_url}")
        
        max_retries = 5
        base_delay = 30
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries} to fetch alphas")
                response = self.sess.get(url, params=params, timeout=(15, 60))
                logger.info(f"Response URL: {response.url}")
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                    self._api.report_rate_limit(wait_time)
                    logger.info(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                logger.info(f"Successfully fetched {len(data.get('results', []))} alphas. Total count: {data.get('count', 0)}")
                return data
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch alphas after {max_retries} attempts due to timeouts")
                    return {"count": 0, "results": []}
                    
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                logger.warning(f"SSL/Connection error on attempt {attempt + 1}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    time.sleep(5)
                else:
                    logger.error(f"Failed to fetch alphas after {max_retries} attempts due to SSL/Connection errors")
                    return {"count": 0, "results": []}
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch alphas after {max_retries} attempts. Last error: {e}")
                    return {"count": 0, "results": []}
        
        return {"count": 0, "results": []}

    def monitor_submission(self, alpha_id: str, max_timeout_minutes: int = 15) -> Dict:
        """Monitor submission status with improved timeout handling."""
        url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/submit"
        
        start_time = time.time()
        max_timeout_seconds = max_timeout_minutes * 60
        base_sleep_time = 5
        max_sleep_time = 60
        
        attempt = 0
        
        while (time.time() - start_time) < max_timeout_seconds:
            attempt += 1
            elapsed_minutes = (time.time() - start_time) / 60
            
            try:
                logger.info(f"Monitoring attempt {attempt} for alpha {alpha_id} (elapsed: {elapsed_minutes:.1f} minutes)")
                response = self.sess.get(url)
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 404:
                    logger.info(f"Alpha {alpha_id} already submitted or not found")
                    return {"status": "already_submitted", "alpha_id": alpha_id}
                
                if response.status_code != 200:
                    logger.error(f"Submission failed for alpha {alpha_id}")
                    logger.error(f"Response status: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    return {"status": "failed", "error": response.text, "alpha_id": alpha_id}
                
                # If response is empty (still submitting)
                if not response.text.strip():
                    logger.info(f"Alpha {alpha_id} still being submitted, waiting...")
                    # Exponential backoff with cap
                    sleep_time = min(base_sleep_time * (1.5 ** (attempt - 1)), max_sleep_time)
                    time.sleep(sleep_time)
                    continue
                
                # Try to parse JSON response (submission complete)
                try:
                    data = response.json()
                    logger.info(f"Submission complete for alpha {alpha_id}")
                    return {"status": "success", "data": data, "alpha_id": alpha_id}
                except json.JSONDecodeError:
                    logger.info(f"Response not in JSON format yet for alpha {alpha_id}, continuing to monitor...")
                    sleep_time = min(base_sleep_time * (1.5 ** (attempt - 1)), max_sleep_time)
                    time.sleep(sleep_time)
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on monitoring attempt {attempt} for alpha {alpha_id}: {str(e)}")
                if (time.time() - start_time) < max_timeout_seconds:
                    sleep_time = min(base_sleep_time * (2 ** attempt), max_sleep_time)
                    logger.info(f"Waiting {sleep_time} seconds before retry...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Monitoring timed out for alpha {alpha_id} after {max_timeout_minutes} minutes")
                    return {"status": "timeout", "error": "Monitoring timed out", "alpha_id": alpha_id}
                    
            except Exception as e:
                logger.warning(f"Monitor attempt {attempt} failed for alpha {alpha_id}: {str(e)}")
                if (time.time() - start_time) < max_timeout_seconds:
                    sleep_time = min(base_sleep_time * (1.5 ** attempt), max_sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Monitoring failed for alpha {alpha_id} after {max_timeout_minutes} minutes")
                    return {"status": "error", "error": str(e), "alpha_id": alpha_id}
        
        logger.error(f"Monitoring timed out for alpha {alpha_id} after {max_timeout_minutes} minutes")
        return {"status": "timeout", "error": "Monitoring timed out", "alpha_id": alpha_id}

    def log_submission_result(self, alpha_id: str, result: Dict) -> None:
        """Log submission result to file."""
        log_file = 'submission_results.json'
        
        # Load existing results
        existing_results = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {log_file}, starting fresh")
        
        # Add new result
        entry = {
            "alpha_id": alpha_id,
            "timestamp": int(time.time()),
            "datetime": datetime.now().isoformat(),
            "result": result
        }
        existing_results.append(entry)
        
        # Save updated results
        with open(log_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        logger.info(f"Logged submission result for alpha {alpha_id}")

    def has_fail_checks(self, alpha: Dict) -> bool:
        """Check if alpha has any FAIL results in checks."""
        checks = alpha.get("checks", [])
        return any(check.get("result") == "FAIL" for check in checks)

    def _mark_local_submitted(self, alpha_id: str):
        """Update the local DB lifecycle_state to SUBMITTED."""
        try:
            from .alpha_db import get_alpha_db
            from .alpha_lifecycle import AlphaState
            db = get_alpha_db()
            db.mark_all_submitted_in_batch([alpha_id])
        except Exception as e:
            logger.debug(f"Could not update local state for {alpha_id}: {e}")

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit a single alpha and monitor its status."""
        url = f"https://api.worldquantbrain.com/alphas/{alpha_id}/submit"
        logger.info(f"Submitting alpha {alpha_id}")
        logger.info(f"Request URL: {url}")
        
        max_retries = 3
        base_delay = 10
        
        for attempt in range(max_retries):
            try:
                # Initial submission
                response = self.sess.post(url)
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 201:
                    logger.info(f"Successfully submitted alpha {alpha_id}, monitoring status...")

                    # Monitor submission status with longer timeout
                    result = self.monitor_submission(alpha_id, max_timeout_minutes=20)
                    if result:
                        self.log_submission_result(alpha_id, result)
                        if result.get("status") in ["success", "already_submitted"]:
                            self._mark_local_submitted(alpha_id)
                            return True
                        else:
                            logger.error(f"Submission failed for alpha {alpha_id}: {result.get('error', 'Unknown error')}")
                            return False
                    else:
                        logger.error(f"Submission monitoring failed for alpha {alpha_id}")
                        return False
                        
                elif response.status_code == 409:
                    logger.info(f"Alpha {alpha_id} already submitted")
                    return True
                    
                else:
                    logger.error(f"Failed to submit alpha {alpha_id}. Status: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
                    
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        return False
                        
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on submission attempt {attempt + 1} for alpha {alpha_id}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Submission timed out for alpha {alpha_id} after {max_retries} attempts")
                    return False
                    
            except Exception as e:
                logger.error(f"Error submitting alpha {alpha_id} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.exception("Full traceback:")
                    return False
        
        return False

    def submit_hopeful_alphas(self, batch_size: int = 3) -> None:
        """Submit hopeful alphas from local DB with quality filtering and quota awareness."""
        logger.info(f"Starting hopeful alphas submission with batch size {batch_size}")

        from .submission_quota import get_submission_quota
        quota = get_submission_quota()
        if not quota.can_submit():
            logger.info(f"Daily quota exhausted ({quota.count_today()}/{quota.daily_limit}), skipping")
            return

        # Load hopeful alphas
        hopeful_alphas = self.load_hopeful_alphas()
        if not hopeful_alphas:
            logger.info("No hopeful alphas to process")
            return

        # Filter: no FAIL checks AND meet quality thresholds
        valid_alphas = [
            alpha for alpha in hopeful_alphas
            if not self.has_fail_checks(alpha)
        ]
        logger.info(f"Found {len(valid_alphas)} alphas without FAIL checks")

        # Sort by fitness DESC so we submit the best alphas first
        valid_alphas.sort(
            key=lambda a: (
                a.get("backtest", {}).get("fitness")
                or a.get("fitness")
                or 0
            ),
            reverse=True,
        )

        if not valid_alphas:
            logger.info("No valid alphas to submit")
            return

        # Submit valid alphas in batches
        total_submitted = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        for i in range(0, len(valid_alphas), batch_size):
            batch = valid_alphas[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(valid_alphas)-1)//batch_size + 1} ({len(batch)} alphas)")
            
            batch_successes = 0
            for alpha in batch:
                basic = alpha.get("basic", {})
                backtest = alpha.get("backtest", {})
                alpha_id = basic.get("alpha_id") or alpha.get("alpha_id")
                if not alpha_id:
                    logger.warning("Alpha missing alpha_id, skipping")
                    continue
                
                expression = basic.get("expression", alpha.get("expression", "Unknown"))
                metrics = (f"Sharpe: {backtest.get('sharpe', alpha.get('sharpe', 'N/A'))}, "
                         f"Fitness: {backtest.get('fitness', alpha.get('fitness', 'N/A'))}")
                logger.info(f"Submitting alpha {alpha_id}:")
                logger.info(f"Expression: {expression}")
                logger.info(f"Metrics: {metrics}")

                if self.submit_alpha(alpha_id):
                    batch_successes += 1
                    total_submitted += 1
                    consecutive_failures = 0
                    quota.record_submission(alpha_id)
                    if not quota.can_submit():
                        logger.info(f"Daily quota reached ({quota.daily_limit}), stopping")
                        break
                else:
                    consecutive_failures += 1

                # Wait between submissions to avoid rate limiting
                time.sleep(30)

            if not quota.can_submit():
                break

            if batch_successes == 0:
                consecutive_failures += 1
                logger.warning(f"Batch failed. Consecutive failures: {consecutive_failures}")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping submission")
                    break
            else:
                consecutive_failures = 0

            # Wait between batches
            if i + batch_size < len(valid_alphas):
                logger.info(f"Waiting 120 seconds before next batch...")
                time.sleep(120)

        logger.info(f"Hopeful alphas submission complete. Total alphas submitted: {total_submitted}")
        
        # Clean up hopeful_alphas.json after successful submission
        if total_submitted > 0:
            self.cleanup_hopeful_alphas()

    def cleanup_hopeful_alphas(self):
        # We don't delete from SQLite, but we could mark them as submitted if needed.
        logger.info("Alpha cleanup requested. No-op for SQLite (data preserved).")

    def mark_alpha_green(self, alpha_id: str) -> bool:
        url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
        try:
            response = self.sess.patch(url, json={"color": "green"})
            if response.status_code in (200, 204):
                logger.info(f"Marked alpha {alpha_id} as GREEN")
                return True
            else:
                logger.warning(f"Failed to mark alpha {alpha_id} green: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error marking alpha {alpha_id} green: {e}")
            return False

    def mark_submittable_alphas_green(self) -> int:
        marked = 0
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": 50,
            "offset": 0,
            "status": "UNSUBMITTED",
            "is.fitness>": 1,
            "is.sharpe>": 1.25,
            "order": "-is.fitness",
            "hidden": "false",
        }

        while True:
            try:
                response = self.sess.get(url, params=params, timeout=(15, 60))
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 30))
                    self._api.report_rate_limit(retry_after)
                    logger.info(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                if response.status_code != 200:
                    logger.error(f"Failed to fetch alphas: {response.status_code}")
                    break

                data = response.json()
                results = data.get("results", [])

                for alpha in results:
                    alpha_id = alpha.get("id")
                    checks = alpha.get("is", {}).get("checks", [])
                    has_fail = any(c.get("result") == "FAIL" for c in checks)

                    if not has_fail and alpha_id:
                        current_color = alpha.get("color")
                        if current_color != "green":
                            if self.mark_alpha_green(alpha_id):
                                marked += 1
                            time.sleep(1)

                if not data.get("next") or len(results) < params["limit"]:
                    break
                params["offset"] += params["limit"]

            except Exception as e:
                logger.error(f"Error in mark_submittable_alphas_green: {e}")
                break

        logger.info(f"Marked {marked} submittable alphas as green")
        return marked

    def batch_submit(self, batch_size: int = 3) -> None:
        """Submit alphas in batches with improved error handling."""
        logger.info(f"Starting batch submission with batch size {batch_size}")

        from .submission_quota import get_submission_quota
        quota = get_submission_quota()
        if not quota.can_submit():
            logger.info(f"Daily quota exhausted ({quota.count_today()}/{quota.daily_limit}), skipping")
            return

        # Mark submittable alphas green before submitting
        try:
            self.mark_submittable_alphas_green()
        except Exception as e:
            logger.warning(f"mark_submittable_alphas_green failed (non-fatal): {e}")

        offset = 0
        total_submitted = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            logger.info(f"Fetching batch at offset {offset}")
            response = self.fetch_successful_alphas(offset=offset, limit=batch_size)
            
            if not response or not response.get("results"):
                logger.info("No more alphas to process")
                break
                
            results = response["results"]
            if not results:
                logger.info("Empty results batch")
                break
                
            logger.info(f"Processing batch of {len(results)} alphas...")
            
            # Filter out alphas with FAIL checks
            valid_alphas = [alpha for alpha in results if not self.has_fail_checks(alpha)]
            logger.info(f"Found {len(valid_alphas)} valid alphas after filtering FAILs")
            
            if not valid_alphas:
                logger.info("No valid alphas in this batch, moving to next")
                offset += batch_size
                continue
            
            # Submit valid alphas sequentially to avoid overwhelming the API
            batch_successes = 0
            for alpha in valid_alphas:
                alpha_id = alpha["id"]
                expression = alpha["regular"]["code"]
                metrics = (f"Sharpe: {alpha['is']['sharpe']}, "
                         f"Fitness: {alpha['is']['fitness']}")
                logger.info(f"Submitting alpha {alpha_id}:")
                logger.info(f"Expression: {expression}")
                logger.info(f"Metrics: {metrics}")

                if self.submit_alpha(alpha_id):
                    batch_successes += 1
                    total_submitted += 1
                    consecutive_failures = 0
                    quota.record_submission(alpha_id)
                    if not quota.can_submit():
                        logger.info(f"Daily quota reached ({quota.daily_limit}), stopping")
                        break
                else:
                    consecutive_failures += 1

                # Wait between submissions to avoid rate limiting
                time.sleep(30)

            if not quota.can_submit():
                break

            if batch_successes == 0:
                consecutive_failures += 1
                logger.warning(f"Batch failed. Consecutive failures: {consecutive_failures}")

                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping submission")
                    break
            else:
                consecutive_failures = 0

            if not response.get("next"):
                logger.info("No more pages to process")
                break

            offset += batch_size
            logger.info(f"Waiting 120 seconds before next batch...")
            time.sleep(120)

        logger.info(f"Submission process complete. Total alphas submitted: {total_submitted}")

def main():
    parser = argparse.ArgumentParser(description='Submit successful alphas to WorldQuant Brain with improved timeout handling')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of alphas to submit per batch (default: 3)')
    parser.add_argument('--interval-hours', type=int, default=24,
                      help='Hours to wait between submission runs (default: 24)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--auto-mode', action='store_true',
                      help='Run in automated mode (single run, no continuous loop)')
    parser.add_argument('--timeout-minutes', type=int, default=20,
                      help='Maximum timeout for submission monitoring in minutes (default: 20)')
    parser.add_argument('--min-hopeful-count', type=int, default=50,
                      help='Minimum count of hopeful alphas required to start submission (default: 50)')
    parser.add_argument('--use-hopeful-file', action='store_true',
                      help='Use hopeful_alphas.json file instead of fetching from API')
    
    args = parser.parse_args()
    
    import logging
    from .log_manager import setup_logger
    setup_logger(__name__, "submitter", level=getattr(logging, args.log_level))
    
    interval_seconds = args.interval_hours * 3600
    
    try:
        if args.auto_mode:
            logger.info(f"Starting single submission run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if not args.use_hopeful_file:
                submitter = ImprovedAlphaSubmitter()
                submitter.batch_submit(batch_size=args.batch_size)
            else:
                submitter = ImprovedAlphaSubmitter()
                if submitter.check_hopeful_alphas_count(args.min_hopeful_count):
                    submitter.submit_hopeful_alphas(batch_size=args.batch_size)
                else:
                    logger.info("Insufficient hopeful alphas, skipping submission")
            
            logger.info("Single submission run complete")
        else:
            while True:
                logger.info(f"Starting submission run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                try:
                    submitter = ImprovedAlphaSubmitter()
                    
                    if args.use_hopeful_file:
                        if submitter.check_hopeful_alphas_count(args.min_hopeful_count):
                            submitter.submit_hopeful_alphas(batch_size=args.batch_size)
                        else:
                            logger.info("Insufficient hopeful alphas, skipping submission")
                    else:
                        submitter.batch_submit(batch_size=args.batch_size)
                    
                    logger.info(f"Submission run complete. Waiting {args.interval_hours} hours before next run...")
                except Exception as e:
                    logger.error(f"Error during submission run: {str(e)}")
                    logger.exception("Full traceback:")
                
                # Sleep until next run
                next_run = time.time() + interval_seconds
                next_run_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(next_run))
                logger.info(f"Next run scheduled for: {next_run_time}")
                time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, exiting gracefully...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
