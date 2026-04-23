import argparse
import json
import os
import time
from typing import List
import requests
from requests.auth import HTTPBasicAuth

from core.log_manager import setup_logger
logger = setup_logger(__name__, "miner")


def auth_session() -> requests.Session:
    from core.config import load_credentials, fix_session_proxy
    u, p = load_credentials()
    s = requests.Session()
    fix_session_proxy(s)
    s.auth = HTTPBasicAuth(u, p)
    r = s.post('https://api.worldquantbrain.com/authentication')
    if r.status_code != 201:
        raise RuntimeError(f"Auth failed: {r.text}")
    return s


def submit_sim(s: requests.Session, expression: str) -> dict:
    data = {
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
        'regular': expression
    }
    r = s.post('https://api.worldquantbrain.com/simulations', json=data)
    if r.status_code != 201:
        return {'status': 'error', 'message': r.text}
    progress_url = r.headers.get('location')
    return {'status': 'success', 'progress_url': progress_url}


def monitor_sim(s: requests.Session, progress_url: str, timeout_s: int = 3600) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        r = s.get(progress_url)
        if r.status_code == 429:
            time.sleep(5)
            continue
        if r.status_code != 200:
            return {'status': 'error', 'message': r.text}
        data = r.json()
        st = data.get('status')
        if st == 'COMPLETE':
            alpha_id = data.get('alpha')
            if not alpha_id:
                return {'status': 'error', 'message': 'missing alpha id'}
            a = s.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
            if a.status_code != 200:
                return {'status': 'error', 'message': a.text}
            return {'status': 'complete', 'alpha': a.json()}
        if st == 'ERROR':
            return {'status': 'error', 'message': 'simulation error'}
        time.sleep(5)
    return {'status': 'timeout'}


def generate_templates() -> List[str]:
    wins = [20, 60, 120, 252]
    ranks = [5, 20]
    exprs: List[str] = []
    for w in wins:
        exprs.append(f'group_neutralize(zscore(ts_mean(returns, {w})), sector)')
    for w1 in wins:
        for w2 in wins:
            if w2 > w1:
                exprs.append(f'group_neutralize(zscore(ts_mean(returns, {w1}) - ts_mean(returns, {w2})), sector)')
    for w in wins:
        for r in ranks:
            exprs.append(f'group_neutralize(zscore(ts_rank(ts_mean(returns, {w}), {r})), sector)')
    exprs.append('group_neutralize(zscore(rank(divide(revenue, assets))), sector)')
    return exprs


def main():
    parser = argparse.ArgumentParser(description='Template+Grid Alpha Miner')
    parser.add_argument('--max', type=int, default=30, help='Max expressions to test')
    parser.add_argument('--timeout', type=int, default=3600, help='Per-simulation timeout (s)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    s = auth_session()
    exprs = generate_templates()[:args.max]
    results = []

    for i, expr in enumerate(exprs, 1):
        logger.info(f'Testing {i}/{len(exprs)}: {expr}')
        sub = submit_sim(s, expr)
        if sub.get('status') != 'success':
            logger.warning(f"Submit error: {sub.get('message')}")
            continue
        mon = monitor_sim(s, sub['progress_url'], timeout_s=args.timeout)
        if mon.get('status') == 'complete':
            alpha = mon['alpha']
            is_data = alpha.get('is', {})
            fitness = is_data.get('fitness')
            sharpe = is_data.get('sharpe')
            logger.info(f"Completed. Fitness={fitness}, Sharpe={sharpe}")
            # Append to hopeful if strong
            if fitness is not None and fitness > 1:
                try:
                    from core.alpha_store import save_alpha
                    save_alpha(expr, alpha, source="template_grid")
                except Exception as e:
                    logger.error(f'Failed to save alpha: {e}')
        elif mon.get('status') == 'timeout':
            logger.warning('Monitoring timeout')
        else:
            logger.warning(f"Simulation failed: {mon.get('message')}")
        results.append({'expression': expr, 'result': mon})

    with open(os.path.join('results', 'template_grid_results.json'),'w') as f:
        json.dump(results, f, indent=2)
    logger.info('Template+Grid mining complete')


if __name__ == '__main__':
    main()


