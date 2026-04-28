import argparse
import json
import logging as py_logging
import os
import time
from typing import List, Dict, cast
import requests
from requests.auth import HTTPBasicAuth

from core.log_manager import setup_logger
from core.region_config import get_region_config
from core.expression_compiler import ExpressionCompiler
from core.ast_validator import ASTValidator
from core.data_fetcher import OperatorFetcher, DataFieldFetcher

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


def submit_sim(s: requests.Session, expression: str, region_key: str = "USA") -> dict:
    region_config = get_region_config(region_key)
    data = {
        'type': 'REGULAR',
        'settings': region_config.to_simulation_settings(),
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


def generate_templates_from_compiler(
    compiler: ExpressionCompiler,
    field_ids: List[str],
    max_expressions: int = 500,
) -> List[str]:
    templates = ExpressionCompiler.get_default_templates()
    replacements: Dict[str, List[str]] = {
        "FIELD": [str(v) for v in field_ids[:30]],
        "FIELD2": [str(v) for v in field_ids[:20]],
        "WINDOW": ["5", "20", "60", "120", "252"],
        "WINDOW2": ["60", "120", "252"],
        "GROUP": ["sector", "industry"],
        "RANK": ["5", "20"],
        "OP": ["rank", "zscore", "log", "sqrt"],
    }
    expressions = compiler.compile_templates(templates, cast(Dict[str, List[str]], replacements), max_expressions=max_expressions)
    return expressions


def main():
    parser = argparse.ArgumentParser(description='Template+Grid Alpha Miner')
    parser.add_argument('--max', type=int, default=30, help='Max expressions to test')
    parser.add_argument('--timeout', type=int, default=3600, help='Per-simulation timeout (s)')
    parser.add_argument('--region', type=str, default='USA', help='Region key (USA, CHN, EUR, etc.)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    args = parser.parse_args()

    py_logging.basicConfig(level=getattr(py_logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    s = auth_session()

    operator_fetcher = OperatorFetcher(session=s)
    operators = operator_fetcher.fetch_operators()
    operator_names = [op.get("name", "") for op in operators if op.get("name")]

    data_field_fetcher = DataFieldFetcher(session=s)
    region_config = get_region_config(args.region)
    fields = data_field_fetcher.fetch_data_fields(
        region=region_config.region,
        universe=region_config.universe,
        delay=region_config.delay,
    )
    field_ids = [f.get("id", "") for f in fields if f.get("id")]

    compiler = ExpressionCompiler(
        operators=operator_names,
        fields=field_ids,
    )
    ast_validator = ASTValidator(known_operators=set(operator_names))

    exprs = generate_templates_from_compiler(compiler, field_ids, max_expressions=args.max * 5)
    valid_exprs, invalid = ast_validator.validate_batch(exprs)
    exprs = valid_exprs[:args.max]

    logger.info(f"Testing {len(exprs)} expressions (from {len(valid_exprs)} valid, {len(invalid)} rejected by AST)")

    results = []
    for i, expr in enumerate(exprs, 1):
        logger.info(f'Testing {i}/{len(exprs)}: {expr}')
        sub = submit_sim(s, expr, region_key=args.region)
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
            if fitness is not None and fitness > 1:
                try:
                    from core.alpha_db import get_alpha_db
                    get_alpha_db().save_alpha(expr, alpha, source="template_grid")
                except Exception as e:
                    logger.error(f'Failed to save alpha: {e}')
        elif mon.get('status') == 'timeout':
            logger.warning('Monitoring timeout')
        else:
            logger.warning(f"Simulation failed: {mon.get('message')}")
        results.append({'expression': expr, 'result': mon})

    os.makedirs('results', exist_ok=True)
    with open(os.path.join('results', 'template_grid_results.json'),'w') as f:
        json.dump(results, f, indent=2)
    logger.info('Template+Grid mining complete')


if __name__ == '__main__':
    main()
