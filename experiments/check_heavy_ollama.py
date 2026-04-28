import requests
import time

def test_heavy_prompt():
    seeds = [
        "rank(open - (ts_sum(vwap, 60) / 60)) * (-1 * abs(returns))",
        "group_neutralize(rank(ts_mean(mdl77_2adverint * volume, 5)), industry)",
        "-1 * rank(ts_covariance(rank(high), rank(volume), 60))",
        "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(returns)) * (abs(returns) > 0.5)"
    ]
    prompt = f"""
            Generate 10 WorldQuant alpha expressions based on these seeds:
            {seeds}
            
            Mutation rules:
            1. Use 'ts_mean' or 'ts_rank' instead of 'ts_sum'.
            2. Add 'rank()' around components.
            3. Use 'returns' and 'volume' as common fields.
            
            Output only the expressions, one per line.
            """

    ollama_data = {
        'model': 'qwen3.5:35b',
        'prompt': prompt,
        'stream': False,
    }

    print("Sending request to Ollama...")
    start = time.time()
    try:
        response = requests.post('http://localhost:11434/api/generate', json=ollama_data, timeout=600)
        end = time.time()
        print(f"Time taken: {end - start:.2f}s")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json().get('response')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_heavy_prompt()


