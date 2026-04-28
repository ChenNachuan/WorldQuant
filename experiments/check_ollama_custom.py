from core.alpha_generator_ollama import AlphaGenerator

def test_ollama():
    gen = AlphaGenerator(region_key="USA")
    gen.initialize()
    # Let's modify the generator to log raw content for a moment
    prompt = "Generate 3 WorldQuant alpha expressions using rank and ts_mean. Return only the formulas."
    # I'll manually call the API here to see the raw output
    ollama_data = {
        'model': gen.model_name,
        'prompt': prompt,
        'stream': False,
    }
    response = gen.sess.post(f'{gen.ollama_url}/api/generate', json=ollama_data, timeout=300)
    print(f"Raw Response: {response.json().get('response')}")

if __name__ == "__main__":
    test_ollama()


