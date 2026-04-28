from core.alpha_generator_ollama import AlphaGenerator
import json

def audit_cloud_alphas():
    gen = AlphaGenerator(region_key="USA")
    gen.initialize()

    # Fetch the 10 most recent alphas from the API directly
    url = "https://api.worldquantbrain.com/users/self/alphas"
    params = {
        "limit": 10,
        "offset": 0,
        "order": "-dateCreated",
    }

    response = gen.sess.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch: {response.text}")
        return

    alphas = response.json().get("results", [])
    print(f"\n--- CLOUD AUDIT: Top {len(alphas)} Recent Simulations ---\n")

    for a in alphas:
        a_id = a.get("id")
        # Get details for each to see the expression
        detail_resp = gen.sess.get(f"https://api.worldquantbrain.com/alphas/{a_id}")
        if detail_resp.status_code == 200:
            detail = detail_resp.json()
            # Try various keys for the formula/expression
            expr = detail.get("regular", {}).get("code") or detail.get("logic", {}).get("formula") or detail.get("code")
            sharpe = detail.get("is", {}).get("sharpe")
            fitness = detail.get("is", {}).get("fitness")
            print(f"ID: {a_id}")
            print(f"EXPR: {expr}")
            print(f"SHARPE: {sharpe} | FITNESS: {fitness}")
            print("-" * 50)

if __name__ == "__main__":
    audit_cloud_alphas()

