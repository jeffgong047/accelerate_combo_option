import pickle 
import pandas as pd
path = 'results.pkl'
with open(path, 'rb') as f:
    results = pickle.load(f)
for r in results:
    status_symbol = "✓" if r['status'] == 'Success' else "❌"
    print(f"{status_symbol} Market {r['market_idx']}: {r['status']}")
    if r['error']:
        print(f"   Error: {r['error']}")
