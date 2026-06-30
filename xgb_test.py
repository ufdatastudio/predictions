import sys
import xgboost as xgb

print(f"Python Version: {sys.version}")
print(f"XGBoost Version: {xgb.__version__}")

try:
    # Attempt a mini GPU training task to capture the crash logs
    import numpy as np
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    
    test_model = xgb.XGBClassifier(device='cuda', tree_method='hist')
    test_model.fit(X, y)
    print("XGBoost GPU configuration is working fine!")
except Exception as e:
    print("\n--- DETECTED CUDA ERROR ---")
    print(str(e))
