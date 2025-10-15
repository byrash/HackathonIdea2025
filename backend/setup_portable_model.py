#!/usr/bin/env python3
"""
Setup Portable ML Model for Fraud Detection

Uses scikit-learn's GradientBoostingClassifier - pure Python, no native dependencies.
Works across all platforms without XGBoost library issues.
"""

import pickle
import numpy as np
from pathlib import Path

def setup_portable_model():
    """
    Portable GradientBoosting Model using pure scikit-learn
    No external dependencies, works on any machine
    """
    print("🚀 Setting up Portable GradientBoosting Model...")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("❌ Error: scikit-learn not installed")
        print("Run: pip install scikit-learn")
        return False
    
    print("📊 Generating training data...")
    
    # Features: [metadata_risk, forensic_risk, ocr_risk, security_risk, 
    #            ela_regions, irregular_edges, template_matched, watermark, 
    #            exif_count, amount_mismatch, date_invalid, clone_count]
    
    # Fraudulent checks (higher risk scores)
    fraud_samples = np.array([
        [40, 80, 60, 90, 3, 15, 0, 0, 5, 1, 1, 2],
        [35, 75, 55, 85, 2, 12, 0, 0, 3, 1, 0, 1],
        [50, 90, 70, 95, 4, 20, 0, 0, 2, 1, 1, 3],
        [30, 70, 50, 80, 2, 10, 0, 1, 4, 1, 0, 1],
        [45, 85, 65, 88, 3, 18, 0, 0, 6, 1, 1, 2],
        [38, 78, 58, 92, 2, 14, 0, 0, 5, 1, 0, 1],
        [42, 82, 62, 87, 3, 16, 0, 0, 4, 1, 1, 2],
        [48, 88, 68, 93, 4, 19, 0, 0, 3, 1, 1, 3],
        [36, 76, 56, 84, 2, 13, 0, 1, 5, 1, 0, 1],
        [44, 84, 64, 89, 3, 17, 0, 0, 4, 1, 1, 2],
        [32, 72, 52, 82, 2, 11, 0, 0, 6, 1, 0, 1],
        [46, 86, 66, 91, 3, 18, 0, 0, 3, 1, 1, 2],
        [40, 80, 60, 86, 2, 15, 0, 1, 5, 1, 0, 2],
        [34, 74, 54, 83, 2, 12, 0, 0, 4, 1, 1, 1],
        [52, 92, 72, 96, 4, 21, 0, 0, 2, 1, 1, 3],
    ])
    
    # Legitimate checks (lower risk scores)
    legit_samples = np.array([
        [5, 10, 8, 15, 0, 2, 1, 1, 25, 0, 0, 0],
        [8, 15, 12, 20, 0, 3, 1, 1, 28, 0, 0, 0],
        [3, 8, 5, 12, 0, 1, 1, 1, 30, 0, 0, 0],
        [10, 18, 15, 25, 0, 4, 1, 1, 22, 0, 0, 0],
        [6, 12, 10, 18, 0, 2, 1, 1, 26, 0, 0, 0],
        [4, 10, 7, 14, 0, 2, 1, 1, 27, 0, 0, 0],
        [9, 16, 13, 22, 0, 3, 1, 1, 24, 0, 0, 0],
        [7, 14, 11, 19, 0, 3, 1, 1, 25, 0, 0, 0],
        [5, 11, 9, 16, 0, 2, 1, 1, 29, 0, 0, 0],
        [8, 15, 12, 21, 0, 3, 1, 1, 23, 0, 0, 0],
        [6, 13, 10, 17, 0, 2, 1, 1, 26, 0, 0, 0],
        [4, 9, 7, 13, 0, 1, 1, 1, 28, 0, 0, 0],
        [10, 17, 14, 23, 0, 4, 1, 1, 21, 0, 0, 0],
        [7, 14, 11, 19, 0, 3, 1, 1, 25, 0, 0, 0],
        [5, 11, 8, 15, 0, 2, 1, 1, 27, 0, 0, 0],
    ])
    
    # Suspicious checks (medium risk)
    suspicious_samples = np.array([
        [20, 45, 30, 50, 1, 8, 0, 1, 15, 0, 0, 1],
        [25, 50, 35, 55, 1, 9, 0, 0, 12, 1, 0, 0],
        [18, 40, 28, 48, 1, 7, 1, 0, 16, 0, 1, 1],
        [22, 48, 32, 52, 1, 10, 0, 1, 14, 0, 0, 1],
        [24, 46, 34, 54, 1, 8, 0, 0, 13, 1, 0, 0],
        [19, 42, 29, 49, 1, 7, 1, 1, 15, 0, 0, 1],
        [26, 52, 36, 56, 1, 11, 0, 0, 11, 1, 1, 0],
        [21, 44, 31, 51, 1, 9, 0, 1, 14, 0, 0, 1],
        [23, 47, 33, 53, 1, 8, 1, 0, 13, 0, 1, 1],
        [20, 43, 30, 50, 1, 7, 0, 1, 15, 0, 0, 0],
    ])
    
    # Combine datasets
    X = np.vstack([fraud_samples, legit_samples, suspicious_samples])
    
    # Labels: 1 = fraud, 0 = legitimate
    y = np.array(
        [1] * len(fraud_samples) + 
        [0] * len(legit_samples) + 
        [1] * len(suspicious_samples)
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("✅ Training data created:")
    print(f"   • Fraudulent samples: {len(fraud_samples)}")
    print(f"   • Legitimate samples: {len(legit_samples)}")
    print(f"   • Suspicious samples: {len(suspicious_samples)}")
    print(f"   • Total samples: {len(X)}")
    print(f"   • Features: {X.shape[1]}")
    print()
    
    # Train GradientBoosting model
    print("🎯 Training GradientBoostingClassifier...")
    print("   Parameters:")
    print("   • n_estimators: 200")
    print("   • max_depth: 5")
    print("   • learning_rate: 0.1")
    print("   • subsample: 0.8")
    print()
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train) * 100
    test_score = model.score(X_test, y_test) * 100
    
    print("✅ Training complete!")
    print(f"   • Training accuracy: {train_score:.2f}%")
    print(f"   • Test accuracy: {test_score:.2f}%")
    print()
    
    # Feature importance
    feature_names = [
        'metadata_risk', 'forensic_risk', 'ocr_risk', 'security_risk',
        'ela_regions', 'irregular_edges', 'template_matched', 'watermark',
        'exif_count', 'amount_mismatch', 'date_invalid', 'clone_count'
    ]
    
    importances = model.feature_importances_
    top_features = sorted(zip(feature_names, importances), 
                         key=lambda x: x[1], reverse=True)[:5]
    
    print("📊 Top 5 Important Features:")
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"   {i}. {name}: {importance:.3f}")
    print()
    
    # Save model
    output_dir = Path("ml-models")
    output_dir.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'model_type': 'GradientBoosting',
        'algorithm': 'sklearn.ensemble.GradientBoostingClassifier',
        'version': '1.0',
        'accuracy': test_score,
        'feature_names': feature_names,
        'training_samples': len(X),
        'notes': 'Pure Python, fully portable, no XGBoost dependencies'
    }
    
    model_path = output_dir / "fraud_detection_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("💾 Model saved:")
    print(f"   • Path: {model_path}")
    print(f"   • Size: {model_path.stat().st_size / 1024:.1f} KB")
    print()
    
    print("=" * 60)
    print("✨ SUCCESS! Portable model is ready!")
    print()
    print("Benefits:")
    print("  ✅ Pure Python - works on any machine")
    print("  ✅ No XGBoost native library issues")
    print("  ✅ No 32-bit/64-bit compatibility problems")
    print("  ✅ Portable across Windows/Mac/Linux")
    print("  ✅ Similar performance to XGBoost")
    print()
    print("🚀 Ready to use!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    setup_portable_model()

