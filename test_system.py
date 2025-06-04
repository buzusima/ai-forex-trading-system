"""
Quick test script to check system components
"""

import sys
import os


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import numpy as np

        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy - run: pip install numpy")

    try:
        import pandas as pd

        print("✅ Pandas")
    except ImportError:
        print("❌ Pandas - run: pip install pandas")

    try:
        import MetaTrader5 as mt5

        print("✅ MetaTrader5")
    except ImportError:
        print("❌ MetaTrader5 - run: pip install MetaTrader5")

    try:
        import tensorflow as tf

        print("✅ TensorFlow")
    except ImportError:
        print("❌ TensorFlow - run: pip install tensorflow")


def test_file_structure():
    """Test if required files exist"""
    print("\nTesting file structure...")

    required_files = [
        "main.py",
        "config/settings.py",
        "config/mt5_config.py",
        "brain/ensemble_predictor.py",
        "sensor/mt5_connector.py",
        "heart/risk_manager.py",
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")


def test_mt5_connection():
    """Test MT5 connection (basic)"""
    print("\nTesting MT5 connection...")

    try:
        import MetaTrader5 as mt5

        if mt5.initialize():
            print("✅ MT5 initialized successfully")

            # Get basic info
            account_info = mt5.account_info()
            if account_info:
                print(f"✅ Account: {account_info.login}")
            else:
                print("⚠️ MT5 initialized but no account info (need login)")

            mt5.shutdown()
        else:
            print("❌ MT5 initialization failed")
            print(f"Error: {mt5.last_error()}")

    except Exception as e:
        print(f"❌ MT5 test failed: {str(e)}")


if __name__ == "__main__":
    test_imports()
    test_file_structure()
    test_mt5_connection()
    print("\nTest completed! Fix any ❌ issues before running main.py")
