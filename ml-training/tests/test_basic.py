#!/usr/bin/env python3
"""
Basic test to verify pytest works in CI
"""

def test_basic_functionality():
    """Test that basic Python functionality works"""
    assert 1 + 1 == 2
    
def test_imports():
    """Test that basic imports work"""
    import os
    import sys
    import json
    assert True
    
def test_file_operations():
    """Test that file operations work"""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        assert os.path.exists(test_file)
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        assert content == "test"