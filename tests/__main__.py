if __name__ == '__main__':
    import sys
    import unittest

    loader = unittest.TestLoader()
    tests = loader.discover('.')
    testRunner = unittest.runner.TextTestRunner()
    result = testRunner.run(tests)
    sys.exit(0 if result.wasSuccessful() else 1)
