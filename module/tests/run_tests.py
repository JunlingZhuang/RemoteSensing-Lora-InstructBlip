#!/usr/bin/env python3
"""
Test runner for LoRA fine-tuning module.
Runs all test cases and provides comprehensive coverage report.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add module to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
from test_config import TestConfig, TestDefaultConfig
from test_data_loading import TestRSICapDataset, TestLoadRSICapData
from test_lora_model import TestLoRAInstructBLIP
from test_training import TestLoRATrainer, TestTrainerUtilities
from test_integration import TestLoRAIntegration, TestTrainingStateManagement
from test_inference import TestModelInferencer, TestInferenceDemo, TestUtilityFunctions, TestInferenceErrorHandling


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_start_time = {}
    
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time[test] = time.time()
        if self.verbosity > 1:
            print(f"\nRunning: {test._testMethodName} ({test.__class__.__name__})")
    
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.test_start_time.get(test, 0)
        if self.verbosity > 1:
            print(f"PASSED: {test._testMethodName} ({duration:.3f}s)")
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.test_start_time.get(test, 0)
        if self.verbosity > 1:
            print(f"ERROR: {test._testMethodName} ({duration:.3f}s)")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.test_start_time.get(test, 0)
        if self.verbosity > 1:
            print(f"FAILED: {test._testMethodName} ({duration:.3f}s)")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.test_start_time.get(test, 0)
        if self.verbosity > 1:
            print(f"SKIPPED: {test._testMethodName} ({duration:.3f}s) - {reason}")


class TestRunner:
    """Main test runner class"""
    
    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.results = None
    
    def add_test_modules(self):
        """Add all test modules to the suite"""
        test_classes = [
            # Configuration tests
            TestConfig,
            TestDefaultConfig,
            
            # Data loading tests
            TestRSICapDataset,
            TestLoadRSICapData,
            
            # Model tests
            TestLoRAInstructBLIP,
            
            # Training tests
            TestLoRATrainer,
            TestTrainerUtilities,
            
            # Integration tests
            TestLoRAIntegration,
            TestTrainingStateManagement,
            
            # Inference tests
            TestModelInferencer,
            TestInferenceDemo,
            TestUtilityFunctions,
            TestInferenceErrorHandling,
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            self.test_suite.addTests(tests)
    
    def run_tests(self, verbosity=2):
        """Run all tests with specified verbosity"""
        print("Starting LoRA Fine-tuning Module Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create custom test runner
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            resultclass=ColoredTextTestResult,
            stream=sys.stdout,
            buffer=True
        )
        
        # Run tests
        self.results = runner.run(self.test_suite)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print summary
        self.print_summary(execution_time)
        
        return self.results.wasSuccessful()
    
    def print_summary(self, execution_time):
        """Print test execution summary"""
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        total_tests = self.results.testsRun
        failures = len(self.results.failures)
        errors = len(self.results.errors)
        skipped = len(self.results.skipped)
        passed = total_tests - failures - errors - skipped
        
        print(f"Total Tests:    {total_tests}")
        print(f"Passed:         {passed}")
        print(f"Failed:         {failures}")
        print(f"Errors:         {errors}")
        print(f"Skipped:        {skipped}")
        print(f"Time:           {execution_time:.2f}s")
        
        if failures > 0 or errors > 0:
            print(f"\nOVERALL RESULT: FAILED")
            
            if failures > 0:
                print(f"\nFAILED TESTS:")
                for test, traceback in self.results.failures:
                    print(f"  - {test}")
            
            if errors > 0:
                print(f"\nERROR TESTS:")
                for test, traceback in self.results.errors:
                    print(f"  - {test}")
        else:
            print(f"\nOVERALL RESULT: SUCCESS")
        
        # Calculate success rate
        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
    
    def run_specific_test(self, test_pattern):
        """Run specific tests matching a pattern"""
        loader = unittest.TestLoader()
        
        # Load all tests
        all_tests = unittest.TestSuite()
        for test_class in [TestConfig, TestRSICapDataset, TestLoRAInstructBLIP, 
                          TestLoRATrainer, TestLoRAIntegration]:
            tests = loader.loadTestsFromTestCase(test_class)
            all_tests.addTests(tests)
        
        # Filter tests by pattern
        filtered_suite = unittest.TestSuite()
        for test_group in all_tests:
            for test in test_group:
                if test_pattern.lower() in test._testMethodName.lower():
                    filtered_suite.addTest(test)
        
        if filtered_suite.countTestCases() == 0:
            print(f"No tests found matching pattern: {test_pattern}")
            return False
        
        print(f"Running {filtered_suite.countTestCases()} tests matching '{test_pattern}'")
        
        runner = unittest.TextTestRunner(verbosity=2, resultclass=ColoredTextTestResult)
        result = runner.run(filtered_suite)
        
        return result.wasSuccessful()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning tests")
    parser.add_argument(
        "--pattern", "-p", 
        type=str, 
        help="Run only tests matching this pattern"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Quiet output (minimal)"
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 1
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0
    
    # Create test runner
    runner = TestRunner()
    
    if args.pattern:
        # Run specific tests
        success = runner.run_specific_test(args.pattern)
    else:
        # Run all tests
        runner.add_test_modules()
        success = runner.run_tests(verbosity=verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()