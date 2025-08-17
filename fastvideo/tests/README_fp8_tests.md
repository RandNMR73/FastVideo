# FP8 Quantization Tests

This directory contains comprehensive tests for the FP8 quantization components in FastVideo.

## Test Files

- **`test_fp8_quantization.py`** - Main test suite for FP8 quantization
- **`run_fp8_tests.py`** - Simple test runner script
- **`README_fp8_tests.md`** - This documentation file

## Test Coverage

The test suite covers the following components:

### 1. Utility Functions

- **`per_token_cast_to_fp8`** - Tests for per-token FP8 conversion
- **`per_block_cast_to_fp8`** - Tests for block-wise FP8 conversion

### 2. Core Classes

- **`FP8QuantizeMethod`** - Tests for the quantization method implementation
- **`FP8Config`** - Tests for the quantization configuration
- **`convert_model_to_fp8`** - Tests for model conversion utility

### 3. Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow testing
- **Edge Cases** - Boundary conditions and error handling
- **Mock Testing** - Isolated testing without external dependencies

## Running the Tests

### Option 1: Using the Test Runner Script

```bash
cd fastvideo/tests
python run_fp8_tests.py
```

### Option 2: Using pytest directly

```bash
cd fastvideo/tests
pytest test_fp8_quantization.py -v
```

### Option 3: Running specific test classes

```bash
# Run only utility function tests
pytest test_fp8_quantization.py::TestPerTokenCastToFP8 -v

# Run only configuration tests
pytest test_fp8_quantization.py::TestFP8Config -v

# Run only quantization method tests
pytest test_fp8_quantization.py::TestFP8QuantizeMethod -v
```

### Option 4: Running specific test methods

```bash
# Run a specific test method
pytest test_fp8_quantization.py::TestFP8QuantizeMethod::test_apply_with_existing_fp8_weights -v

# Run tests matching a pattern
pytest test_fp8_quantization.py -k "test_apply" -v
```

## Test Dependencies

The tests use the following testing libraries:

- **pytest** - Main testing framework
- **torch** - PyTorch for tensor operations
- **unittest.mock** - Mocking and patching for isolated testing

## Mock Strategy

The tests use extensive mocking to:

1. **Isolate Components** - Test individual functions without external dependencies
2. **Avoid CUDA Requirements** - Mock `deep_gemm` to run tests on CPU
3. **Control Dependencies** - Mock base classes and external modules
4. **Test Error Conditions** - Simulate various failure scenarios

## Key Test Patterns

### 1. Shape and Type Validation

```python
def test_basic_functionality(self):
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    fp8_data, scale = per_token_cast_to_fp8(x)

    # Check output shapes
    assert fp8_data.shape == x.shape
    # Check data types
    assert fp8_data.dtype == torch.float8_e4m3fn
```

### 2. Error Handling

```python
def test_invalid_input(self):
    x = torch.randn(128, dtype=torch.bfloat16)  # 1D tensor
    with pytest.raises(AssertionError):
        per_token_cast_to_fp8(x)
```

### 3. Mocking External Dependencies

```python
@patch('fastvideo.layers.quantization.fp8_config.deep_gemm')
def test_apply_method(self, mock_deep_gemm):
    # Mock the deep_gemm function
    mock_deep_gemm.gemm_fp8_fp8_bf16_nt.return_value = None
    # ... test implementation
```

## Test Data

The tests use synthetic data:

- **Random Tensors** - `torch.randn()` for realistic input data
- **Edge Cases** - Zero-sized tensors, exact multiples of block size
- **Various Shapes** - Different tensor dimensions to test padding logic

## Expected Test Results

### Successful Tests

- ✅ All utility functions work correctly
- ✅ FP8 conversion maintains tensor shapes
- ✅ Scale factors are within expected ranges
- ✅ Error conditions are properly handled
- ✅ Mock dependencies work as expected

### Known Limitations

- **CUDA Tests** - Some tests are skipped on non-CUDA systems
- **DeepGEMM** - Actual GEMM operations are mocked
- **Hardware Dependencies** - FP8 operations require specific GPU capabilities

## Adding New Tests

To add new tests:

1. **Follow Naming Convention** - Use descriptive test method names
2. **Add Documentation** - Include docstrings explaining test purpose
3. **Use Appropriate Mocks** - Mock external dependencies
4. **Test Edge Cases** - Include boundary conditions
5. **Validate Outputs** - Check shapes, types, and values

### Example Test Structure

```python
def test_new_feature(self):
    """Test description of what this test validates."""
    # Arrange - Set up test data and mocks
    test_data = create_test_data()

    # Act - Execute the function being tested
    result = function_under_test(test_data)

    # Assert - Validate the results
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
    # ... additional assertions
```

## Troubleshooting

### Common Issues

1. **Import Errors** - Ensure you're running from the correct directory
2. **Mock Failures** - Check that all external dependencies are properly mocked
3. **CUDA Errors** - Some tests may fail on non-CUDA systems
4. **Version Mismatches** - Ensure PyTorch version compatibility

### Debug Mode

Run tests with increased verbosity:

```bash
pytest test_fp8_quantization.py -v -s --tb=long
```

### Isolated Testing

Test individual components:

```bash
# Test only utility functions
pytest test_fp8_quantization.py::TestPerTokenCastToFP8 -v

# Test with specific markers
pytest test_fp8_quantization.py -m "not slow" -v
```

## Contributing

When contributing to the test suite:

1. **Maintain Coverage** - Ensure new features have corresponding tests
2. **Follow Patterns** - Use existing test patterns and conventions
3. **Update Documentation** - Keep this README current
4. **Test Locally** - Verify tests pass before submitting changes

## References

- [PyTorch Testing Guide](https://pytorch.org/docs/stable/testing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [FastVideo FP8 Guide](../FP8_QUANTIZATION_GUIDE.md)
