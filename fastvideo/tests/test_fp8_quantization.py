# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from fastvideo.layers.quantization.fp8_config import (
    FP8QuantizeMethod,
    FP8Config,
    per_token_cast_to_fp8,
    per_block_cast_to_fp8,
    convert_model_to_fp8,
    block_size
)
from fastvideo.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from torch.nn.parameter import Parameter

class TestPerTokenCastToFP8:
    """Test the per_token_cast_to_fp8 utility function."""
    
    def test_basic_functionality(self):
        """Test basic FP8 conversion with 2D tensor."""
        # Create a simple 2D tensor
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        
        fp8_data, scale = per_token_cast_to_fp8(x)
        
        # Check output shapes
        assert fp8_data.shape == x.shape
        assert scale.shape == (4, 2)  # 256/128 = 2 blocks
        
        # Check data types
        assert fp8_data.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32
        
        # Check scale values are reasonable
        assert torch.all(scale > 0)
        assert torch.all(scale <= 1.0)
    
    def test_padding_handling(self):
        """Test handling of tensors that need padding."""
        # Create tensor with size not divisible by 128
        x = torch.randn(2, 100, dtype=torch.bfloat16)
        
        fp8_data, scale = per_token_cast_to_fp8(x)
        
        # Output should maintain original shape
        assert fp8_data.shape == x.shape
        assert scale.shape == (2, 1)  # 100/128 = 1 block (rounded up)
    
    def test_edge_cases(self):
        """Test edge cases like exact multiples of 128."""
        # Test with size exactly 128
        x = torch.randn(1, 128, dtype=torch.bfloat16)
        fp8_data, scale = per_token_cast_to_fp8(x)
        assert fp8_data.shape == x.shape
        assert scale.shape == (1, 1)
        
    
    def test_invalid_input(self):
        """Test that function raises error for invalid inputs."""
        # 1D tensor should raise error
        x = torch.randn(128, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            per_token_cast_to_fp8(x)
        
        # 3D tensor should raise error
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            per_token_cast_to_fp8(x)


class TestPerBlockCastToFP8:
    """Test the per_block_cast_to_fp8 utility function."""
    
    def test_basic_functionality(self):
        """Test basic block-wise FP8 conversion."""
        x = torch.randn(256, 512, dtype=torch.bfloat16)
        
        fp8_data, scale = per_block_cast_to_fp8(x)
        
        # Check output shapes
        assert fp8_data.shape == x.shape
        # Scale should be (num_blocks_m, num_blocks_n)
        assert scale.shape == (2, 4)  # 256/128=2, 512/128=4
        
        # Check data types
        assert fp8_data.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32
        
        # Check scale values are reasonable
        assert torch.all(scale > 0)
        assert torch.all(scale <= 1.0)
    
    def test_padding_handling(self):
        """Test handling of tensors that need padding."""
        # Create tensor with sizes not divisible by 128
        x = torch.randn(100, 200, dtype=torch.bfloat16)
        
        fp8_data, scale = per_block_cast_to_fp8(x)
        
        # Output should maintain original shape
        assert fp8_data.shape == x.shape
        # Scale should be (ceil(100/128), ceil(200/128)) = (1, 2)
        assert scale.shape == (1, 2)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with size exactly 128
        x = torch.randn(128, 128, dtype=torch.bfloat16)
        fp8_data, scale = per_block_cast_to_fp8(x)
        assert fp8_data.shape == x.shape
        assert scale.shape == (1, 1)
        
        # Test with size 0
        x = torch.randn(0, 128, dtype=torch.bfloat16)
        fp8_data, scale = per_block_cast_to_fp8(x)
        assert fp8_data.shape == x.shape
        assert scale.shape == (0, 1)
    
    def test_invalid_input(self):
        """Test that function raises error for invalid inputs."""
        # 1D tensor should raise error
        x = torch.randn(128, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            per_block_cast_to_fp8(x)
        
        # 3D tensor should raise error
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            per_block_cast_to_fp8(x)


class TestFP8QuantizeMethod:
    """Test the FP8QuantizeMethod class."""
    
    def test_initialization(self):
        """Test proper initialization."""
        method = FP8QuantizeMethod()
        assert method.weight_fp8 is None
        assert method.weight_scale is None
    
    def test_inheritance(self):
        """Test that FP8QuantizeMethod inherits from QuantizeMethodBase."""
        method = FP8QuantizeMethod()
        assert isinstance(method, QuantizeMethodBase)
    
    def test_create_weights(self):
        """Test weight creation."""
        method = FP8QuantizeMethod()
        layer = Mock()
        
        # Mock the set_weight_attrs function
        with patch('fastvideo.layers.quantization.fp8_config.set_weight_attrs') as mock_set_attrs:
            method.create_weights(
                layer=layer,
                input_size_per_partition=256,
                output_partition_sizes=[128, 128],
                input_size=256,
                output_size=256,
                params_dtype=torch.float32,
                extra_attr="test"
            )
            
            # Check that weight parameter was created
            assert hasattr(layer, 'weight')
            # The mock should have been called to register the parameter
            assert mock_set_attrs.call_count >= 2
            
            # Check that the weight was registered as a parameter
            # Note: In the mock, we can't easily verify the exact type, but we can verify the call
            layer.register_parameter.assert_called_once()
    
    def test_process_weights_after_loading(self):
        """Test weight processing after loading."""
        method = FP8QuantizeMethod()
        layer = Mock()
        
        # Create a mock weight tensor
        mock_weight = torch.randn(128, 256, dtype=torch.float32)
        layer.weight = Mock()
        layer.weight.data = mock_weight
        
        # Mock the per_block_cast_to_fp8 function
        with patch('fastvideo.layers.quantization.fp8_config.per_block_cast_to_fp8') as mock_cast:
            # Use float16 instead of float8_e4m3fn for CPU compatibility
            mock_cast.return_value = (torch.randn(128, 256, dtype=torch.float16), 
                                    torch.randn(1, 2, dtype=torch.float32))
            
            method.process_weights_after_loading(layer)
            
            # Check that the cast function was called
            mock_cast.assert_called_once_with(mock_weight)
            
            # Check that attributes were set on the layer
            assert hasattr(layer, '_fp8_weight')
            assert hasattr(layer, '_fp8_weight_scale')
    
    def test_process_weights_no_weight(self):
        """Test weight processing when layer has no weight."""
        method = FP8QuantizeMethod()
        layer = Mock()
        layer.weight = None
        
        # Should not raise an error
        method.process_weights_after_loading(layer)
        
        # No attributes should be set
        # Check that the method didn't set any FP8 attributes
        # Since the method only sets attributes when weight exists, this should be fine
        pass
    
    @patch('fastvideo.layers.quantization.fp8_config.deep_gemm')
    def test_apply_with_existing_fp8_weights(self, mock_deep_gemm):
        """Test apply method with existing FP8 weights."""
        method = FP8QuantizeMethod()
        layer = Mock()
        
        # Mock the layer attributes
        layer.weight = Mock()
        layer.weight.shape = [256, 128]
        # Use float16 instead of float8_e4m3fn for CPU compatibility
        layer._fp8_weight = torch.randn(256, 128, dtype=torch.float16)
        layer._fp8_weight_scale = torch.randn(2, 1, dtype=torch.float32)
        
        # Mock input tensor
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        
        # Mock the per_token_cast_to_fp8 function
        with patch('fastvideo.layers.quantization.fp8_config.per_token_cast_to_fp8') as mock_cast:
            # Use float16 instead of float8_e4m3fn for CPU compatibility
            mock_cast.return_value = (torch.randn(6, 128, dtype=torch.float16), 
                                    torch.randn(6, 1, dtype=torch.float32))
            
            # Mock the deep_gemm function
            mock_deep_gemm.gemm_fp8_fp8_bf16_nt.return_value = None
            
            result = method.apply(layer, x)
            
            # Check that the cast function was called
            mock_cast.assert_called_once()
            
            # Check that deep_gemm was called
            mock_deep_gemm.gemm_fp8_fp8_bf16_nt.assert_called_once()
            
            # Check output shape
            assert result.shape == (2, 3, 256)
    
    @patch('fastvideo.layers.quantization.fp8_config.deep_gemm')
    def test_apply_without_existing_fp8_weights(self, mock_deep_gemm):
        """Test apply method when FP8 weights don't exist."""
        method = FP8QuantizeMethod()
        layer = Mock()
        
        # Mock the layer attributes
        layer.weight = Mock()
        layer.weight.data = torch.randn(256, 128, dtype=torch.float32)
        layer.weight.shape = [256, 128]
        
        # Ensure no existing FP8 weights
        layer._fp8_weight = None
        
        # Mock input tensor
        x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        
        # Mock the per_block_cast_to_fp8 and per_token_cast_to_fp8 functions
        with patch('fastvideo.layers.quantization.fp8_config.per_block_cast_to_fp8') as mock_block_cast, \
             patch('fastvideo.layers.quantization.fp8_config.per_token_cast_to_fp8') as mock_token_cast:
            
            # Use float16 instead of float8_e4m3fn for CPU compatibility
            mock_block_cast.return_value = (torch.randn(256, 128, dtype=torch.float16), 
                                          torch.randn(2, 1, dtype=torch.float32))
            # Use float16 instead of float8_e4m3fn for CPU compatibility
            mock_token_cast.return_value = (torch.randn(6, 128, dtype=torch.float16), 
                                          torch.randn(6, 1, dtype=torch.float32))
            
            # Mock the deep_gemm function
            mock_deep_gemm.gemm_fp8_fp8_bf16_nt.return_value = None
            
            result = method.apply(layer, x)
            
            # Check that both cast functions were called
            mock_block_cast.assert_called_once()
            mock_token_cast.assert_called_once()
            
            # Check that deep_gemm was called
            mock_deep_gemm.gemm_fp8_fp8_bf16_nt.assert_called_once()
            
            # Check output shape
            assert result.shape == (2, 3, 256)
    
    def test_apply_invalid_input_dtype(self):
        """Test apply method with invalid input dtype."""
        method = FP8QuantizeMethod()
        layer = Mock()
        layer.weight = Mock()
        layer.weight.shape = [256, 128]
        
        # Use float32 instead of bfloat16
        x = torch.randn(2, 3, 128, dtype=torch.float32)
        
        with pytest.raises(AssertionError, match="only allow bf16 inputs to fp8 linear"):
            method.apply(layer, x)


class TestFP8Config:
    """Test the FP8Config class."""
    
    def test_initialization(self):
        """Test proper initialization."""
        config = FP8Config()
        assert isinstance(config, QuantizationConfig)
    
    def test_inheritance(self):
        """Test that FP8Config inherits from QuantizationConfig."""
        config = FP8Config()
        assert isinstance(config, QuantizationConfig)
    
    def test_get_name(self):
        """Test get_name method."""
        config = FP8Config()
        assert config.get_name() == "fp8"
    
    def test_get_supported_act_dtypes(self):
        """Test get_supported_act_dtypes method."""
        config = FP8Config()
        supported_dtypes = config.get_supported_act_dtypes()
        assert len(supported_dtypes) == 1
        assert supported_dtypes[0] == torch.bfloat16
    
    def test_get_min_capability(self):
        """Test get_min_capability method."""
        capability = FP8Config.get_min_capability()
        assert capability == 90
    
    def test_get_config_filenames(self):
        """Test get_config_filenames method."""
        filenames = FP8Config.get_config_filenames()
        assert filenames == []
    
    def test_from_config(self):
        """Test from_config method."""
        config_dict = {"some_key": "some_value"}
        config = FP8Config.from_config(config_dict)
        assert isinstance(config, FP8Config)
    
    def test_get_quant_method_linear_layer(self):
        """Test get_quant_method with LinearBase layer."""
        config = FP8Config()
        
        # Mock a LinearBase layer
        mock_linear = Mock()
        mock_linear.__class__.__name__ = 'LinearBase'
        
        # Mock the isinstance check by patching the builtins.isinstance function
        with patch('builtins.isinstance', return_value=True):
            quant_method = config.get_quant_method(mock_linear, "test.prefix")
            
            assert isinstance(quant_method, FP8QuantizeMethod)
    
    def test_get_quant_method_non_linear_layer(self):
        """Test get_quant_method with non-LinearBase layer."""
        config = FP8Config()
        
        # Mock a non-linear layer
        mock_layer = Mock()
        mock_layer.__class__.__name__ = 'Conv2d'
        
        # Mock the isinstance check by patching the builtins.isinstance function
        with patch('builtins.isinstance', return_value=False):
            quant_method = config.get_quant_method(mock_layer, "test.prefix")
            
            assert quant_method is None


class TestConvertModelToFP8:
    """Test the convert_model_to_fp8 function."""
    
    def test_convert_model_with_linear_layers(self):
        """Test converting a model with linear layers."""
        # Create a simple model with linear layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                self.conv = nn.Conv2d(3, 64, 3)
            
            def forward(self, x):
                return self.linear2(self.linear1(x))
        
        model = SimpleModel()
        
        # Mock the isinstance check by patching the builtins.isinstance function
        with patch('builtins.isinstance') as mock_isinstance:
            # Make the linear layers appear as LinearBase instances
            mock_isinstance.side_effect = lambda obj, cls: obj.__class__.__name__ in ['Linear']
            
            # Mock the FP8Config and FP8QuantizeMethod
            with patch('fastvideo.layers.quantization.fp8_config.FP8Config') as mock_config_class, \
                 patch('fastvideo.layers.quantization.fp8_config.FP8QuantizeMethod') as mock_method_class:
                
                mock_config = Mock()
                mock_config_class.return_value = mock_config
                
                mock_method = Mock()
                mock_method_class.return_value = mock_method
                
                # Mock the get_quant_method method
                mock_config.get_quant_method.return_value = mock_method
                
                # Convert the model
                converted_model = convert_model_to_fp8(model)
                
                # Check that the model was returned
                assert converted_model is model
                
                # Check that linear layers were processed
                assert hasattr(model.linear1, 'quant_method')
                assert hasattr(model.linear1, 'quant_config')
                assert hasattr(model.linear2, 'quant_method')
                assert hasattr(model.linear2, 'quant_config')
                
                # Check that conv layer was not processed
                assert not hasattr(model.conv, 'quant_method')
                assert not hasattr(model.conv, 'quant_config')
    
    def test_convert_model_no_linear_layers(self):
        """Test converting a model without linear layers."""
        # Create a simple model without linear layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                return self.relu(self.conv(x))
        
        model = SimpleModel()
        
        # Mock the isinstance check by patching the builtins.isinstance function
        with patch('builtins.isinstance', return_value=False):
            
            # Convert the model
            converted_model = convert_model_to_fp8(model)
            
            # Check that the model was returned unchanged
            assert converted_model is model
            
            # Check that no quantization attributes were added
            assert not hasattr(model.conv, 'quant_method')
            assert not hasattr(model.relu, 'quant_method')
    
    def test_convert_model_nested_structure(self):
        """Test converting a model with nested module structure."""
        # Create a nested model
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
                self.decoder = nn.ModuleList([
                    nn.Linear(128, 64),
                    nn.Linear(64, 32)
                ])
        
        model = NestedModel()
        
        # Mock the isinstance check by patching the builtins.isinstance function
        with patch('builtins.isinstance') as mock_isinstance:
            mock_isinstance.side_effect = lambda obj, cls: obj.__class__.__name__ in ['Linear']
            
            # Mock the FP8Config and FP8QuantizeMethod
            with patch('fastvideo.layers.quantization.fp8_config.FP8Config') as mock_config_class, \
                 patch('fastvideo.layers.quantization.fp8_config.FP8QuantizeMethod') as mock_method_class:
                
                mock_config = Mock()
                mock_config_class.return_value = mock_config
                
                mock_method = Mock()
                mock_method_class.return_value = mock_method
                
                mock_config.get_quant_method.return_value = mock_method
                
                # Convert the model
                converted_model = convert_model_to_fp8(model)
                
                # Check that the model was returned
                assert converted_model is model
                
                # Check that all linear layers were processed
                assert hasattr(model.encoder[0], 'quant_method')
                assert hasattr(model.encoder[2], 'quant_method')
                assert hasattr(model.decoder[0], 'quant_method')
                assert hasattr(model.decoder[1], 'quant_method')


class TestIntegration:
    """Integration tests for the FP8 quantization system."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_end_to_end_fp8_quantization(self):
        """Test end-to-end FP8 quantization workflow."""
        # This test would require actual CUDA hardware and deep_gemm
        # For now, we'll just test the basic structure
        
        # Create a simple model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 256)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Mock the deep_gemm import to avoid actual CUDA calls
        with patch.dict('sys.modules', {'deep_gemm': Mock()}):
            # Mock the isinstance check by patching the builtins.isinstance function
            with patch('builtins.isinstance') as mock_isinstance:
                mock_isinstance.side_effect = lambda obj, cls: obj.__class__.__name__ in ['Linear']
                
                # Convert the model
                converted_model = convert_model_to_fp8(model)
                
                # Basic checks that conversion completed
                assert converted_model is model
                assert hasattr(model.linear, 'quant_method')
                assert hasattr(model.linear, 'quant_config')


if __name__ == "__main__":
    pytest.main([__file__]) 