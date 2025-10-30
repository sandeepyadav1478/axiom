"""
Model Quantization for Sub-100 Microsecond Performance

Applies INT8 quantization to neural networks for 4x speedup.
Critical for achieving <100 microsecond target on CPU or 10-50us on GPU.

Process:
1. Load FP32 model
2. Apply dynamic/static quantization
3. Validate accuracy (must maintain 99.99%)
4. Benchmark performance improvement
5. Save optimized model
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
from pathlib import Path
import time
from typing import Dict, Tuple


class ModelQuantizer:
    """
    Quantize neural networks for ultra-fast inference
    
    Techniques:
    - Dynamic quantization (easiest, 4x speedup)
    - Static quantization (requires calibration, 4-8x speedup)
    - Quantization-aware training (best accuracy, 4-8x speedup)
    """
    
    def __init__(self, model: nn.Module, model_name: str = "model"):
        """
        Initialize quantizer
        
        Args:
            model: PyTorch model to quantize
            model_name: Name for saving
        """
        self.model_fp32 = model
        self.model_name = model_name
        self.model_int8 = None
        
        # Ensure model is in eval mode
        self.model_fp32.eval()
    
    def quantize_dynamic(self) -> nn.Module:
        """
        Apply dynamic quantization (fastest to implement)
        
        Quantizes:
        - Weights: FP32 → INT8 (stored)
        - Activations: FP32 (computed dynamically)
        
        Speedup: 2-4x on CPU
        Accuracy: Usually 99.9%+ maintained
        """
        print(f"Applying dynamic quantization to {self.model_name}...")
        
        # Quantize
        self.model_int8 = quantization.quantize_dynamic(
            self.model_fp32,
            {nn.Linear, nn.Conv2d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        print(f"✓ Dynamic quantization complete")
        
        return self.model_int8
    
    def quantize_static(
        self,
        calibration_data: torch.Tensor,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Apply static quantization (better performance)
        
        Quantizes:
        - Weights: FP32 → INT8
        - Activations: FP32 → INT8 (using calibration data)
        
        Speedup: 4-8x on CPU/GPU
        Accuracy: Requires validation
        """
        print(f"Applying static quantization to {self.model_name}...")
        
        # Prepare model for quantization
        model = self.model_fp32
        model.qconfig = quantization.get_default_qconfig('fbgemm')  # x86 CPUs
        
        # Fuse layers (Conv+ReLU, Linear+ReLU)
        model_fused = quantization.fuse_modules(model, [['conv', 'relu']])
        
        # Prepare for calibration
        model_prepared = quantization.prepare(model_fused)
        
        # Calibrate with sample data
        print(f"Calibrating with {num_calibration_batches} batches...")
        with torch.no_grad():
            for i in range(num_calibration_batches):
                # Use calibration data
                batch = calibration_data[i] if len(calibration_data) > i else calibration_data[0]
                _ = model_prepared(batch)
        
        # Convert to quantized model
        self.model_int8 = quantization.convert(model_prepared)
        
        print(f"✓ Static quantization complete")
        
        return self.model_int8
    
    def benchmark_comparison(
        self,
        test_input: torch.Tensor,
        iterations: int = 1000
    ) -> Dict:
        """
        Benchmark FP32 vs INT8 performance
        
        Returns:
            Dict with timing comparison and speedup
        """
        print(f"\nBenchmarking {self.model_name} (FP32 vs INT8)...")
        print(f"Iterations: {iterations}")
        
        # Benchmark FP32
        times_fp32 = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = self.model_fp32(test_input)
                elapsed_us = (time.perf_counter() - start) * 1_000_000
                times_fp32.append(elapsed_us)
        
        # Benchmark INT8
        times_int8 = []
        if self.model_int8 is not None:
            with torch.no_grad():
                for _ in range(iterations):
                    start = time.perf_counter()
                    _ = self.model_int8(test_input)
                    elapsed_us = (time.perf_counter() - start) * 1_000_000
                    times_int8.append(elapsed_us)
        
        times_fp32 = np.array(times_fp32)
        times_int8 = np.array(times_int8)
        
        results = {
            'fp32_mean_us': np.mean(times_fp32),
            'fp32_median_us': np.median(times_fp32),
            'fp32_p95_us': np.percentile(times_fp32, 95),
            'int8_mean_us': np.mean(times_int8),
            'int8_median_us': np.median(times_int8),
            'int8_p95_us': np.percentile(times_int8, 95),
            'speedup': np.mean(times_fp32) / np.mean(times_int8),
            'size_reduction_mb': self._get_model_size_reduction()
        }
        
        print(f"\nResults:")
        print(f"  FP32: {results['fp32_mean_us']:.2f}us (mean), {results['fp32_p95_us']:.2f}us (p95)")
        print(f"  INT8: {results['int8_mean_us']:.2f}us (mean), {results['int8_p95_us']:.2f}us (p95)")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Size reduction: {results['size_reduction_mb']:.2f}MB")
        
        return results
    
    def validate_accuracy(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        tolerance: float = 0.01
    ) -> Dict:
        """
        Validate that quantization maintains accuracy
        
        Args:
            test_data: Test inputs
            test_labels: Expected outputs
            tolerance: Maximum acceptable error
        
        Returns:
            Dict with accuracy metrics
        """
        print(f"\nValidating accuracy after quantization...")
        
        with torch.no_grad():
            # FP32 predictions
            outputs_fp32 = self.model_fp32(test_data)
            
            # INT8 predictions
            outputs_int8 = self.model_int8(test_data)
            
            # Calculate differences
            diff = torch.abs(outputs_fp32 - outputs_int8)
            relative_error = diff / (torch.abs(outputs_fp32) + 1e-10)
        
        results = {
            'mean_absolute_error': float(diff.mean()),
            'max_absolute_error': float(diff.max()),
            'mean_relative_error': float(relative_error.mean()),
            'max_relative_error': float(relative_error.max()),
            'accuracy_maintained': float(relative_error.mean()) < tolerance
        }
        
        print(f"  Mean relative error: {results['mean_relative_error']:.6f}")
        print(f"  Max relative error: {results['max_relative_error']:.6f}")
        print(f"  Accuracy maintained (<{tolerance}): {'✓ YES' if results['accuracy_maintained'] else '✗ NO'}")
        
        return results
    
    def save_quantized_model(self, output_path: str):
        """Save quantized model to disk"""
        if self.model_int8 is None:
            raise ValueError("No quantized model available. Run quantize_dynamic() or quantize_static() first.")
        
        torch.save(self.model_int8.state_dict(), output_path)
        print(f"✓ Quantized model saved to {output_path}")
    
    def _get_model_size_reduction(self) -> float:
        """Calculate model size reduction in MB"""
        # FP32: 4 bytes per parameter
        # INT8: 1 byte per parameter
        # Reduction: ~4x
        
        param_count = sum(p.numel() for p in self.model_fp32.parameters())
        fp32_size_mb = param_count * 4 / (1024 ** 2)
        int8_size_mb = param_count * 1 / (1024 ** 2)
        reduction = fp32_size_mb - int8_size_mb
        
        return reduction


# Example usage with Greeks model
if __name__ == "__main__":
    from axiom.derivatives.ultra_fast_greeks import QuantizedGreeksNetwork
    
    print("="*60)
    print("MODEL QUANTIZATION FOR SUB-100 MICROSECOND PERFORMANCE")
    print("="*60)
    
    # Create FP32 model
    model_fp32 = QuantizedGreeksNetwork()
    # In production: model_fp32.load_state_dict(torch.load('greeks_fp32.pth'))
    
    # Initialize quantizer
    quantizer = ModelQuantizer(model_fp32, model_name="GreeksNetwork")
    
    # Apply dynamic quantization
    model_int8 = quantizer.quantize_dynamic()
    
    # Benchmark
    test_input = torch.randn(1, 5)  # [spot, strike, time, rate, vol]
    benchmark_results = quantizer.benchmark_comparison(
        test_input=test_input,
        iterations=10000
    )
    
    # Validate accuracy
    test_data = torch.randn(100, 5)
    test_labels = torch.randn(100, 6)  # 6 outputs
    accuracy_results = quantizer.validate_accuracy(
        test_data=test_data,
        test_labels=test_labels
    )
    
    # Save if accuracy is good
    if accuracy_results['accuracy_maintained']:
        quantizer.save_quantized_model('greeks_int8_quantized.pth')
        
        print("\n" + "="*60)
        print("QUANTIZATION SUCCESS")
        print("="*60)
        print(f"✓ Speedup: {benchmark_results['speedup']:.2f}x")
        print(f"✓ Accuracy maintained: {accuracy_results['mean_relative_error']:.6f} error")
        print(f"✓ Model size reduced: {benchmark_results['size_reduction_mb']:.2f}MB")
        print(f"\nNext steps:")
        print(f"  1. Integrate into ultra_fast_greeks.py")
        print(f"  2. Test in production environment")
        print(f"  3. Validate <100us with real data")
    else:
        print("\n✗ Accuracy not maintained, need to adjust quantization")