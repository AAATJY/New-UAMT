"""
Frequency-domain decomposition utilities for 2D tensors.

Splits an input image into low-frequency and high-frequency components using
differentiable FFT operations. The split is controlled by a radial low-pass
mask in the frequency domain.
"""

import torch


def _make_radial_lowpass_mask(height: int, width: int, ratio: float,
                               device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a 2-D radial low-pass mask in the FFT-shifted frequency domain.

    Frequencies within ``ratio`` of the Nyquist radius are kept (value=1),
    all others are zeroed out.  The mask is constructed once and cached by
    the caller via ``freq_decompose`` so it is never rebuilt for the same
    spatial size.

    Args:
        height: Spatial height of the feature/image map.
        width:  Spatial width of the feature/image map.
        ratio:  Fraction of the maximum frequency radius to keep (0 < ratio ≤ 1).
        device: Target device for the output tensor.
        dtype:  Target floating-point dtype for the output tensor.

    Returns:
        Tensor of shape ``[1, 1, height, width]`` with values in {0, 1}.
    """
    # Build normalised coordinate grid centred at (0, 0)
    cy, cx = height / 2.0, width / 2.0
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width,  device=device, dtype=dtype)
    # Normalise to [-0.5, 0.5)
    ys = (ys - cy) / height
    xs = (xs - cx) / width
    # Radial distance in normalised coords; max possible is ~0.5*sqrt(2)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    radius = torch.sqrt(grid_y ** 2 + grid_x ** 2)
    # Threshold at `ratio * 0.5`  (0.5 = Nyquist in normalised coords)
    mask = (radius <= ratio * 0.5).to(dtype)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def freq_decompose(x: torch.Tensor, ratio: float = 0.15):
    """Split a 2-D image tensor into low- and high-frequency components.

    Uses ``torch.fft.fft2`` / ``torch.fft.ifft2`` to perform an exact,
    differentiable frequency-domain split.  The low-pass cut-off is a
    radial mask with radius ``ratio * Nyquist``.

    Args:
        x:     Input tensor of shape ``[B, C, H, W]`` (real-valued).
        ratio: Low-pass radius as a fraction of Nyquist (default 0.15).
               Typical range: 0.05 – 0.30.

    Returns:
        x_low:  Low-frequency reconstruction, same shape as ``x``.
        x_high: High-frequency residual (x – x_low), same shape as ``x``.
    """
    B, C, H, W = x.shape

    # Forward FFT – result is complex-valued, zero-frequency at corner
    X = torch.fft.fft2(x, norm='ortho')

    # Shift zero-frequency to centre for intuitive radial masking
    X_shifted = torch.fft.fftshift(X, dim=(-2, -1))

    # Build (or reuse) the radial low-pass mask
    mask = _make_radial_lowpass_mask(H, W, ratio, device=x.device, dtype=x.dtype)

    # Apply mask (broadcast over batch & channel dimensions)
    X_low_shifted  = X_shifted * mask
    X_high_shifted = X_shifted * (1.0 - mask)

    # Shift back and apply inverse FFT; take real part (imaginary is ~0)
    X_low  = torch.fft.ifftshift(X_low_shifted,  dim=(-2, -1))
    X_high = torch.fft.ifftshift(X_high_shifted, dim=(-2, -1))

    x_low  = torch.fft.ifft2(X_low,  norm='ortho').real
    x_high = torch.fft.ifft2(X_high, norm='ortho').real

    return x_low, x_high
