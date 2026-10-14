# MIMO-UNet-Enhanced

Experimental extensions of [MIMO-UNet](https://github.com/chosj95/MIMO-UNet) (attention, frequency branch,
refinement) for single-image deblurring.

## Important: two silent bugs before commit `0150a37` (fixed)

Results produced with this repository **before the fix commit** are not valid frequency-branch ablations:

1. **The frequency branch never executed.** `FrequencyBranch` combined the processed magnitude
   (`out_channels` channels) with the 3-channel imaginary part in `torch.complex`, which always raises a shape
   error (`size of tensor a (32) must match the size of tensor b (3)`). A bare `except:` silently replaced it with
   a plain spatial convolution, so every "with / without frequency" comparison compared two convolution stacks.
2. **The FFT loss was ~128x too weak.** The PyTorch-API migration replaced
   `torch.rfft(normalized=False, onesided=False)` with `torch.fft.rfft2(norm='ortho')`, which divides by
   `sqrt(H*W)`, and summed two L1 terms while keeping the 0.1 weight. On 256x256 crops the frequency term was
   128x smaller than in the official MIMO-UNet (exactly `sqrt(HW)/2`). Bare `except:` blocks could also set it to 0.

The fix commit makes the frequency branch refine the magnitude spectrum and keep the phase, restores the
unnormalised full-FFT loss (`fft_l1_loss`), and removes the silent fallbacks so errors surface.
Checkpoints trained before the fix (e.g. `Best.pkl`, `Final.pkl`) have frequency-branch weights that were trained
as a spatial convolution; they do not load into the fixed branch (`strict=False` re-initialises it).

A controlled study of what does and does not improve the original MIMO-UNet, including minimal reproductions of
both bugs, is at https://github.com/ManasShouche/mimo-unet-zero-cost-study.
