"""
encoder.py  –  SegMAN backbone for CISRMamba
===========================================
Wraps SegMANEncoder (segman_encode.py) as the hierarchical backbone.

  dims   = [64, 144, 288, 512]   (SegMAN-s variant)
  depths = [ 2,   2,  10,   4]
  stem stride = 4  (two stride-2 convs)

Returns 4 channel-first feature maps at strides 4 / 8 / 16 / 32.

Source: https://github.com/yunxiangfu2001/SegMAN
"""

from backbone import MobileMambaEncoder as VSSEncoder  # public interface for CISRMamba
