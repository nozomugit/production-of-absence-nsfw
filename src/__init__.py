"""
production-of-absence-nsfw
==========================

Source code accompanying:

    Kubota, Nozomu. 2026. "Production of Absence: Reading Stable Diffusion's
    NSFW Filter through Code and Typology." Expanded Conference 2026.

Modules:

    safety_bypass : The dummy_safety_checker function (paper §5.1)
    generate      : Image generation pipeline (paper §3.1, §3.2)
    extract_clip  : CLIP embedding extraction (paper §5.3)
    compose_grid  : Typological grid composition (paper §6.1, Figure 8)
    figures       : Regenerate paper figures (Figures 1, 5, 6, 7)

Author: Nozomu Kubota <nozomu.k@cnxt.jp>
License: MIT (code), CC-BY 4.0 (annotations)
"""

__version__ = "1.0.0"
__author__ = "Nozomu Kubota"
__email__ = "nozomu.k@cnxt.jp"
