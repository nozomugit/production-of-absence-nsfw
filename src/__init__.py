"""
production-of-absence-nsfw
==========================

Source code accompanying:

    Kubota, Nozomu. 2026. "Production of Absence: Reading Stable Diffusion's
    NSFW Filter through Code and Typology." Expanded 2026: Conference on
    Animation and Interactive Art.

Modules:

    safety_bypass : The dummy_safety_checker function (paper §5.1)
    generate      : Image generation pipeline (paper §3.1, §3.2)
    extract_clip  : Auxiliary CLIP embedding extraction (paper §4.2--4.3)
    compose_grid  : Typological grid composition (paper §6.1, Figure 8)
    figures       : Generate descriptive analysis figures from available data

Author: Nozomu Kubota <nozomu.k@cnxt.jp>
License: MIT (code), CC-BY 4.0 (annotations)
"""

__version__ = "1.1.0"
__author__ = "Nozomu Kubota"
__email__ = "nozomu.k@cnxt.jp"
