"""
safety_bypass.py
================

The methodological core of the study: a dummy NSFW safety checker that bypasses
Stable Diffusion's pipeline-internal first-stage filter.

This single function is the technical implementation of the "forensic access"
discussed in §5.1 of the paper. By replacing the pipeline's `safety_checker`
with this dummy, images that would otherwise be removed before reaching the
user's visual field become observable.

    pipe.safety_checker = dummy_safety_checker

Read critically (§5.1), this single line retroactively proves the existence of
the first-stage erasure mechanism that is invisible to ordinary users.

Reference: Kubota (2026), §5.1 "The Bypass That Reveals Structure".
"""

from typing import List, Tuple, Any


def dummy_safety_checker(
    images: Any,
    clip_input: Any,
) -> Tuple[Any, List[bool]]:
    """Bypass Stable Diffusion's pipeline-internal NSFW safety checker.

    This function returns the input images unchanged with all-False NSFW flags,
    effectively disabling the pipeline-internal first-stage filter. Images
    classified as NSFW by the standalone (second-stage) checker can then be
    observed externally for analysis.

    The technical operation is trivially simple. The critical significance lies
    in what this single line reveals about the architecture: that the
    invisibility of erasure is itself implemented as technical structure.

    Parameters
    ----------
    images : array-like
        The batch of generated images, returned unchanged.
    clip_input : tensor
        CLIP input tensor (ignored by this dummy implementation).

    Returns
    -------
    images : array-like
        The input images, unmodified.
    has_nsfw_concepts : list of bool
        A list of False values, one per image, indicating no NSFW content
        was detected (because no detection is performed).

    Notes
    -----
    This function is intended for research purposes only. It is used to study
    how the NSFW filter mechanism operates by making its outputs externally
    observable. It is NOT a recommendation to disable safety mechanisms in
    production deployments of Stable Diffusion.

    See §3.3 (manual observation, ethical considerations) and §7 (discussion
    of methodological labor) for the ethical context of this intervention.

    Examples
    --------
    >>> from diffusers import StableDiffusionPipeline
    >>> from src.safety_bypass import dummy_safety_checker
    >>> pipe = StableDiffusionPipeline.from_pretrained(
    ...     "runwayml/stable-diffusion-v1-5"
    ... )
    >>> pipe.safety_checker = dummy_safety_checker
    """
    # Determine batch size from images (handles numpy arrays, tensors, lists)
    if hasattr(images, "shape"):
        batch_size = images.shape[0]
    else:
        batch_size = len(images)
    return images, [False] * batch_size
