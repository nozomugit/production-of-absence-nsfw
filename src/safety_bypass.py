"""
safety_bypass.py
================

Audit utility that replaces Stable Diffusion's configured safety-checker call.

In an ordinary pipeline, the configured safety checker is called once after
generation. Replacing it with this pass-through function preserves the generated
image for inspection. The audit workflow can then apply the original checker
once outside the pipeline and record its result. That bypass-plus-reapplication
is an audit observation procedure, not evidence that the ordinary pipeline has
two checker stages.

    pipe.safety_checker = dummy_safety_checker

The returned all-False values mean that this dummy performs no detection. They
must not be interpreted as content labels or safety certifications. Likewise,
a later checker flag is a model-specific threshold outcome rather than verified
ground truth about the image.

Reference: Kubota (2026), §5.1 "The Bypass That Reveals Structure".
"""

from typing import Any, List, Tuple


def dummy_safety_checker(
    images: Any,
    clip_input: Any,
) -> Tuple[Any, List[bool]]:
    """Preserve images by replacing the pipeline's checker invocation.

    This function returns the input images unchanged and supplies all-False
    placeholder flags. It does not inspect or classify image content. In the
    audit workflow, the configured checker is separately re-applied once to the
    preserved image so that its Boolean threshold outcome can be recorded.

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
        A list of False placeholders, one per image. No detection is performed,
        so these values do not indicate that the images are safe or unflagged
        by an actual checker.

    Notes
    -----
    This function is intended for research purposes only. It is used to study
    a configured checker's behavior by making generated images available for a
    controlled re-application. It is NOT a recommendation to disable safety
    mechanisms in production deployments of Stable Diffusion.

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
