"""Clin-JEPA: a latent world model for intensive-care patient trajectories.

An LLM encoder reads each hour of the record as text; an action-conditioned Transformer predictor, trained together
with it under a multi-phase curriculum, rolls the patient state forward in the encoder's latent space.
"""

__version__ = "0.1.0"
