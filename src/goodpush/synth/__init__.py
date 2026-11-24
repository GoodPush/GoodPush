"""Synthetic riders: a made-up rider with a known physiology, their whole history generated as goodpush data
(rides, nights, tests), and how well goodpush's analysis finds that physiology again (SynthCheck)."""
from .profile import RiderProfile
from .rider import SyntheticRider

__all__ = ["RiderProfile", "SyntheticRider"]
