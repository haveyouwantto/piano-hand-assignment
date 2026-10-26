"""Assign piano notes to the right or left hand.

    from piano_hand_assignment import assign_hands
    assign_hands("input.mid", "output.mid")
"""

from .predict import HandAssigner, assign_hands, get_assigner

__all__ = ["HandAssigner", "assign_hands", "get_assigner"]
__version__ = "0.1.0"
