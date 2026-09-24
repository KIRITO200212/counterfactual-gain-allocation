"""Qwen-independent tensor counterfactual operators.

Every operator accepts embeddings ``[B,N,D]``, position ids ``[B,N,3]``, and
boolean token masks ``[B,N]``.  For full multimodal sequences, pass
``active_visual_token_mask`` so touched-token ratios use visual tokens only;
visual-only tensors may omit it for backward compatibility.  See
:class:`OperatorResult` for the shared rejection and audit contract.
"""

from .content_slot_swap import content_slot_swap, swap_content_slots
from .context_transport import context_transport, context_transport_null
from .geometry import OperatorResult, RejectReason, relation_sign
from .irrelevant_pair import irrelevant_pair_control, irrelevant_pair_edit
from .position_slot_swap import position_slot_swap, swap_position_slots
from .resampling_null import resample_in_place_null, resampling_null

__all__ = [
    "OperatorResult",
    "RejectReason",
    "content_slot_swap",
    "context_transport",
    "context_transport_null",
    "irrelevant_pair_control",
    "irrelevant_pair_edit",
    "position_slot_swap",
    "relation_sign",
    "resample_in_place_null",
    "resampling_null",
    "swap_content_slots",
    "swap_position_slots",
]
