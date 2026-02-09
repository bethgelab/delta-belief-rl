import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

""""
From Search-R1 codebase
Credits to authors:
"""


@dataclass
class TensorConfig:
    pad_token_id: int
    end_token_id: int
    start_token_id: int
    assistant_token_id: int
    new_line_token_id: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(
        self,
        tensor_dict: Dict[str, torch.Tensor],
        keys: List[str],
        cut_left: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Cut tensors to their effective length based on attention mask."""
        effective_len = tensor_dict["attention_mask"].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(
        self, tensor: torch.Tensor, pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert padding structure and return sorted tensor with indices."""
        mask = (
            tensor != self.config.pad_token_id
            if pad_to_left
            else tensor == self.config.pad_token_id
        )
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input ids."""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position ids from attention mask."""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor], pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding."""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(
        self, responses: torch.Tensor, active_mask: torch.Tensor, padding: int = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Pad responses for non-active examples with pad tokens.
        """
        assert active_mask.sum() == responses.shape[0]
        # Create masked responses tensor
        if padding is None:
            padding = self.config.pad_token_id

        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len),
            padding,  # self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        return padded_responses

    def _pad_active(
        self, current_states: List[int], active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            current_states: List[int] - list of current states, where 1 is finished and 0 is active
            active_mask: torch.Tensor - boolean tensor of shape (bsz_all,)

        Returns:
            torch.Tensor - boolean tensor of shape (bsz_all,)
        """
        assert active_mask.sum() == len(current_states)
        bsz = active_mask.shape[0]
        all_states = torch.zeros(
            (bsz,), dtype=torch.int
        )  # set all trajectories to finished

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                # if current_state is 1, set to 0 (finished), otherwise keep 1 (active)
                all_states[i] = 1 if not current_states[s] else 0
                s += 1

        return all_states.bool()

    def _pad_action(
        self, current_actions: List[int], active_mask: torch.Tensor
    ) -> torch.Tensor:
        assert active_mask.sum() == len(current_actions)
        bsz = active_mask.shape[0]
        all_actions = torch.ones((bsz,), dtype=torch.int)  # set all actions to valid

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                # set to 0 if action was not valid
                all_actions[i] = 0 if not current_actions[s] else 1
                s += 1

        return all_actions

    def _remove_end_token(self, responses_id: torch.Tensor) -> torch.Tensor:
        end_id = self.config.end_token_id  # an int
        pad_id = self.config.pad_token_id  # an int

        # mask is True everywhere that it's _not_ the end token
        mask = responses_id != end_id

        # replace any end-token slot with pad_id, leave everything else untouched
        return torch.where(mask, responses_id, pad_id)

    def _remove_assistant_token(self, response_ids: torch.Tensor) -> torch.Tensor:
        special = torch.tensor(
            [
                self.config.start_token_id,
                self.config.assistant_token_id,
                self.config.end_token_id,
                self.config.new_line_token_id,
            ],
            device=response_ids.device,
        )
        # mask == True for tokens *not* in `special`
        mask = ~torch.isin(response_ids, special)
        out = torch.where(mask, response_ids, self.config.pad_token_id)
        out, _ = self.convert_pad_structure(out, pad_to_left=True)
        effective_length = self.create_attention_mask(out).sum(dim=1).max()
        return out[:, -effective_length:]

    def _contiguous_spans(self, idx_1d: torch.Tensor) -> List[torch.Tensor]:
        """Split a sorted 1D index tensor into contiguous runs."""
        if idx_1d.numel() == 0:
            return []
        breaks = torch.where(idx_1d[1:] != idx_1d[:-1] + 1)[0] + 1
        return torch.tensor_split(idx_1d, breaks.tolist())

