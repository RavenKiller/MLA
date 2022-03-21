from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

# import numpy as np
import torch
from gym import spaces

# from habitat.core.utils import try_cv2_import
from habitat.utils import profiling_wrapper

# from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import DictTree, TensorDict

# from gym.spaces import Box
# from habitat import logger
# from habitat.core.dataset import Episode


# from habitat_baselines.common.tensorboard_utils import TensorboardWriter
# from PIL import Image
# from torch import Size, Tensor
# from torch import nn as nn

PAD_LEN = 25
PAD_SEQ = [0] * PAD_LEN
MAX_SUB = 10


def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    sub_tokens: bool = True,
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    sub_instruction_sensor_uuid = "sub_" + instruction_sensor_uuid
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][
                instruction_sensor_uuid
            ]["tokens"]
            observations[i][sub_instruction_sensor_uuid] = observations[i][
                sub_instruction_sensor_uuid
            ]["tokens"]
        # else:
        #     break
    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    """removes the frame stack dimension of a Box space shape if it exists."""
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )


# My cumstomized batch_obs() function
@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))
