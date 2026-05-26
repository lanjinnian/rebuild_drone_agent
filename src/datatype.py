from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


GPSLocation: TypeAlias = tuple[float, float, float]


@dataclass
class BaseFrame:
    id: int
    image: NDArray[np.uint8]
    gps_location: GPSLocation | None
    mask: NDArray[np.uint8] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.mask is None:
            self.mask = np.zeros_like(self.image)
            return

        if self.mask.shape != self.image.shape:
            raise ValueError(
                f"frame mask shape {self.mask.shape} does not match image shape "
                f"{self.image.shape}"
            )
        if self.mask.dtype != self.image.dtype:
            raise ValueError(
                f"frame mask dtype {self.mask.dtype} does not match image dtype "
                f"{self.image.dtype}"
            )


@dataclass
class FrameInChunk(BaseFrame):
    chunk_id: int
    world_points: NDArray[np.floating]
    world_points_conf: NDArray[np.floating]


@dataclass
class ChunkToProcess:
    chunk_id: int
    frames: list[FrameInChunk] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frame_ids:
            self.frame_ids = [frame.id for frame in self.frames]

    def add_frame(self, frame: FrameInChunk) -> None:
        self.frames.append(frame)
        self.frame_ids.append(frame.id)


@dataclass
class OriginalFrames:
    frames: list[BaseFrame] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frame_ids:
            self.frame_ids = [frame.id for frame in self.frames]

    def add_frame(self, frame: BaseFrame) -> None:
        self.frames.append(frame)
        self.frame_ids.append(frame.id)

    def delete_frame(self, frame_id: int) -> None:
        index = self.frame_ids.index(frame_id)
        del self.frames[index]
        del self.frame_ids[index]


@dataclass
class Chunk:
    id: int
    frames: list[BaseFrame] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.frame_ids:
            self.frame_ids = [frame.id for frame in self.frames]
