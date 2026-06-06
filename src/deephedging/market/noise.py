"""Addressable noise streams.

A stateful ``torch.Generator`` cannot be rewound to an arbitrary draw, so
a fused CUDA backward that regenerates path noise instead of storing it
has no way to name the noise it needs. A counter-based pair fixes this.
``(seed, stream)`` fully determines every draw. The eager simulators
derive a freshly seeded generator from the pair; the Philox kernel will
consume the same pair as its ``(seed, subsequence)`` directly. Replay is
therefore exact per backend, while cross-backend bit equality is not
promised because the underlying RNG algorithms differ. Distributed
training assigns disjoint streams per rank so ranks draw independent
paths instead of silently identical ones.
"""

from dataclasses import dataclass

import torch

_GOLDEN = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return (value ^ (value >> 31)) & _MASK


@dataclass(frozen=True)
class NoiseSpec:
    """Names one independent noise stream.

    Attributes:
        seed: Experiment-level seed.
        stream: Stream index within the experiment, for example the
            training iteration or the distributed rank.
    """

    seed: int
    stream: int = 0

    def child(self, index: int) -> "NoiseSpec":
        """Derives the stream for one unit of work.

        Args:
            index: Iteration or rank index.

        Returns:
            A spec naming an independent stream.
        """
        return NoiseSpec(seed=self.seed, stream=self.stream + index)

    def torch_generator(self, device: torch.device | str = "cpu") -> torch.Generator:
        """Materialises a generator seeded deterministically from the pair.

        Streams are decorrelated through a splitmix64 mix of seed and
        stream, so adjacent stream indices produce statistically
        independent draws.

        Args:
            device: Device for the generator.

        Returns:
            A freshly seeded generator.
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(_splitmix64((self.seed ^ (self.stream * _GOLDEN)) & _MASK))
        return generator
