from collections.abc import Iterator, Mapping, Sequence
import json
import logging
import math
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

DOMAIN_KEY = "domain"


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _frame_indices_for_episodes(episode_indices: Sequence[int], episodes: Sequence[int]) -> np.ndarray:
    """Return full-dataset frame indices belonging to the selected episodes."""
    selected = {int(episode) for episode in episodes}
    if not selected:
        raise ValueError("Episode selection is empty.")

    frame_episodes = np.asarray(episode_indices, dtype=np.int64)
    if frame_episodes.ndim != 1:
        raise ValueError("The episode_index column must be one-dimensional.")
    available = {int(episode) for episode in np.unique(frame_episodes)}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"Unknown episode(s) requested: {unknown}.")

    frame_indices = np.flatnonzero(np.isin(frame_episodes, tuple(selected))).astype(np.int64, copy=False)
    if frame_indices.size == 0:
        raise ValueError(f"No frames found for selected episode(s): {sorted(selected)}.")
    return frame_indices


class FrameIndexDataset(Dataset[T_co]):
    """Expose selected frames while keeping reads on the full source dataset."""

    def __init__(self, dataset: Dataset[T_co], frame_indices: Sequence[int]):
        self._dataset = dataset
        self._frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if self._frame_indices.ndim != 1 or self._frame_indices.size == 0:
            raise ValueError("Frame-index selection must contain at least one frame.")

    def __getitem__(self, index: SupportsIndex) -> T_co:
        index = index.__index__()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self._dataset[int(self._frame_indices[index])]

    def __len__(self) -> int:
        return int(self._frame_indices.size)


class MixtureDataset(Dataset[dict]):
    """Concatenate real and simulation datasets and retain their domain."""

    def __init__(self, real: Dataset, sim: Dataset):
        self.real = real
        self.sim = sim
        self._real_len = len(real)
        self._sim_len = len(sim)

    def __len__(self) -> int:
        return self._real_len + self._sim_len

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self._real_len:
            sample = self.real[index]
            domain = 0
        else:
            sample = self.sim[index - self._real_len]
            domain = 1
        if not isinstance(sample, dict):
            raise TypeError("MixtureDataset expects each dataset sample to be a dictionary.")
        # Keep this alongside the model payload. Observation.from_dict ignores
        # unknown keys, while train.py consumes it for domain-specific metrics.
        return {**sample, DOMAIN_KEY: np.int32(domain)}


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Seeded, resumable batches with an exact simulation fraction.

    Samples are drawn with replacement so differently sized datasets can be
    mixed indefinitely. ``batches_consumed`` is advanced before each yielded
    batch, making ``state_dict`` safe to capture at a checkpoint boundary.
    """

    def __init__(
        self,
        real_size: int,
        sim_size: int,
        batch_size: int,
        sim_weight: float,
        *,
        seed: int = 0,
        num_batches: int | None = None,
    ):
        if real_size < 0 or sim_size < 0:
            raise ValueError("Dataset sizes must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        n_sim_float = float(sim_weight) * batch_size
        n_sim = round(n_sim_float)
        if not math.isclose(n_sim_float, n_sim, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"sim_weight * batch_size must be an integer for stratified batches; "
                f"got {sim_weight} * {batch_size} = {n_sim_float}."
            )
        if n_sim < 0 or n_sim > batch_size:
            raise ValueError(f"sim_weight must be in [0, 1], got {sim_weight}.")
        if n_sim and sim_size == 0:
            raise ValueError("Simulation samples are required when sim_weight is non-zero.")
        if batch_size - n_sim and real_size == 0:
            raise ValueError("Real samples are required when sim_weight is less than one.")

        self.real_size = real_size
        self.sim_size = sim_size
        self.batch_size = batch_size
        self.sim_weight = float(sim_weight)
        self.n_sim = n_sim
        self.n_real = batch_size - n_sim
        self.seed = int(seed)
        self.epoch = 0
        self.batches_consumed = 0
        self._num_batches = num_batches

    def __len__(self) -> int:
        if self._num_batches is None:
            raise TypeError("The infinite stratified sampler has no finite length.")
        return self._num_batches

    def _batch(self, batch_number: int) -> list[int]:
        # A per-batch seed avoids dependence on Python or NumPy generator state
        # and therefore makes a resumed sampler bit-for-bit deterministic.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, batch_number]))
        real = (
            rng.integers(0, self.real_size, size=self.n_real, dtype=np.int64)
            if self.n_real
            else np.empty(0, dtype=np.int64)
        )
        sim = (
            rng.integers(0, self.sim_size, size=self.n_sim, dtype=np.int64) + self.real_size
            if self.n_sim
            else np.empty(0, dtype=np.int64)
        )
        indices = np.concatenate((real, sim))
        rng.shuffle(indices)
        return indices.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        if self._num_batches is not None and self.batches_consumed >= self._num_batches:
            # TorchDataLoader starts another epoch when its finite inner loader
            # ends. Explicit RLinf set_epoch calls reset this before iteration.
            self.epoch += 1
            self.batches_consumed = 0
        batch_number = self.batches_consumed
        while self._num_batches is None or batch_number < self._num_batches:
            batch = self._batch(batch_number)
            self.batches_consumed = batch_number + 1
            batch_number += 1
            yield batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.batches_consumed = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.seed = int(state["seed"])
        self.epoch = int(state.get("epoch", 0))
        self.batches_consumed = int(state.get("batches_consumed", 0))


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _resolve_lerobot_root(repo_id: str, metadata: object) -> pathlib.Path | None:
    root = getattr(metadata, "root", None)
    if root is not None:
        root = pathlib.Path(root).expanduser()
        if (root / "meta" / "info.json").is_file() or (root / "meta" / "episode_sources.jsonl").is_file():
            return root
    local = pathlib.Path(repo_id).expanduser()
    if (local / "meta" / "info.json").is_file() or (local / "meta" / "episode_sources.jsonl").is_file():
        return local.resolve()
    hf_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    cached = hf_home / repo_id
    if (cached / "meta" / "info.json").is_file() or (cached / "meta" / "episode_sources.jsonl").is_file():
        return cached.resolve()
    return None


def _episodes_for_source(repo_id: str, metadata: object, source: str) -> tuple[int, ...]:
    root = _resolve_lerobot_root(repo_id, metadata)
    if root is None:
        raise FileNotFoundError(
            f"Cannot resolve the local LeRobot root for {repo_id!r}; "
            "episode_source selection requires meta/episode_sources.jsonl."
        )
    source_path = root / "meta" / "episode_sources.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Episode source metadata not found: {source_path}")
    episodes: list[int] = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != source:
                continue
            episode = row.get("episode_index", row.get("episode"))
            if episode is None:
                raise ValueError(f"Missing episode_index in {source_path}:{line_number}.")
            episodes.append(int(episode))
    if not episodes:
        raise ValueError(f"No episodes with source={source!r} found in {source_path}.")
    return tuple(episodes)


def _feature_shape(feature: object) -> tuple[int, ...] | None:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    elif isinstance(feature, (tuple, list)):
        shape = feature
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _metadata_features(source: object) -> Mapping | None:
    metadata = source.get("features") if isinstance(source, Mapping) else getattr(source, "features", None)
    if metadata is None:
        nested = source.get("metadata") if isinstance(source, Mapping) else getattr(source, "metadata", None)
        metadata = nested.get("features") if isinstance(nested, Mapping) else getattr(nested, "features", None)
    return metadata if isinstance(metadata, Mapping) else None


def _metadata_fps(source: object) -> float | None:
    fps = source.get("fps") if isinstance(source, Mapping) else getattr(source, "fps", None)
    if fps is None:
        nested = source.get("metadata") if isinstance(source, Mapping) else getattr(source, "metadata", None)
        fps = nested.get("fps") if isinstance(nested, Mapping) else getattr(nested, "fps", None)
    return None if fps is None else float(fps)


def validate_dataset_compatibility(real: object, sim: object) -> None:
    """Validate the metadata needed to mix two LeRobot datasets safely."""
    real_fps, sim_fps = _metadata_fps(real), _metadata_fps(sim)
    if real_fps is not None or sim_fps is not None:
        if real_fps is None or sim_fps is None or not math.isclose(real_fps, sim_fps, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Co-training datasets must use the same fps; real={real_fps}, sim={sim_fps}.")

    real_features, sim_features = _metadata_features(real), _metadata_features(sim)
    if real_features is not None or sim_features is not None:
        if real_features is None or sim_features is None:
            raise ValueError("Co-training datasets must both expose feature metadata.")
        if set(real_features) != set(sim_features):
            raise ValueError(
                "Co-training datasets must have identical feature keys; "
                f"real-only={sorted(set(real_features) - set(sim_features))}, "
                f"sim-only={sorted(set(sim_features) - set(real_features))}."
            )
        for key in sorted(real_features):
            real_shape, sim_shape = _feature_shape(real_features[key]), _feature_shape(sim_features[key])
            if real_shape != sim_shape:
                raise ValueError(
                    f"Co-training feature {key!r} has incompatible shapes: real={real_shape}, sim={sim_shape}."
                )
        action_key = "actions" if "actions" in real_features else "action" if "action" in real_features else None
        if action_key is not None and _feature_shape(real_features[action_key]) != _feature_shape(
            sim_features[action_key]
        ):
            raise ValueError("Co-training datasets must have identical action dimensions.")


def _create_full_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    *,
    repo_id: str,
    dataset_meta: object | None = None,
) -> tuple[Dataset, object]:
    """Load one full LeRobot dataset, leaving episode selection to a frame wrapper."""
    rev = data_config.lerobot_revision
    if dataset_meta is None:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, revision=rev)
    dataset_kwargs = dict(
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        revision=rev,
    )
    return lerobot_dataset.LeRobotDataset(repo_id, **dataset_kwargs), dataset_meta


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    episodes: Sequence[int] | None = None,
    repo_id: str | None = None,
    full_dataset: Dataset | None = None,
    dataset_meta: object | None = None,
) -> Dataset:
    """Create a dataset for training, selecting episodes by full-dataset frame indices."""
    repo_id = repo_id or data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if full_dataset is None:
        full_dataset, dataset_meta = _create_full_torch_dataset(
            data_config,
            action_horizon,
            repo_id=repo_id,
            dataset_meta=dataset_meta,
        )
    if episodes is not None:
        try:
            episode_indices = full_dataset.hf_dataset["episode_index"]  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise TypeError("Episode selection requires the underlying LeRobot dataset.") from exc
        full_dataset = FrameIndexDataset(full_dataset, _frame_indices_for_episodes(episode_indices, episodes))

    if data_config.prompt_from_task:
        if dataset_meta is None:
            raise ValueError("LeRobot metadata is required when prompt_from_task is enabled.")
        full_dataset = TransformedDataset(full_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return full_dataset


def create_cotrain_datasets(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> tuple[Dataset, Dataset]:
    """Create the real and simulation datasets selected by ``data_config.cotrain``."""
    if data_config.cotrain is None:
        raise ValueError("create_cotrain_datasets requires data_config.cotrain.")

    specs = (data_config.cotrain.real, data_config.cotrain.sim)
    metadata: list[object] = []
    metadata_by_repo: dict[str, object] = {}
    full_datasets: dict[str, Dataset] = {}
    episodes = []
    for spec in specs:
        if spec.repo_id not in metadata_by_repo:
            metadata_by_repo[spec.repo_id] = lerobot_dataset.LeRobotDatasetMetadata(
                spec.repo_id, revision=data_config.lerobot_revision
            )
        meta = metadata_by_repo[spec.repo_id]
        metadata.append(meta)
        episodes.append(
            spec.episodes
            if spec.episodes is not None
            else _episodes_for_source(spec.repo_id, meta, spec.episode_source)
            if spec.episode_source is not None
            else None
        )
    validate_dataset_compatibility(metadata[0], metadata[1])
    for spec in specs:
        if spec.repo_id not in full_datasets:
            full_datasets[spec.repo_id], _ = _create_full_torch_dataset(
                data_config,
                action_horizon,
                repo_id=spec.repo_id,
                dataset_meta=metadata_by_repo[spec.repo_id],
            )
    return tuple(
        create_torch_dataset(
            data_config,
            action_horizon,
            model_config,
            repo_id=spec.repo_id,
            episodes=selected_episodes,
            full_dataset=full_datasets[spec.repo_id],
            dataset_meta=metadata_by_repo[spec.repo_id],
        )
        for spec, selected_episodes in zip(specs, episodes, strict=True)
    )  # type: ignore[return-value]


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    cotrain = data_config.cotrain
    if cotrain is None:
        dataset = create_torch_dataset(data_config, action_horizon, model_config)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    else:
        real_dataset, sim_dataset = create_cotrain_datasets(data_config, action_horizon, model_config)
        # Apply exactly the same transform objects and ordering to each side,
        # then concatenate so the domain tag is added after transforms.
        real_dataset = transform_dataset(real_dataset, data_config, skip_norm_stats=skip_norm_stats)
        sim_dataset = transform_dataset(sim_dataset, data_config, skip_norm_stats=skip_norm_stats)
        dataset = MixtureDataset(real_dataset, sim_dataset)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    batch_sampler = None
    sampler_seed = seed
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if batch_size % torch.distributed.get_world_size() != 0:
                raise ValueError("Global batch size must be divisible by the PyTorch world size.")
            if cotrain is None:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            local_batch_size = batch_size // torch.distributed.get_world_size()
            sampler_seed += torch.distributed.get_rank()
        else:
            local_batch_size = batch_size
    else:
        if batch_size % jax.process_count() != 0:
            raise ValueError("Global batch size must be divisible by the JAX process count.")
        local_batch_size = batch_size // jax.process_count()

    if local_batch_size <= 0:
        raise ValueError(f"Global batch size {batch_size} is too small for the active process count.")

    if cotrain is not None:
        # The ratio is defined on each process's local batch, as required by
        # Eq. 7's plain batch mean.  The epoch length only controls RLinf's
        # epoch/offset API; the sampler itself can be resumed mid-epoch.
        epoch_batches = num_batches
        if epoch_batches is None:
            max_size = max(len(dataset.real), len(dataset.sim))
            epoch_batches = max(1, math.ceil(max_size / local_batch_size))
        batch_sampler = StratifiedBatchSampler(
            len(dataset.real),
            len(dataset.sim),
            local_batch_size,
            cotrain.sim_weight,
            seed=sampler_seed,
            num_batches=epoch_batches,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(batch_sampler is None and sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._batch_sampler = batch_sampler

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = dict(
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )
        if batch_sampler is not None:
            if sampler is not None or shuffle:
                raise ValueError("batch_sampler cannot be combined with sampler or shuffle.")
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            # RLinf's OpenPI wrapper forwards set_epoch through the PyTorch
            # ``sampler`` attribute.  The batch sampler drives iteration, but
            # exposing it there preserves that existing resume contract.
            self._data_loader.sampler = batch_sampler
        else:
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
                sampler=sampler,
                drop_last=True,
                **loader_kwargs,
            )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        sampler = self._data_loader.batch_sampler if self._batch_sampler is not None else self._data_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def state_dict(self) -> dict[str, object]:
        sampler = self._data_loader.batch_sampler if self._batch_sampler is not None else self._data_loader.sampler
        state: dict[str, object] = {"epoch": getattr(self, "_epoch", 0)}
        if hasattr(sampler, "state_dict"):
            state["sampler"] = sampler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        sampler = self._data_loader.batch_sampler if self._batch_sampler is not None else self._data_loader.sampler
        sampler_state = state.get("sampler")
        if sampler_state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(typing.cast(dict[str, int], sampler_state))

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._data_loader, "set_epoch"):
            self._data_loader.set_epoch(epoch)

    def state_dict(self) -> dict[str, object]:
        if hasattr(self._data_loader, "state_dict"):
            return self._data_loader.state_dict()
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if hasattr(self._data_loader, "load_state_dict"):
            self._data_loader.load_state_dict(state)

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if DOMAIN_KEY in batch:
                yield observation, batch["actions"], batch[DOMAIN_KEY]
            else:
                yield observation, batch["actions"]
