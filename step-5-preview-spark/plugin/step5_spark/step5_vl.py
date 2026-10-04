# SPDX-License-Identifier: Apache-2.0
"""Step-5 vision-language wrapper: the Step-3.7 perception-encoder VL model with the Step-5 text model.

Images go through the unchanged Step-3-VL path (728x728 global view + optional 504x504 patches).

Video support (the checkpoint has no temporal module): the video loader samples frames (vLLM default: 32 frames
uniformly, override with --media-io-kwargs '{"video": {"num_frames": N}}' or {"fps": F}); each frame is encoded through
the image encoder as a global 728x728 view only (no high-res patches), and the video placeholder is replaced with the
concatenation of per-frame image token blocks, each exactly what a patch-less image produces
(<im_start> <im_patch>*169 <im_end>). The video embedding is the concatenation of the per-frame image embeddings.

The video placeholder in prompts is STEP5_VIDEO_PLACEHOLDER ("<im_start><im_end>", two special tokens that never
appear adjacent in an expanded image). The shipped chat template only renders image parts, so step5_spark.__init__
rewrites OpenAI-style video parts into that placeholder text before the template runs.
"""
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate, PromptUpdateDetails
from vllm.model_executor.models.step3_vl import (
    Step3VLDummyInputsBuilder,
    Step3VLMultiModalProcessor,
    Step3VLProcessingInfo,
)
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration

STEP5_VIDEO_PLACEHOLDER = "<im_start><im_end>"
DEFAULT_MAX_VIDEO_FRAMES = 32  # matches vLLM's VideoMediaIO default num_frames
_ENCODER_FRAME_CHUNK = 8  # frames per vision-encoder forward (bounds activation memory)


class Step5VLProcessingInfo(Step3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_num_frame_tokens(self) -> int:
        # Token block of one patch-less image: <im_start> + image features + <im_end>.
        return self.get_image_processor().num_image_feature_size + 2

    def get_max_video_frames(self) -> int:
        n = DEFAULT_MAX_VIDEO_FRAMES
        try:
            vk = (self.ctx.get_mm_config().media_io_kwargs or {}).get("video") or {}
            if int(vk.get("num_frames", -1)) > 0:
                n = int(vk["num_frames"])
        except Exception:
            pass
        max_len = self.ctx.model_config.max_model_len
        return max(1, min(n, max_len // self.get_num_frame_tokens()))

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_frames() * self.get_num_frame_tokens(),
        }


class Step5VLDummyInputsBuilder(Step3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<im_patch>" * mm_counts.get("image", 0) + STEP5_VIDEO_PLACEHOLDER * mm_counts.get("video", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
        num_videos = mm_counts.get("video", 0)
        if num_videos:
            size = self.info.get_image_processor().image_size
            data["video"] = self._get_dummy_videos(
                width=size,
                height=size,
                num_frames=self.info.get_max_video_frames(),
                num_videos=num_videos,
                overrides=mm_options.get("video"),
            )
        return data


def _frames_to_numpy(video: Any) -> np.ndarray:
    if isinstance(video, tuple):  # (frames, metadata)
        video = video[0]
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if isinstance(video, list):
        video = np.stack([np.asarray(f.convert("RGB")) if hasattr(f, "convert") else np.asarray(f) for f in video])
    video = np.asarray(video)
    if video.ndim != 4:
        raise ValueError(f"Step-5 video input must be (T, H, W, C) frames, got shape {video.shape}")
    if video.shape[-1] not in (1, 3, 4) and video.shape[1] in (1, 3, 4):  # (T, C, H, W) -> (T, H, W, C)
        video = video.transpose(0, 2, 3, 1)
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    elif video.shape[-1] == 4:
        video = video[..., :3]
    if video.dtype != np.uint8:
        video = np.clip(video, 0, 255).astype(np.uint8)
    return video


class Step5VLMultiModalProcessor(Step3VLMultiModalProcessor):
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        counts = mm_items.get_all_counts()
        num_videos = counts.get("video", 0)
        if not num_videos:
            return super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)

        out = super()._apply_hf_processor_main(
            mm_items.select({k for k in counts if k != "video"}), hf_processor_mm_kwargs
        )
        image_processor = self.info.get_image_processor()
        max_frames = self.info.get_max_video_frames()
        frames_pv: list[torch.Tensor] = []
        num_frames: list[int] = []
        for video in mm_items["video"].get_all():
            frames = _frames_to_numpy(video)
            if len(frames) == 0:
                raise ValueError("Step-5 video input has no frames")
            if len(frames) > max_frames:
                frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
            # Global 728x728 view per frame, same transform as an image's global view, no patches.
            frames_pv.extend(image_processor.image_preprocessor(np.ascontiguousarray(f))["pixel_values"] for f in frames)
            num_frames.append(len(frames))
        out["pixel_values_videos"] = torch.cat(frames_pv)
        out["video_num_frames"] = torch.tensor(num_frames, dtype=torch.long)
        return out

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        video_num_frames = hf_inputs.get("video_num_frames", torch.empty(0, dtype=torch.long))
        fields.update(
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes("video", video_num_frames),
            video_num_frames=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        frame_ids = hf_processor.get_image_repl_feature_ids(1, 0, [])
        target = [hf_processor.image_start_token_id, hf_processor.image_end_token_id]

        def get_replacement_video(item_idx: int):
            n = int(out_mm_kwargs["video"][item_idx]["video_num_frames"].data)
            return PromptUpdateDetails.select_token_id(
                seq=frame_ids * n,
                embed_token_id=hf_processor.image_token_id,
            )

        updates.append(PromptReplacement(modality="video", target=target, replacement=get_replacement_video))
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    Step5VLMultiModalProcessor,
    info=Step5VLProcessingInfo,
    dummy_inputs=Step5VLDummyInputsBuilder,
)
class Step5ForConditionalGeneration(Step3p7ForConditionalGeneration):
    STEP5_VIDEO_PLACEHOLDER = STEP5_VIDEO_PLACEHOLDER

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<im_patch>"
        if modality.startswith("video"):
            return STEP5_VIDEO_PLACEHOLDER
        raise ValueError("Only image and video modalities are supported")

    def _process_video_input(self, pixel_values_videos: torch.Tensor, video_num_frames) -> list[torch.Tensor]:
        pv = pixel_values_videos.to(self.dtype)
        feats = []
        for s in range(0, pv.shape[0], _ENCODER_FRAME_CHUNK):
            f = self._process_image_features(self._get_vision_model_output(pv[s : s + _ENCODER_FRAME_CHUNK]))
            feats.append(f.reshape(f.shape[0], -1, f.shape[-1]))
        feats = torch.cat(feats)  # (total_frames, tokens_per_frame, hidden)
        sizes = [int(n) for n in (video_num_frames.tolist() if torch.is_tensor(video_num_frames) else video_num_frames)]
        return [v.reshape(-1, v.shape[-1]) for v in torch.split(feats, sizes)]

    def embed_multimodal(self, **kwargs):
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_num_frames = kwargs.pop("video_num_frames", None)
        kwargs.pop("video_embeds", None)
        out = []
        seen = set()
        # Keep modality order as given (the runner normally groups one modality per call).
        for key in list(kwargs.keys()) + ["pixel_values_videos"]:
            if key in ("pixel_values", "image_embeds") and "image" not in seen:
                seen.add("image")
                out.extend(super().embed_multimodal(**kwargs))
            elif key == "pixel_values_videos" and "video" not in seen and pixel_values_videos is not None:
                seen.add("video")
                if isinstance(pixel_values_videos, (list, tuple)):
                    pixel_values_videos = torch.cat(list(pixel_values_videos))
                out.extend(self._process_video_input(pixel_values_videos, video_num_frames))
        return out
