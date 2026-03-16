from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from rl_vla_bootstrapping.cli.run_cdpr_policy import (
    _FallbackGenerateConfig,
    _load_generate_config,
    _predict_normalized_action_chunk,
    _resolve_llm_dim,
    _set_num_images_in_input,
)


class PolicyRunnerConfigTests(unittest.TestCase):
    def test_load_generate_config_falls_back_without_libero(self):
        with mock.patch(
            "rl_vla_bootstrapping.cli.run_cdpr_policy.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'libero'"),
        ):
            config_cls, note = _load_generate_config()

        self.assertIs(config_cls, _FallbackGenerateConfig)
        self.assertIsNotNone(note)
        self.assertIn("libero", note)

    def test_load_generate_config_prefers_openvla_definition_when_available(self):
        fake_module = types.SimpleNamespace(GenerateConfig=object)
        with mock.patch(
            "rl_vla_bootstrapping.cli.run_cdpr_policy.importlib.import_module",
            return_value=fake_module,
        ):
            config_cls, note = _load_generate_config()

        self.assertIs(config_cls, object)
        self.assertIsNone(note)

    def test_resolve_llm_dim_from_wrapped_text_config(self):
        leaf = types.SimpleNamespace(
            config=types.SimpleNamespace(
                text_config=types.SimpleNamespace(hidden_size=4096),
            )
        )
        wrapped = types.SimpleNamespace(base_model=leaf)

        self.assertEqual(_resolve_llm_dim(wrapped), 4096)

    def test_set_num_images_in_input_updates_wrapped_backbone(self):
        backbone = types.SimpleNamespace(num_images_in_input=1)
        wrapped = types.SimpleNamespace(base_model=types.SimpleNamespace(vision_backbone=backbone))

        updated = _set_num_images_in_input(wrapped, 2)

        self.assertEqual(updated, 2)
        self.assertEqual(backbone.num_images_in_input, 2)

    def test_set_num_images_in_input_falls_back_to_one_without_api(self):
        updated = _set_num_images_in_input(types.SimpleNamespace(), 2)
        self.assertEqual(updated, 1)

    def test_predict_normalized_action_chunk_uses_action_head_path(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this environment")

        class _FakeProcessor:
            def __call__(self, prompt, image, return_tensors="pt"):
                del prompt, image, return_tensors
                return {
                    "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                    "pixel_values": torch.zeros((1, 1, 2, 2), dtype=torch.float32),
                }

        class _FakeLanguageModel:
            def __call__(self, **kwargs):
                inputs_embeds = kwargs["inputs_embeds"]
                hidden = torch.arange(
                    inputs_embeds.numel(), dtype=torch.float32
                ).reshape(inputs_embeds.shape)
                return types.SimpleNamespace(hidden_states=[hidden])

        class _FakeCoreModel:
            def __init__(self):
                self.language_model = _FakeLanguageModel()

            def get_input_embeddings(self):
                def _embed(input_ids):
                    bsz, seq_len = input_ids.shape
                    return torch.zeros((bsz, seq_len, 3), dtype=torch.float32)

                return _embed

            def _process_vision_features(self, pixel_values, language_embeddings, use_film=False):
                del pixel_values, language_embeddings, use_film
                return torch.zeros((1, 2, 3), dtype=torch.float32)

            def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attn_prep):
                del attn_prep
                multimodal_embeddings = torch.cat(
                    [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
                    dim=1,
                )
                multimodal_attention = torch.ones(
                    (multimodal_embeddings.shape[0], multimodal_embeddings.shape[1]),
                    dtype=torch.long,
                )
                return multimodal_embeddings, multimodal_attention

        class _FakeActionHead:
            def predict_action(self, action_hidden_states):
                self.last_shape = tuple(action_hidden_states.shape)
                return torch.full((1, 2, 5), 0.5, dtype=torch.float32)

        action_head = _FakeActionHead()
        obs = {
            "full_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

        chunk = _predict_normalized_action_chunk(
            vla=_FakeCoreModel(),
            processor=_FakeProcessor(),
            action_head=action_head,
            obs=obs,
            instruction="pick up apple",
            chunk_length=2,
            num_images_in_input=1,
            device=torch.device("cpu"),
            pixel_dtype=torch.float32,
        )

        self.assertEqual(action_head.last_shape, (1, 10, 3))
        self.assertEqual(chunk.shape, (2, 5))
        self.assertTrue(np.allclose(chunk, np.tanh(0.5)))


if __name__ == "__main__":
    unittest.main()
