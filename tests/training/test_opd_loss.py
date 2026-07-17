from __future__ import annotations

import unittest

import numpy as np

from my_agent.training.opd_loss import (
    chunked_full_vocab_kl,
    chunked_hidden_state_kl,
    gather_completion_logits,
)


class OPDLossTests(unittest.TestCase):
    def test_chunked_kl_matches_independent_fp32_reference(self) -> None:
        teacher = np.array([[[2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 2.0, -2.0]]], dtype=np.float16)
        student = np.array([[[1.5, 0.2, -0.4, 0.1], [0.4, 0.5, 1.4, -1.0]]], dtype=np.float16)
        mask = np.array([[1, 1]], dtype=np.int64)

        output = chunked_full_vocab_kl(
            teacher,
            student,
            mask,
            vocab_chunk_size=2,
        )
        teacher32 = teacher.astype(np.float32)
        student32 = student.astype(np.float32)
        teacher_logp = teacher32 - _logsumexp(teacher32, axis=-1)[..., None]
        student_logp = student32 - _logsumexp(student32, axis=-1)[..., None]
        reference = np.mean(np.sum(
            np.exp(teacher_logp) * (teacher_logp - student_logp),
            axis=-1,
        ))

        self.assertAlmostEqual(output.loss, float(reference), places=6)

    def test_teacher_is_stop_gradient_and_student_receives_gradient(self) -> None:
        import torch

        teacher = torch.nn.Parameter(torch.tensor([[[2.0, 0.0, -1.0]]]))
        student = torch.nn.Parameter(torch.tensor([[[0.0, 1.0, -0.5]]]))
        mask = torch.ones((1, 1), dtype=torch.long)

        output = chunked_full_vocab_kl(
            teacher,
            student,
            mask,
            vocab_chunk_size=2,
            torch_module=torch,
        )
        output.loss.backward()

        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_hidden_state_kl_matches_full_logits_without_teacher_gradient(self) -> None:
        import torch

        teacher_hidden = torch.nn.Parameter(torch.tensor([[[1.0, -0.5]]]))
        student_hidden = torch.nn.Parameter(torch.tensor([[[0.2, 0.7]]]))
        weight = torch.tensor([
            [0.5, -0.2],
            [0.1, 0.8],
            [-0.4, 0.3],
            [0.9, 0.2],
        ])
        mask = torch.ones((1, 1), dtype=torch.long)

        output = chunked_hidden_state_kl(
            teacher_hidden,
            student_hidden,
            weight,
            None,
            mask,
            vocab_chunk_size=2,
            torch_module=torch,
        )
        reference = chunked_full_vocab_kl(
            torch.nn.functional.linear(teacher_hidden, weight),
            torch.nn.functional.linear(student_hidden, weight),
            mask,
            vocab_chunk_size=4,
            torch_module=torch,
        )
        output.loss.backward()

        self.assertAlmostEqual(
            float(output.loss.detach()),
            float(reference.loss.detach()),
            places=6,
        )
        self.assertIsNone(teacher_hidden.grad)
        self.assertIsNotNone(student_hidden.grad)
        self.assertGreater(float(student_hidden.grad.abs().sum()), 0.0)

    def test_full_distribution_matters_even_when_teacher_argmax_is_same(self) -> None:
        student = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        concentrated = np.array([[[4.0, 0.0, -1.0]]], dtype=np.float32)
        diffuse = np.array([[[0.5, 0.4, 0.0]]], dtype=np.float32)
        mask = np.ones((1, 1), dtype=np.int64)

        first = chunked_full_vocab_kl(concentrated, student, mask).loss
        second = chunked_full_vocab_kl(diffuse, student, mask).loss

        self.assertEqual(int(concentrated.argmax()), int(diffuse.argmax()))
        self.assertNotAlmostEqual(first, second, places=5)

    def test_gather_uses_prompt_specific_prediction_indexes(self) -> None:
        logits = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        indexes = np.array([[0, 2], [1, -1]])
        mask = np.array([[1, 1], [1, 0]])

        gathered = gather_completion_logits(logits, indexes, mask)

        np.testing.assert_array_equal(gathered[0, 0], logits[0, 0])
        np.testing.assert_array_equal(gathered[0, 1], logits[0, 2])
        np.testing.assert_array_equal(gathered[1, 0], logits[1, 1])


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis)
    return maximum + np.log(np.sum(np.exp(values - maximum[..., None]), axis=axis))


if __name__ == "__main__":
    unittest.main()
