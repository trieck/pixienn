import unittest

import numpy as np

from pixienn import CudaTensor, Tensor


class TensorTests(unittest.TestCase):
    @unittest.skipUnless(CudaTensor is not None,
                         "PixieNN Python bindings were built without CUDA")
    def test_cuda_tensor_matches_cpu_tensor(self):
        values = [float(index) - 3.5 for index in range(24)]
        cpu = Tensor([2, 3, 4], values)
        cuda = CudaTensor([2, 3, 4], values)

        self.assertEqual(cuda.device, "cuda")
        self.assertEqual(cuda.shape, cpu.shape)
        self.assertEqual(cuda.strides, cpu.strides)
        self.assertEqual(cuda.ndim, cpu.ndim)
        self.assertEqual(cuda.size, cpu.size)
        self.assertEqual(cuda.values(), cpu.values())

    @unittest.skipUnless(CudaTensor is not None,
                         "PixieNN Python bindings were built without CUDA")
    def test_cuda_fill_and_clone_match_cpu_tensor(self):
        cpu = Tensor([2, 2], 1.5)
        cuda = CudaTensor([2, 2], 1.5)
        cpu.fill(-2.0)
        cuda.fill(-2.0)
        self.assertEqual(cuda.values(), cpu.values())
        self.assertEqual(cuda.clone().values(), cpu.clone().values())

    def test_shape_strides_and_size(self):
        tensor = Tensor([2, 3, 4])
        self.assertEqual(tensor.shape, [2, 3, 4])
        self.assertEqual(tensor.strides, [12, 4, 1])
        self.assertEqual(tensor.ndim, 3)
        self.assertEqual(tensor.size, 24)
        self.assertEqual(tensor.dim_size(0), 2)
        self.assertEqual(tensor.dim_size(2), 4)
        self.assertEqual(tensor.device, "cpu")
        with self.assertRaises(IndexError):
            tensor.dim_size(3)

    def test_fill_and_values(self):
        tensor = Tensor([2, 2], 1.5)
        self.assertEqual(tensor.values(), [1.5] * 4)
        tensor.fill(-2.0)
        self.assertEqual(tensor.values(), [-2.0] * 4)

    def test_indexing(self):
        tensor = Tensor([2, 2], 0.0)
        tensor[0] = 3.5
        tensor[-1] = -1.25
        tensor[1, 0] = 6.0
        self.assertEqual(tensor[0], 3.5)
        self.assertEqual(tensor[-1], -1.25)
        self.assertEqual(tensor[1, 0], 6.0)
        self.assertEqual(tensor[-1, -2], 6.0)
        self.assertEqual(len(tensor), 4)
        with self.assertRaises(IndexError):
            _ = tensor[4]
        with self.assertRaises(IndexError):
            _ = tensor[0, 0, 0]
        with self.assertRaises(IndexError):
            _ = tensor[2, 0]

    def test_buffer_protocol(self):
        tensor = Tensor([2, 2], 1.0)
        view = memoryview(tensor)
        self.assertEqual(view.format, "f")
        self.assertEqual(view.shape, (2, 2))
        self.assertEqual(view.strides, (8, 4))
        self.assertEqual(view.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        view[0, 1] = 7.5
        self.assertEqual(tensor[1], 7.5)

    def test_construct_from_values(self):
        tensor = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(tensor.values(), [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            Tensor([2, 2], [1.0, 2.0])

    def test_scalar_arithmetic(self):
        tensor = Tensor([2], [2.0, 4.0])
        self.assertEqual((tensor + 1.0).values(), [3.0, 5.0])
        self.assertEqual((tensor - 1.0).values(), [1.0, 3.0])
        self.assertEqual((tensor * 2.0).values(), [4.0, 8.0])
        self.assertEqual((tensor / 2.0).values(), [1.0, 2.0])
        with self.assertRaises(ValueError):
            tensor / 0.0

    def test_clone_is_independent(self):
        tensor = Tensor([2], [1.0, 2.0])
        clone = tensor.clone()
        clone[0] = 9.0
        self.assertEqual(tensor.values(), [1.0, 2.0])
        self.assertEqual(clone.values(), [9.0, 2.0])

    def test_numpy_interoperability_is_zero_copy(self):
        tensor = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
        array = np.asarray(tensor)
        self.assertEqual(array.shape, (2, 2))
        self.assertEqual(array.dtype, np.float32)
        array[1, 0] = 8.0
        self.assertEqual(tensor[2], 8.0)

    def test_rank_validation(self):
        with self.assertRaises(ValueError):
            Tensor([])
        with self.assertRaises(ValueError):
            Tensor([1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
