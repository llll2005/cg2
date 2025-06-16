import cupy as cp

class DensityField:
    def __init__(self, num_cells):
        """使用 CuPy 陣列來存儲密度場"""
        self.rho = cp.zeros(num_cells, dtype=cp.float64)

    def add_source(self, indices, amount):
        """向指定索引的 cell 添加額外密度"""
        indices = cp.array(indices, dtype=cp.int32)
        valid_mask = (indices >= 0) & (indices < len(self.rho))
        valid_indices = indices[valid_mask]

        cp.scatter_add(self.rho, valid_indices, amount)  # CuPy 的 GPU 版本 scatter_add

    def advect(self, advected_rho):
        """密度場的對流更新"""
        self.rho[:] = cp.array(advected_rho, dtype=cp.float64)

    def get_values(self):
        """返回 NumPy 陣列，方便後續 VDB 或 CPU 處理"""
        return self.rho.get()