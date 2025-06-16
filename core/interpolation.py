import cupy as cp
from cupyx.scipy.sparse.linalg import lsqr

class VelocityInterpolator:
    def __init__(self, mesh, velocity_field):
        self.mesh = mesh
        self.velocity_field = velocity_field
        self.epsilon_factor = cp.float64(0.03)

    def interpolate_velocity(self, point):
        """Interpolate velocity at an arbitrary point using MLS."""
        cell = self.mesh.locate_cell(point)
        if cell is None:
            return cp.zeros(3)

        face_ids = self.mesh.get_faces_around_point(point, rings=1)
        if len(face_ids) < 3:
            return cp.zeros(3)

        positions = cp.array(self.mesh.face_centers[face_ids] - point)
        normals = cp.array(self.velocity_field.face_normals[face_ids], dtype=cp.float64)
        values = cp.array(self.velocity_field.u_normal[face_ids])

        avg_edge_length = cp.float64(self.mesh.get_average_edge_length_around_cell(cell))
        epsilon = self.epsilon_factor * avg_edge_length

        return self._linear_mls_interpolation(positions, normals, values, epsilon)

    def solve_least_squares(A, b):
        """加權最小二乘解"""
        AtA = A.T @ A
        Atb = A.T @ b
        solution = cp.linalg.solve(AtA, Atb)
        return solution

    def _linear_mls_interpolation(self, positions, normals, values, epsilon):
        n_faces = len(positions)
        if n_faces < 4:
            return self._constant_mls_interpolation(positions, normals, values, epsilon)

        A = cp.zeros((n_faces, 12), dtype=cp.float64)
        weights = cp.zeros(n_faces)

        for i in range(n_faces):
            x, y, z = positions[i]
            n = normals[i]
            r_i = cp.linalg.norm(positions[i])
            weights[i] = 1.0 / (r_i**2 + epsilon**2)

            A[i, 0:3] = n
            A[i, 3:6] = x * n
            A[i, 6:9] = y * n
            A[i, 9:12] = z * n

        W = cp.diag(cp.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ values

        try:
            coeffs = self.solve_least_squares(A_weighted, b_weighted)  # ✅ 使用手動 `lstsq()` 或 `lsqr()`
            rank = cp.linalg.matrix_rank(A_weighted, tol=1e-12)  # ✅ 加入數值穩定性閾值

            if rank < 12:
                print(f"Warning: Matrix rank too low ({rank}), switching to constant interpolation")  # ✅ 提示問題
                return self._constant_mls_interpolation(positions, normals, values, epsilon)

            return coeffs[:3]  # ✅ 確保回傳的是前 3 個分量
        except Exception as e:
            print(f"Error in MLS interpolation: {e}")  # ✅ 記錄錯誤，方便除錯
            return cp.zeros(3)

    def _constant_mls_interpolation(self, positions, normals, values, epsilon):
        """CuPy Fallback: constant-weighted least squares."""
        n_faces = len(positions)
        A = cp.zeros((n_faces, 3), dtype=cp.float64)
        weights = cp.zeros(n_faces)

        for i in range(n_faces):
            r_i = cp.linalg.norm(positions[i])
            weights[i] = 1.0 / (r_i**2 + epsilon**2)
            A[i, :] = normals[i]

        W = cp.diag(cp.sqrt(weights))
        A_weighted = W @ A
        b_weighted = W @ values

        try:
            u = self.solve_least_squares(A_weighted, b_weighted)  # ✅ 使用手動 `lstsq()` 或 `lsqr()`
            _, s, _ = cp.linalg.svd(A_weighted)  # ✅ SVD 計算奇異值
            rank = cp.sum(s > 1e-12)  # ✅ 過濾小於閾值的特徵值

            if rank < 3:
                print(f"Warning: Matrix rank too low ({rank}), switching to zero vector")  # ✅ 輸出警告
                return cp.zeros(3)

            return u
        except Exception as e:
            print(f"Error in weighted least squares: {e}")  # ✅ 記錄錯誤
            return cp.zeros(3)
