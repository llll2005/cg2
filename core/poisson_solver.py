import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cpla

class PoissonSolver:
    def __init__(self, divergence_matrix, gradient_matrix, mesh, dt=0.1, rho=1.0):
        """初始化泊松求解器（CuPy 版）"""
        self.D = divergence_matrix
        self.G = gradient_matrix
        self.mesh = mesh
        self.dt = cp.float64(dt)
        self.rho = cp.float64(rho)

        # ✅ CuPy 稀疏矩陣
        self.poisson_matrix = self._build_poisson_matrix()

    def _build_poisson_matrix(self):
        """構建對稱正定的泊松矩陣（CuPy 稀疏格式）"""
        V_diag = csp.diags(1.0 / cp.array(self.mesh.cell_volumes), format='csr')
        D_scaled = V_diag @ self.D
        return D_scaled @ self.G

    def solve_pressure(self, u_star):
        """求解壓力修正（CuPy 共軛梯度法）"""
        u_star = cp.array(u_star, dtype=cp.float64)  # ✅ 確保數據在 GPU
        divergence = self.D @ u_star
        rhs = (self.rho / self.dt) * divergence

        pressure, info = cpla.cg(self.poisson_matrix, rhs, tol=1e-6, maxiter=1000)

        if info != 0:
            print(f"Warning: Pressure solver convergence issue, info={info}")

        return pressure

    def correct_velocity(self, u_star, pressure):
        """修正速度場"""
        grad_p = self.G @ pressure
        return u_star - (self.dt / self.rho) * grad_p

    def apply_boundary_conditions(self, pressure, u_corrected):
        """應用邊界條件"""
        for i, (a, b) in enumerate(self.mesh.face_to_cells):
            if a == b or b == -1:
                if hasattr(self.mesh, 'is_closed_boundary') and self.mesh.is_closed_boundary(i):
                    u_corrected[i, :] = 0.0  # ✅ 保持在 GPU 運算
        return u_corrected