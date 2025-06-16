import cupy as cp

class ForceApplicator:
    def __init__(self, gravity=None, reference_density=1.0):
        """初始化力場，改用 CuPy 儲存重力與參考密度"""
        if gravity is None:
            gravity = cp.array([0.0, 0.0, -9.8], dtype=cp.float64)
        self.gravity = gravity
        self.reference_density = cp.float64(reference_density)

    def apply_body_forces(self, velocity_field, density_field, mesh, dt):
        face_to_cells = cp.array(velocity_field.face_to_cells)
        face_normals = cp.array(velocity_field.face_normals)
        cell_volumes = cp.array(mesh.cell_volumes)
        densities = cp.array(density_field.rho)

        # GPU 並行計算
        u_normal = velocity_field.u_normal
        dt = cp.float64(dt)

        # 計算浮力
        buoyancy_a = (self.reference_density - densities) * self.gravity / densities
        buoyancy_b = buoyancy_a.copy()

        # 索引修正
        a = face_to_cells[:, 0]
        b = face_to_cells[:, 1]
        b[b == -1] = a[b == -1]  # 邊界條件

        # 體積修正
        V_a = cell_volumes[a]
        V_b = cell_volumes[b]

        total_force_a = V_a[:, None] * buoyancy_a[a]
        total_force_b = V_b[:, None] * buoyancy_b[b]

        avg_acceleration = (total_force_a + total_force_b) / (densities[:, None] * (V_a[:, None] + V_b[:, None]))

        normal_accel = cp.sum(face_normals * avg_acceleration, axis=1)
        u_normal += normal_accel * dt

    def step(self, velocity_field, density_field, mesh, dt, time=0.0):
        """執行所有力場更新"""
        self.apply_body_forces(velocity_field, density_field, mesh, dt)