import cupy as cp
from enum import Enum
from mesh.mesh import HybridMesh

class CellType(Enum):
    HEXAHEDRAL = 1
    TETRAHEDRAL = 2
    TRANSITION = 3

class SemiLagrangianAdvection:
    def __init__(self, mesh, velocity_field):
        self.mesh = mesh
        self.velocity_field = velocity_field

        self.vertex_velocities = cp.zeros((len(self.mesh.vertices), 3), dtype=cp.float64)
        self.vertex_velocities_valid = False

    def _update_vertex_velocities(self):
        # MLS計算頂點速度
        if self.vertex_velocities_valid:
            return

        vertices = cp.array(self.mesh.vertices, dtype=cp.float64)
        adjacent_faces_list = [self.mesh.get_adjacent_faces(v_idx) for v_idx in range(len(vertices))]
        velocities = cp.array([
            self._mls_constant_interpolation(vertices[v_idx], cp.array(adj_faces))
            if len(adj_faces) > 0 else cp.zeros(3)
            for v_idx, adj_faces in enumerate(adjacent_faces_list)
        ])

        self.vertex_velocities[:] = velocities
        self.vertex_velocities_valid = True

    def _mls_constant_interpolation(self, position, face_indices):
        # 移動最小二乘插值
        if len(face_indices) == 0:
            return cp.zeros(3)

        normals = cp.array([self.mesh.face_normals[i] for i in face_indices])
        z = cp.array([self.velocity_field.u_normal[i] for i in face_indices])
        weights = self._compute_weights(position, face_indices)

        return _solve_weighted_least_squares(normals, z, weights)

    def _compute_weights(self, position, face_indices):
        if len(face_indices) == 0:
            return cp.array([])

        face_centers = cp.array([self.mesh.face_centers[i] for i in face_indices])
        distances = cp.linalg.norm(face_centers - position, axis=1)
        avg_edge_length = self._get_average_edge_length()
        epsilon = 0.03 * avg_edge_length

        weights = 1.0 / (distances**2 + epsilon**2)
        return weights

    def _get_average_edge_length(self):
        if not hasattr(self.mesh, 'global_avg_edge_length'):
            total_length = cp.zeros(1)
            edge_count = cp.zeros(1)

            for cell_idx in range(len(self.mesh.cells)):
                edges = self.mesh.get_cell_edges(cell_idx)
                for v1, v2 in edges:
                    total_length += cp.linalg.norm(self.mesh.vertices[v1] - self.mesh.vertices[v2])
                    edge_count += 1

            self.mesh.global_avg_edge_length = (total_length / edge_count).item()  # 轉回 Python float
        return self.mesh.global_avg_edge_length

    def advect_velocity(self, dt):
        self._update_vertex_velocities()
        face_centers = cp.array(self.mesh.face_centers, dtype=cp.float64)
        face_normals = cp.array(self.mesh.face_normals, dtype=cp.float64)
        new_velocity = _advect_velocity_cupy(face_centers, face_normals, dt, self.mesh, self.velocity_field, self)

        self.velocity_field.set_normal_component(new_velocity)
        self.vertex_velocities_valid = False

    def advect_particles(self, particles, dt):
        self._update_vertex_velocities()
        positions = cp.array([p.position for p in particles], dtype=cp.float64)

        new_positions = _advect_particles_cupy(positions, dt, self.mesh, self.vertex_velocities)
        for i, particle in enumerate(particles):
            particle.position = new_positions[i].get()  # 轉回 NumPy

def _solve_weighted_least_squares(normals, z, weights):
    n_faces, n_dim = normals.shape
    NtWN = cp.zeros((n_dim, n_dim), dtype=cp.float64)
    NtWz = cp.zeros(n_dim, dtype=cp.float64)

    for i in range(n_faces):
        w = weights[i]
        n = normals[i]
        NtWN += w * cp.outer(n, n)
        NtWz += w * n * z[i]

    return cp.linalg.solve(NtWN, NtWz)

def _advect_velocity_cupy(face_centers, face_normals, dt, mesh, velocity_field, advector):
    n_faces = face_centers.shape[0]
    new_velocity = cp.zeros(n_faces, dtype=cp.float64)

    for i in cp.arange(n_faces):  # 使用 CuPy 向量化
        face_center = face_centers[i]
        face_normal = face_normals[i]
        back_pos = _backward_trace_cupy(face_center, dt, advector)
        interpolated_velocity = _interpolate_at_point_cupy(back_pos, mesh, advector)
        new_velocity[i] = cp.dot(interpolated_velocity, face_normal)

    return new_velocity

def _backward_trace_cupy(start_pos, dt, advector):
    velocity = _interpolate_at_point_cupy(start_pos, advector.mesh, advector)
    return start_pos - dt * velocity

def _interpolate_at_point_cupy(position, mesh, advector):
    closest_face_idx = cp.argmin(cp.linalg.norm(mesh.face_centers - position, axis=1))
    face_normal = mesh.face_normals[closest_face_idx]
    velocity_at_face = advector.velocity_field.u_normal[closest_face_idx]

    return velocity_at_face * face_normal

def _advect_particles_cupy(positions, dt, mesh, vertex_velocities):
    new_positions = positions + dt * cp.array([_interpolate_velocity_at_particle_cupy(p, mesh, vertex_velocities) for p in positions])
    return new_positions

def _interpolate_velocity_at_particle_cupy(position, mesh, vertex_velocities):
    closest_vertex_idx = cp.argmin(cp.linalg.norm(mesh.vertices - position, axis=1))
    return vertex_velocities[closest_vertex_idx]