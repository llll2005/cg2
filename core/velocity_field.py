import cupy as cp

class VelocityField:
    def __init__(self, num_faces, face_normals, face_to_cells,
                 cell_face_offsets=None, cell_face_indices=None):
        """初始化速度場"""
        self.u_normal = cp.zeros(num_faces, dtype=cp.float64)
        self.face_normals = cp.array(face_normals, dtype=cp.float64)
        self.face_to_cells = cp.array(face_to_cells, dtype=cp.int32)

        if cell_face_offsets is None or cell_face_indices is None:
            raise ValueError("Must provide cell_face_offsets and cell_face_indices")

        self.cell_face_offsets = cp.array(cell_face_offsets, dtype=cp.int32)
        self.cell_face_indices = cp.array(cell_face_indices, dtype=cp.int32)

    def extrapolate_ghost_cells(self):
        """CuPy 版本的幽靈單元外插"""
        self.u_normal = _extrapolate_ghost_cells_cupy(
            self.u_normal, self.face_to_cells,
            self.cell_face_offsets, self.cell_face_indices
        )

def _extrapolate_ghost_cells_cupy(u_normal, face_to_cells, cell_face_offsets, cell_face_indices):
    """CuPy 版幽靈單元外插"""
    ghost_mask = (face_to_cells[:, 1] == -1)
    ghost_indices = cp.where(ghost_mask)[0]

    for i in ghost_indices:
        a = face_to_cells[i, 0]
        start = cell_face_offsets[a]
        end = cell_face_offsets[a + 1]

        neighbor_faces = cell_face_indices[start:end]
        neighbor_mask = (face_to_cells[neighbor_faces, 1] != -1)
        valid_faces = neighbor_faces[neighbor_mask]

        if len(valid_faces) > 0:
            u_normal[i] = cp.mean(u_normal[valid_faces])
        else:
            u_normal[i] = 0.0

    return u_normal