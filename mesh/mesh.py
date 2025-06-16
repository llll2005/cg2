import cupy as cp

class HybridMesh:
    def __init__(self, vertices, faces, cells, face_to_cells):
        self.vertices = cp.array(vertices, dtype=cp.float64)
        self.faces = faces
        self.cells = cells
        self.face_to_cells = cp.array(face_to_cells, dtype=cp.int32)
        self.num_cells = len(cells)
        self.num_faces = len(faces)
        self.cell_centers = self.compute_cell_centers()
        self.face_centers = self.compute_face_centers()
        self.face_normals = self.compute_face_normals()
        # 建立 cell_to_faces CSR 結構
        self.cell_face_offsets, self.cell_face_indices = self.build_cell_to_faces()


    def build_cell_to_faces(self):
        # 將 cell_to_faces 轉換為 CSR 格式
        temp = [[] for _ in range(self.num_cells)]
        for f_idx, (a, b) in enumerate(self.face_to_cells):
            if a >= 0:
                temp[int(a)].append(f_idx)
            if b >= 0 and b != a:
                temp[int(b)].append(f_idx)

        total_faces = sum(len(flist) for flist in temp)
        cell_face_offsets = cp.zeros(self.num_cells + 1, dtype=cp.int32)
        cell_face_indices = cp.zeros(total_faces, dtype=cp.int32)

        pos = 0
        for i in range(self.num_cells):
            flist = temp[i]
            l = len(flist)
            cell_face_offsets[i] = pos
            cell_face_indices[pos:pos + l] = flist
            pos += l
        cell_face_offsets[self.num_cells] = pos

        return cell_face_offsets, cell_face_indices

    def compute_cell_centers(self):
        centers = cp.zeros((self.num_cells, 3), dtype=cp.float64)
        for i, cell in enumerate(self.cells):
            pts = self.vertices[cell]
            centers[i] = cp.mean(pts, axis=0)
        return centers

    def compute_face_centers(self):
        valid_faces = self.faces[cp.all(self.faces < len(self.vertices), axis=1)]  # ✅ 確保索引有效
        centers = cp.mean(self.vertices[valid_faces], axis=1)
        return centers

    def compute_face_normals(self):
        normals = cp.zeros((self.num_faces, 3), dtype=cp.float64)
        for i, face in enumerate(self.faces):
            if len(face) < 3:
                normals[i] = cp.array([0, 0, 0])
                continue
            p0, p1, p2 = self.vertices[face[:3]]
            v1 = p1 - p0
            v2 = p2 - p0
            n = cp.cross(v1, v2)
            norm = cp.linalg.norm(n)
            if norm > 1e-12:
                n /= norm
            normals[i] = n
        return normals

    def get_faces_around_cell(self, cell_idx, rings=1):
        visited_cells = set([cell_idx])
        frontier = set([cell_idx])
        for _ in range(rings):
            new_frontier = set()
            for c in frontier:
                start = self.cell_face_offsets[c]
                end = self.cell_face_offsets[c+1]
                faces = self.cell_face_indices[start:end]
                for f in faces:
                    a, b = self.face_to_cells[f]
                    if a != c and a >= 0 and a not in visited_cells:
                        new_frontier.add(a)
                    if b != c and b >= 0 and b not in visited_cells:
                        new_frontier.add(b)
            visited_cells |= new_frontier
            frontier = new_frontier
            if not frontier:
                break

        face_set = set()
        for c in visited_cells:
            start = self.cell_face_offsets[c]
            end = self.cell_face_offsets[c+1]
            for f in self.cell_face_indices[start:end]:
                face_set.add(f)
        return list(face_set)

    def get_adjacent_faces(self, vertex_idx):
        adjacent = [f_idx for f_idx, face in enumerate(self.faces) if vertex_idx in face]
        return adjacent

    def get_cell_edges(self, cell_idx):
        cell_vertices = self.cells[cell_idx]
        edges = [tuple(sorted([cell_vertices[i], cell_vertices[j]])) for i in range(len(cell_vertices)) for j in range(i+1, len(cell_vertices))]
        return edges

    def compute_k_factor(mesh, cell_idx, face_idx):
        cell_type = mesh.cell_types[cell_idx]

        if cell_type in ["hex", "tran"]:  # 六面體 / 過渡單元
            h = mesh.get_cell_size(cell_idx)
            return 0.5 * h  # h/2

        elif cell_type == "tet":  # 四面體單元
            V_j = mesh.cell_volumes[cell_idx]  # Cell 體積
            A_i = mesh.face_areas[face_idx]  # Face 面積
            return 3.0 * V_j / (4.0 * A_i)  # 公式 3V/4A

        else:
            return 1.0  # 預設值，避免例外