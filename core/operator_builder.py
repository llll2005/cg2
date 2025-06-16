import cupy as cp
import cupyx.scipy.sparse as sp

def compute_k_factor(mesh, cell_idx, face_idx):
    cell_type = mesh.cell_types[cell_idx]
    if cell_type in ["hex", "tran"]:
        return 0.5 * mesh.get_cell_size(cell_idx)
    elif cell_type == "tet":
        return 3.0 * mesh.cell_volumes[cell_idx] / (4.0 * mesh.face_areas[face_idx])
    return 1.0

def build_operators(mesh):
    num_cells = len(mesh.cells)
    num_faces = len(mesh.faces)
    rows_D, cols_D, data_D = [], [], [] # 散度
    rows_G, cols_G, data_G = [], [], [] # 壓力梯度
    for i, (a, b) in enumerate(mesh.face_to_cells):
        if a < 0 or a >= num_cells:
            continue
        if b < 0 or b >= num_cells:
            b = a  # 邊界處理

        A = mesh.face_areas[i]  # 面積
        k_a = compute_k_factor(mesh, a, i)
        k_b = compute_k_factor(mesh, b, i)
        k_total = k_a + k_b

        rows_D.append(a)
        cols_D.append(i)
        data_D.append(
            (mesh.face_normals[i] @ mesh.velocity[i]) * A / mesh.cell_volumes[a])

        if b != a:
            rows_D.append(b)
            cols_D.append(i)
            data_D.append(
                (-mesh.face_normals[i] @ mesh.velocity[i]) * A / mesh.cell_volumes[b])

            rows_G.extend([i, i])
            cols_G.extend([a, b])
            data_G.extend([-(mesh.pressure[b] - mesh.pressure[a]) / max(k_total, 1e-12),
               (mesh.pressure[b] - mesh.pressure[a]) / max(k_total, 1e-12)])
        else:
            rows_G.append(i)
            cols_G.append(a)
            data_G.append(1.0 / k_a)

    D = sp.coo_matrix((data_D, (rows_D, cols_D)), shape=(num_cells, num_faces)).tocsr()
    G = sp.coo_matrix((data_G, (rows_G, cols_G)), shape=(num_faces, num_cells)).tocsr()

    return D, G