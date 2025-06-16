import numpy as np
import pyvista as pv
from mesh.mesh import HybridMesh
from numba import jit
from collections import defaultdict

def build_faces_and_topology(cells, points):
    """
    從 tetra cells 建立面列表、面法向量與面對應的 cell 索引 (face_to_cells)。
    改進版本：保持面向量方向一致性，提高效率，增強數值穩定性。

    參數:
        cells: List[List[int]]，每個 cell 是4個頂點 index
        points: np.ndarray (N,3)，點座標陣列
    回傳:
        faces: np.ndarray (M,3)，每個 face 是3個頂點 index
        face_normals: np.ndarray (M,3)，面法向量（已正規化，方向一致）
        face_to_cells: np.ndarray (M,2)，每個 face 對應的兩側 cell 索引，若邊界則為 -1
        face_centers: np.ndarray (M,3)，面中心點座標
    """
    if len(cells) == 0:
        return (np.empty((0, 3), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0, 3), dtype=np.float64))

    # 使用更高效的數據結構
    face_dict = defaultdict(list)  # key = canonical face, value = [cell_indices]

    # 預估面數量以預分配空間 (每個tetra有4個面，但內部面會重複)
    estimated_faces = len(cells) * 2  # 經驗估計

    # tetra 的四個面的局部頂點索引 (注意順序，確保外向法向量)
    tetra_faces_local = [
        [0, 2, 1],  # 面 0: 底面 (外向)
        [0, 1, 3],  # 面 1:
        [1, 2, 3],  # 面 2:
        [2, 0, 3],  # 面 3:
    ]

    for cell_idx, cell in enumerate(cells):
        cell = np.array(cell, dtype=np.int32)

        for face_local in tetra_faces_local:
            # 獲取全局頂點索引
            face_vertices = cell[face_local]

            # 創建canonical key (保持方向性)
            canonical_face, is_flipped = get_canonical_face(face_vertices)

            face_dict[canonical_face].append((cell_idx, face_vertices, is_flipped))

    # 構建最終的面列表
    faces_list = []
    face_normals_list = []
    face_to_cells_list = []
    face_centers_list = []

    for canonical_face, cell_data in face_dict.items():
        if len(cell_data) > 2:
            print(f"Warning: Face {canonical_face} shared by {len(cell_data)} cells")
            continue

        # 選擇面的頂點順序 (優先使用非翻轉的)
        face_vertices = None
        face_to_cells = [-1, -1]

        for i, (cell_idx, vertices, is_flipped) in enumerate(cell_data):
            face_to_cells[i] = cell_idx
            if face_vertices is None or not is_flipped:
                face_vertices = vertices

        if face_vertices is None:
            continue

        # 計算面法向量和中心點
        normal, center, is_valid = compute_face_properties(points, face_vertices)

        if not is_valid:
            print(f"Warning: Degenerate face {face_vertices}, skipping")
            continue

        faces_list.append(face_vertices)
        face_normals_list.append(normal)
        face_to_cells_list.append(face_to_cells)
        face_centers_list.append(center)

    # 轉換為numpy陣列
    faces = np.array(faces_list, dtype=np.int32)
    face_normals = np.array(face_normals_list, dtype=np.float64)
    face_to_cells = np.array(face_to_cells_list, dtype=np.int32)
    face_centers = np.array(face_centers_list, dtype=np.float64)

    # 後處理：統一法向量方向
    face_normals = unify_normal_directions(
        faces, face_normals, face_to_cells, points, cells
    )

    return faces, face_normals, face_to_cells, face_centers

def get_canonical_face(face_vertices):
    """
    獲取面的canonical表示，保持方向一致性
    返回: (canonical_face_tuple, is_flipped)
    """
    # 找到最小的頂點索引
    min_idx = np.argmin(face_vertices)

    # 重新排列，使最小索引在第一位
    if min_idx == 0:
        canonical = tuple(face_vertices)
        is_flipped = False
    elif min_idx == 1:
        canonical = (face_vertices[1], face_vertices[2], face_vertices[0])
        is_flipped = False
    else:  # min_idx == 2
        canonical = (face_vertices[2], face_vertices[0], face_vertices[1])
        is_flipped = False

    # 檢查是否需要翻轉以保持一致的方向
    if canonical[1] > canonical[2]:
        canonical = (canonical[0], canonical[2], canonical[1])
        is_flipped = not is_flipped

    return canonical, is_flipped

@jit(nopython=True)
def compute_face_properties(points, face_vertices):
    """
    使用numba加速計算面的法向量和中心點
    """
    p0 = points[face_vertices[0]]
    p1 = points[face_vertices[1]]
    p2 = points[face_vertices[2]]

    # 計算中心點
    center = (p0 + p1 + p2) / 3.0

    # 計算法向量
    vec1 = p1 - p0
    vec2 = p2 - p0
    normal = np.cross(vec1, vec2)

    # 正規化
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-12:  # 提高數值穩定性閾值
        normal = normal / norm_len
        is_valid = True
    else:
        normal = np.array([0.0, 0.0, 0.0])
        is_valid = False

    return normal, center, is_valid

def unify_normal_directions(faces, face_normals, face_to_cells, points, cells):
    """
    統一面法向量方向，使其指向正確的方向
    """
    unified_normals = face_normals.copy()

    for i, (face, normal, cell_pair) in enumerate(zip(faces, face_normals, face_to_cells)):
        if cell_pair[0] == -1:  # 邊界面
            continue

        # 獲取第一個cell的中心
        cell_vertices = cells[cell_pair[0]]
        cell_center = np.mean(points[cell_vertices], axis=0)

        # 計算面中心
        face_center = np.mean(points[face], axis=0)

        # 從cell中心到面中心的向量
        to_face = face_center - cell_center

        # 如果法向量和to_face方向相同，則翻轉法向量（因為法向量應該指向外部）
        if np.dot(normal, to_face) > 0:
            unified_normals[i] = -normal

    return unified_normals

def validate_mesh_connectivity(faces, face_to_cells, n_cells):
    """
    驗證網格連接性
    """
    issues = []

    # 檢查face_to_cells中的索引
    for i, (c1, c2) in enumerate(face_to_cells):
        if c1 >= n_cells or c2 >= n_cells:
            issues.append(f"Face {i}: cell index out of range ({c1}, {c2})")
        if c1 == c2 and c1 != -1:
            issues.append(f"Face {i}: same cell on both sides ({c1})")

    # 統計邊界面數量
    boundary_faces = np.sum(face_to_cells[:, 1] == -1)
    internal_faces = len(faces) - boundary_faces

    print(f"Mesh connectivity validation:")
    print(f"  Total faces: {len(faces)}")
    print(f"  Internal faces: {internal_faces}")
    print(f"  Boundary faces: {boundary_faces}")

    if issues:
        print(f"  Issues found: {len(issues)}")
        for issue in issues[:5]:  # 只顯示前5個問題
            print(f"    {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print("  No connectivity issues found")

    return len(issues) == 0

def load_vtk_to_hybrid_mesh(filename):
    """
    從 .vtk tetrahedralized UnstructuredGrid 讀取 mesh，並建立 HybridMesh。
    改進版本：更好的錯誤處理、性能優化、數據驗證。
    """
    try:
        vtk = pv.read(filename)
    except Exception as e:
        raise ValueError(f"無法讀取 VTK 檔案 {filename}: {e}")

    if not isinstance(vtk, pv.UnstructuredGrid):
        raise ValueError("需要 tetrahedralized UnstructuredGrid .vtk 檔案")

    # 檢查是否包含四面體
    if 10 not in vtk.celltypes:  # VTK_TETRA = 10
        raise ValueError("VTK 檔案中沒有找到四面體單元")

    # 提取四面體
    tet_ids = np.where(vtk.celltypes == 10)[0]
    print(f"找到 {len(tet_ids)} 個四面體單元")

    # 更高效的cell提取
    cells_array = vtk.cells.reshape(-1)
    offsets = vtk.offset

    cell_array = []
    invalid_cells = 0

    for idx in tet_ids:
        start = offsets[idx]
        n_points = cells_array[start]

        if n_points != 4:
            invalid_cells += 1
            continue

        cell = cells_array[start + 1 : start + 1 + n_points]
        cell_array.append(cell.tolist())

    if invalid_cells > 0:
        print(f"警告: 跳過了 {invalid_cells} 個無效的單元")

    if len(cell_array) == 0:
        raise ValueError("沒有找到有效的四面體單元")

    points = np.array(vtk.points, dtype=np.float64)
    cell_types = ["tet"] * len(cell_array)

    print(f"載入了 {len(points)} 個頂點和 {len(cell_array)} 個四面體")

    # 建立完整面與鄰接關係
    print("建立面和拓撲關係...")
    faces, face_normals, face_to_cells, face_centers = build_faces_and_topology(
        cell_array, points
    )

    print(f"建立了 {len(faces)} 個面")

    # 驗證網格連接性
    is_valid = validate_mesh_connectivity(faces, face_to_cells, len(cell_array))
    if not is_valid:
        print("警告: 網格連接性驗證失敗，可能影響模擬結果")

    # 創建 HybridMesh (假設構造函數已更新以接受 face_centers)
    try:
        mesh = HybridMesh(
            points,   # vertices
            faces,
            cell_array,  # cells
            face_to_cells
        )

        # 手動添加 face_centers 屬性
        mesh.face_centers = face_centers
        return mesh
    except Exception as e:
        raise ValueError(f"建立 HybridMesh 失敗: {e}")

def load_vtk_with_validation(filename, check_quality=True):
    """
    載入VTK並進行額外的品質檢查
    """
    mesh = load_vtk_to_hybrid_mesh(filename)

    if check_quality:
        quality_stats = compute_mesh_quality(mesh)
        print("網格品質統計:")
        for key, value in quality_stats.items():
            print(f"  {key}: {value}")

    return mesh

def compute_mesh_quality(mesh):
    """
    計算網格品質指標
    """
    stats = {}

    # 計算面積分佈
    face_areas = []
    for face in mesh.faces:
        p0, p1, p2 = mesh.points[face]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        face_areas.append(area)

    face_areas = np.array(face_areas)
    stats['min_face_area'] = np.min(face_areas)
    stats['max_face_area'] = np.max(face_areas)
    stats['mean_face_area'] = np.mean(face_areas)
    stats['area_ratio'] = np.max(face_areas) / np.min(face_areas) if np.min(face_areas) > 0 else np.inf

    # 計算體積分佈
    cell_volumes = []
    for cell in mesh.cells:
        p0, p1, p2, p3 = mesh.points[cell]
        # 四面體體積 = |det(p1-p0, p2-p0, p3-p0)| / 6
        matrix = np.column_stack([p1-p0, p2-p0, p3-p0])
        volume = np.abs(np.linalg.det(matrix)) / 6.0
        cell_volumes.append(volume)

    cell_volumes = np.array(cell_volumes)
    stats['min_cell_volume'] = np.min(cell_volumes)
    stats['max_cell_volume'] = np.max(cell_volumes)
    stats['mean_cell_volume'] = np.mean(cell_volumes)
    stats['volume_ratio'] = np.max(cell_volumes) / np.min(cell_volumes) if np.min(cell_volumes) > 0 else np.inf

    return stats