from core.density_field import DensityField
from core.velocity_field import VelocityField
from core.poisson_solver import PoissonSolver
from core.advection import SemiLagrangianAdvection
from core.interpolation import VelocityInterpolator
from core.forces import ForceApplicator
from core.operator_builder import build_operators
from export.vdb_writer import write_density_to_vdb
from mesh.loader_vtk_fast import load_vtk_to_hybrid_mesh
from mesh.loader_vtk_fast import compute_mesh_quality

import os
from time import time
from tqdm import tqdm
import cupy as cp

def main():
    # === 🔧 參數設定 ===
    vtk_path = "data/pipe2.vtk"
    total_steps = 30
    dt = cp.float64(0.1)
    output_dir = "vdb_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # === 📥 載入容器網格 ===
    print("Loading mesh...")
    mesh = load_vtk_to_hybrid_mesh(vtk_path)

    compute_mesh_quality(mesh)
    print(f"Loaded mesh with {len(mesh.cells)} cells and {len(mesh.faces)} faces")
    cell_face_offsets, cell_face_indices = mesh.build_cell_to_faces()

    # === 🌊 建立場與運算子 ===
    velocity = VelocityField(len(mesh.faces), mesh.face_normals, mesh.face_to_cells, cell_face_offsets, cell_face_indices)
    velocity.extrapolate_ghost_cells()

    density = DensityField(len(mesh.cells))
    interpolator = VelocityInterpolator(mesh, velocity)
    advection = SemiLagrangianAdvection(mesh, velocity)

    gravity = cp.array([0, 0, -9.8], dtype=cp.float64)
    reference_density = cp.float64(1.0)
    forces = ForceApplicator(gravity=gravity, reference_density=reference_density)

    D, G = build_operators(mesh)
    poisson = PoissonSolver(D, G, mesh, dt=dt, rho=1.0)

    # === ⏳ 初始條件 ===
    print("Setting initial conditions...")
    x_threshold = cp.min(mesh.cell_centers[:, 0]) + 0.2  # ✅ CuPy 運算
    source_cells = cp.where(mesh.cell_centers[:, 0] < x_threshold)[0]  # ✅ CuPy 版索引
    density.add_source(source_cells, 1.0)

    # === 🔄 模擬時間步進 ===
    print("Starting simulation...")
    start_time = time()

    for step in tqdm(range(total_steps), desc="Simulating", unit="step"):
        forces.step(velocity, density, mesh, dt)

        if step % 5 == 0:
            density.add_source(source_cells, 0.5)

        advection.advect_velocity(dt)
        advection.advect_density(density, dt)

        u_star = velocity.get_normal_component()  # ✅ 直接使用 CuPy
        pressure = poisson.solve_pressure(u_star)  # ✅ 保持 GPU 運算
        u_corrected = poisson.correct_velocity(u_star, pressure)
        u_corrected = poisson.apply_boundary_conditions(pressure, u_corrected)
        velocity.set_normal_component(u_corrected)  # ✅ 保持 GPU 記憶體

        velocity.extrapolate_ghost_cells()  # ✅ 確保幽靈單元外插完全在 GPU

        if step % 2 == 0:
            filename = os.path.join(output_dir, f"output_{step:03d}.vdb")
            try:
                write_density_to_vdb(density.get_values(), mesh.cell_centers, filename=filename)  # ✅ 保持 CuPy
            except Exception as e:
                print(f"Failed to write VDB at step {step}: {e}")

        if step % 50 == 0:
            max_density = cp.max(density.get_values()).item()  # ✅ CuPy → Python float
            max_velocity = cp.max(cp.abs(velocity.u_normal)).item()  # ✅ CuPy → Python float
            print(f"Step {step}: Max density = {max_density:.3f}, Max velocity = {max_velocity:.3f}")

    end_time = time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per step: {(end_time - start_time) / total_steps:.3f} seconds")

if __name__ == "__main__":
    main()