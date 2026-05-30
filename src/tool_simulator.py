from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
import shutil
import subprocess
import sys

import config


def run_command(script_path: str, sim_dir: str, timeout: int = 3600) -> bool:
    command_name = os.path.basename(script_path)
    log_file = os.path.join(sim_dir, f"log.{command_name}")
    command = f"""
                source {config.solver_env_path} &&
                bash {script_path}
                """

    try:
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                ["bash", "-c", command],
                cwd=sim_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=timeout)
            log.write(stdout)
            log.write(stderr)

        if process.returncode != 0:
            print(f"命令执行失败: {command_name}, 错误信息: {stderr}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        print(f"命令超时 ({timeout}s): {command_name}", file=sys.stderr)
        return False
    except (OSError, subprocess.SubprocessError) as e:
        print(f"执行命令失败: {command_name}， 错误信息: {e}", file=sys.stderr)
        return False


def setup_sim_directory(sim_subdir: str) -> bool:
    """为指定 AOA 创建仿真子目录并拷贝 OpenFOAM 模板与网格。"""
    sim_ref_dir = config.install_dir / "sims" / "sim_ref"
    sim_dir = config.work_dir / "sims" / "simulations" / sim_subdir
    try:
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        shutil.copytree(sim_ref_dir, sim_dir)
        src_msh = config.work_dir / "sims" / "airfoil.msh"
        dst_msh = sim_dir / "airfoil.msh"
        shutil.copy2(src_msh, dst_msh)
        return True
    except (OSError, shutil.Error) as e:
        print(f"仿真配置失败: {e}", file=sys.stderr)
        return False


def _freestream_nutilda(nu: float) -> float:
    """Initial nuTilda for Spalart-Allmaras (≈3× molecular viscosity)."""
    return 3.0 * nu


def simulator(AoA: float, uInf: float, sim_subdir: str) -> bool:
    """
    绕流 RANS CFD（simpleFoam 稳态 SIMPLE + Spalart-Allmaras）。
    """
    if not setup_sim_directory(sim_subdir):
        return False

    sim_dir = config.work_dir / "sims" / "simulations" / sim_subdir
    dict_path = sim_dir / "constant" / "keyParameters"

    try:
        params = ParsedParameterFile(str(dict_path), noHeader=True)
        params["Ufar"] = uInf
        params["alpha"] = AoA
        params["Lref"] = config.chord
        params["Aref"] = config.chord
        nu = float(params["nu"]) if "nu" in params else 1.5e-5
        params["nuTildaInf"] = _freestream_nutilda(nu)
        params["block"] = config.block
        params.writeFile()
    except (OSError, ValueError, KeyError) as e:
        print(f"修改字典文件失败: {e}", file=sys.stderr)
        return False

    allclean_path = sim_dir / "Allclean"
    if config.use_parallel_solver:
        allrun_path = sim_dir / "Allrun-parallel"
    else:
        allrun_path = sim_dir / "Allrun"

    if not run_command(str(allclean_path), str(sim_dir), timeout=300):
        return False
    if not run_command(str(allrun_path), str(sim_dir), timeout=3600):
        return False
    return True


if __name__ == "__main__":
    from utils import default_coeffs
    from tool_mesher import mesher

    coeffs = default_coeffs()
    if not mesher(coeffs):
        sys.exit(1)
    result = simulator(AoA=2.0, uInf=config.uinf, sim_subdir="AoA=2.0")
    if result:
        print("✓ RANS 仿真成功")
    else:
        print("✗ RANS 仿真失败", file=sys.stderr)
        sys.exit(1)
