import os
import subprocess
import shutil
import re

def run_simulation(target_dir="activeCase", stl_path=None, num_cores=1):
    print(f"Preparing simulation environment in: {target_dir}...")
    
    # Step 1: Delete the old test run and copy the pristine template
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree("templateCase", target_dir)

    # Step 1.5: Inject the newly generated STL into the simulation directory
    if stl_path and os.path.exists(stl_path):
        dest_stl = os.path.join(target_dir, "constant", "triSurface", "testHull.stl")
        os.makedirs(os.path.dirname(dest_stl), exist_ok=True)
        shutil.copy2(stl_path, dest_stl)
        print(f"  -> Injected newly generated STL: {stl_path}")
    else:
        print("  -> WARNING: No new STL provided. Using whatever is in templateCase.")

    # Step 1.75: Configure for parallel run if multiple cores are requested
    if num_cores > 1:
        print(f"  -> Configuring for parallel run on {num_cores} cores.")
        dict_path = os.path.join(target_dir, "system", "decomposeParDict")
        with open(dict_path, 'r') as f:
            content = f.read()
        # Dynamically set the number of subdomains to match the core count
        content = re.sub(
            r'numberOfSubdomains\s+\d+;',
            f'numberOfSubdomains  {num_cores};',
            content
        )
        with open(dict_path, 'w') as f:
            f.write(content)

    # Step 2: The exact terminal commands we ran manually
    if num_cores > 1:
        commands = [
            "rm -rf 0 && cp -r 0.orig 0",
            "cartesianMesh",
            "setFields",
            "decomposePar",
            f"mpirun -np {num_cores} interFoam -parallel",
            "reconstructPar -latestTime" # Only stitch the last time step together
        ]
    else: # Fallback to the original single-core workflow
        commands = ["rm -rf 0 && cp -r 0.orig 0", "cartesianMesh", "setFields", "interFoam"]

    # Step 3: Execute them one by one
    for cmd in commands:
        print(f"  -> Running: {cmd}")
        # We pipe stdout to DEVNULL so it doesn't spam your VS Code terminal with math,
        # but we capture stderr so we can see if it crashes.
        process = subprocess.run(
            cmd, 
            shell=True, 
            cwd=target_dir, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        if process.returncode != 0:
            print(f"\nCRITICAL ERROR in {cmd}:")
            print(process.stderr.decode('utf-8'))
            return False
            
    print("Simulation completed successfully!")
    return True

if __name__ == "__main__":
    # A quick test just for this file
    run_simulation("testRun")