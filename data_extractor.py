import os

def extract_drag(case_dir):
    # Locate the OpenFOAM forces file
    forces_file = os.path.join(case_dir, "postProcessing", "forces", "0", "forces.dat")
    
    if not os.path.exists(forces_file):
        print(f"Error: Could not find {forces_file}. Has the simulation started?")
        return None

    # Read all lines from the file
    with open(forces_file, 'r') as f:
        lines = f.readlines()

    # Read from the bottom up to find the very last time step
    last_line = None
    for line in reversed(lines):
        if line.strip() and not line.startswith('#'):
            last_line = line.strip()
            break
            
    if not last_line:
        print("Error: forces.dat is empty or invalid.")
        return None

    # OpenFOAM formats data like this: Time ((px py pz) (vx vy vz)) ((mpx mpy mpz) (mvx mvy mvz))
    # We strip out all the parentheses to turn it into a clean list of numbers
    clean_line = last_line.replace('(', '').replace(')', '')
    columns = clean_line.split()
    
    try:
        # Extract the X-direction forces (Index 1 is Pressure X, Index 4 is Viscous X)
        pressure_drag = float(columns[1])
        viscous_drag = float(columns[4])
        total_drag = pressure_drag + viscous_drag
        
        print(f"Time Step:     {columns[0]}")
        print(f"Pressure Drag: {pressure_drag:.2f} N")
        print(f"Viscous Drag:  {viscous_drag:.2f} N")
        print(f"Total Drag:    {total_drag:.2f} N")
        
        return total_drag
        
    except IndexError:
        print("Error: Unexpected data format in forces.dat")
        return None

if __name__ == "__main__":
    # When we run this file directly, test it on our template folder
    print("Testing OpenFOAM Data Extractor...")
    target_directory = "templateCase"
    final_drag = extract_drag(target_directory)
    
    if final_drag:
        print(f"\nSUCCESS: The optimizer will receive a final drag value of: {final_drag:.2f}")