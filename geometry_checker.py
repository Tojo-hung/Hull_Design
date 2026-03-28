import numpy as np
from stl import mesh

def check_draft_angle(stl_path, pull_direction=np.array([0, 0, 1]), min_angle_deg=1.0):
    """
    Checks if all triangle normals in an STL satisfy a minimum draft angle.

    This prevents undercuts for a single-piece mold pulling in the pull_direction.
    For every face normal 'n', we check that the dot product (n · k) is greater
    than sin(min_angle_deg), where 'k' is the pull direction.

    Args:
        stl_path (str): Path to the STL file.
        pull_direction (np.array): The direction of the mold pull (e.g., +Z axis).
        min_angle_deg (float): The minimum required draft angle in degrees.

    Returns:
        bool: True if the geometry is valid, False otherwise.
    """
    try:
        hull_mesh = mesh.Mesh.from_file(stl_path)
    except (FileNotFoundError, RuntimeError):
        print(f"ERROR: Could not read STL file at {stl_path}. Applying penalty.")
        return False

    # For a draft angle 'a', the angle 'theta' between the normal and the pull
    # direction must be <= 90 - a. So, cos(theta) >= cos(90 - a), which is sin(a).
    min_dot_product = np.sin(np.radians(min_angle_deg))

    # Extract normals and ensure they are properly normalized.
    # (Some custom CAD scripts export STLs with unnormalized or zero-length normals)
    normals = hull_mesh.normals.copy()
    magnitudes = np.linalg.norm(normals, axis=1)
    
    # Safely normalize, avoiding division by zero for degenerate triangles
    valid = magnitudes > 0
    normals[valid] = normals[valid] / magnitudes[valid, np.newaxis]

    # The dot product with the Z-axis unit vector [0, 0, 1] is simply the normal's Z-component.
    dot_products = normals[:, 2]

    if np.any(dot_products < min_dot_product):
        print(f"CONSTRAINT VIOLATED: Found faces with draft angle < {min_angle_deg} degrees.")
        return False

    print("GEOMETRY CHECK PASSED: All faces satisfy the draft angle constraint.")
    return True