import optuna

# TODO: Import your new design script once it's ready
# from custom_designer import generate_hull
from openfoam_runner import run_simulation
from geometry_checker import check_draft_angle
from data_extractor import extract_drag

def objective_function(trial):
    """
    This is the function Optuna will try to minimize.
    It now accepts a 'trial' object, which we use to define the search space.
    """
    # Step 1: Define the search space for all 23 parameters.
    # Optuna uses the 'trial' object to suggest a value for each parameter.
    # This is where you define the name and bounds for every variable.
    params = {
        # Example for 23 parameters. Update names and ranges as needed.
        f"param_{i+1}": trial.suggest_float(f"param_{i+1}", 0.0, 1.0) for i in range(23)
    }
    # The rest of the script can use this 'x' array as before.
    x = list(params.values())

    print(f"\n==========================================")
    print(f"TRIAL #{trial.number} | TESTING NEW HULL ITERATION")
    # Printing a dictionary is more readable than a long list
    print(f"Current Parameters: {params}")
    print(f"==========================================")

    # We test everything in a temporary folder so we don't ruin the template
    test_folder = "activeCase"

    # We save the generated shape to a standalone file so it isn't deleted when the case resets
    latest_stl = "latest_design.stl"

    # Step 1.1: Generate the new shape using the custom Python script
    # generate_hull(x, latest_stl)

    # Step 1.2: Check for manufacturing constraints before running CFD.
    # This saves a huge amount of time by skipping invalid designs immediately.
    if not check_draft_angle(latest_stl, min_angle_deg=1.0):
        print("Manufacturing constraint violated. Skipping meshing and CFD to save time.")
        # This tells Optuna this set of parameters is invalid and should be avoided.
        raise optuna.exceptions.TrialPruned()

    # Step 2: Mesh the new shape and run the physics
    success = run_simulation(test_folder, latest_stl, num_cores=8)

    # Step 3: Extract the drag data and hand it back to Optuna
    if success:
        drag = extract_drag(test_folder)
        if drag is not None:
            return drag

    # Step 4: The Penalty Box
    # If the mesher crashed, the simulation blew up, or drag couldn't be extracted,
    # we prune the trial so the optimizer learns to avoid this shape.
    print("Simulation failed or data extraction error. Pruning trial.")
    raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    print("BOOTING OPTIMIZATION LOOP...")
    
    # Create a study object. We specify the TPE sampler, which is excellent for
    # high-dimensional, expensive "black-box" problems like this one.
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler()
    )
    
    # Start the optimization. 'n_trials' is the total number of CFD simulations you want to run.
    # Bayesian Optimization is far more sample-efficient than Evolutionary Strategies.
    # 100-200 trials will likely yield a better result than 1000+ trials with DE.
    study.optimize(objective_function, n_trials=150)
    
    print("\n******************************************")
    print("OPTIMIZATION COMPLETE")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Lowest Drag Achieved:  {best_trial.value:.2f} N")
    print("Best Parameters Found:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")
    print("******************************************")