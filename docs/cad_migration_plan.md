# CAD-Centered Geometry Migration Plan

## 1. Objective

To transition the `Hull_Design` geometric platform from a NumPy-based surface triangulation to a robust, watertight CAD solid model (CadQuery/OCCT), establishing it as the single authoritative source of truth.

This transition solves several structural problems in the current codebase:
- **Geometry Divergence:** Unifies the incompatible 4-point (desktop) and 5-point (web) models behind a shared schema.
- **Topological Fragility:** Replaces mathematical hacks (e.g., NumPy `_bounded_bilge()` fallbacks and `_bow_mesh_starter_section()`) with mathematically rigorous B-spline wires and degenerate-vertex closures.
- **Derivation Accuracy:** Enables exact calculation of displaced volume, prismatic coefficient (Cp), and hydrostatic properties directly from the 3D solid via OCCT Boolean operations, rather than relying on 2D numerical integration approximations.
- **Manufacturability & Interoperability:** Produces standard STEP files intrinsically, ensuring identical geometry is used for CFD (STL tessellated from the solid) and downstream manufacturing.

## 2. Guiding Principles

- **Incremental Migration:** No "flag-day" rewrites. New systems will be built alongside the old, validated, and then swapped. The application must remain runnable throughout the process.
- **CAD as Source of Truth:** Everything (STL, STEP, hydrostatics, DXF) derives from the central OCCT `HullSolid`.
- **HPC/Slurm Compatibility:** CadQuery evaluation runs **only** on the login/pre-processing node. Compute nodes will receive pre-generated STL files, avoiding expensive CAD environment dependencies on the cluster.
- **CFD Pipeline Stability:** The interface to OpenFOAM/cfMesh remains identical (an STL file and parameter metadata). The CFD solver will not "know" the geometry engine changed, except that mesh quality may improve.
- **Open-Source Priority:** Use `cadquery` and `OCP` (OpenCASCADE). Fusion 360 remains an optional downstream consumer of the STEP exports, not the generator.

## 3. Current System Summary

The current pipeline is monolithic and dependent on manual surface mesh generation:

1. **Parameters:** `build_ctrl()` takes a flat dictionary and returns 12 decoupled arrays.
2. **Sections:** `cross_section_spline()` builds 2D arrays of points via PCHIP interpolation. It uses a silent `_bounded_bilge()` fallback if the spline overshoots.
3. **Meshing:** `build_3d_mesh()` evaluates these sections and manually stitches a UV grid of triangles. A fake 0.5mm rectangle is inserted at the bow to avoid zero-area triangles.
4. **CFD Prep:** `build_geometry_stl()` shifts the waterline and exports an ASCII STL.
5. **Optimization:** Constraint checks (rocker, Cp) rely on re-evaluating thousands of points via numerical integration.

**Key Limitations:**
- The bow hack causes mesh issues in cfMesh.
- Constant re-evaluation of geometry for hydrostatics is inefficient.
- Desktop and Web use different copies of the geometry math.
- The existing CadQuery STEP export is broken (creates an open shell, not a solid).

## 4. Target End State (High-Level)

The future architecture separates concerns cleanly:

- **`hull_core.schema`:** Defines the parametric hull constraints explicitly (e.g., `HullForm`, `SectionParams`).
- **`hull_cad`:** A pure geometry engine. It takes a `HullForm`, builds OCCT wires for the stations, lofts a smooth shell, caps the transceiver, and creates a mathematically solid `HullSolid`.
- **`hull_cad` Outputs:** The `HullSolid` serves all downstream needs:
    - `.to_cfmesh_stl()` for the CFD pipeline.
    - `.to_step()` for manufacturing.
    - `.hydrostatics()` for exact displacement and Cp.
- **`hull_cfd`:** Pure OpenFOAM orchestration. It receives an STL, meshes it, runs `interFoam`, and returns drag.

## 5. Migration Strategy Overview

The migration will occur in 6 phases. We will build the CAD proof path first (Phase 1). This is crucial because lofting non-convex geometries (like a hull bilge) is mathematically complex and failure-prone in OCCT. By proving we can reliably loft the hull and validate its volume against the Python implementation *before* integrating it into the optimizer, we isolate the highest technical risk.

Once the CAD engine is robust, we will replace the existing constraint model (Phase 3) and hydrostatics (Phase 4). Finally, we wire the CAD engine into the CFD pipeline (Phase 5) behind a feature flag for A/B testing.

---

## 6. Phase Breakdown

### Phase 1: CAD Proof Path

**Goal:** Establish the foundational geometry layer by building closed CAD wires and lofting them into a watertight OCCT solid.

**Scope:**
- Create `HullForm` and `SectionParams` data models.
- Implement `HullSectionBuilder` to generate `cq.Wire` profiles.
- Implement `HullLoftBuilder` to generate a watertight shell and solid.
- *Explicitly NOT included:* Integration with the optimizer, hydrostatics extraction, or UI updates.

**Implementation Plan:**
- Create `hull_core/schema.py` to define the data structures and a `from_flat_dict` adapter.
- Create `hull_cad/sections.py`. Use OCC `GeomAPI_PointsToBSpline` to fit periodic B-splines through computed points at each station.
- Create `hull_cad/builder.py`. Use a **shell-first** approach: loft the section wires using `BRepOffsetAPI_ThruSections(isSolid=False)`, add a degenerate bow vertex, manually build a planar transom face, sew them together with `BRepBuilderAPI_Sewing`, and cast to a solid.

**Deliverables:**
- `hull_core` and `hull_cad` skeleton modules.
- Isolated test suite (`test_sections.py`, `test_builder.py`) validating the pipeline.

**Success Criteria:**
- `pytest tests/test_builder.py` passes using the default hull parameters.
- The resulting OCCT shape passes `BRepCheck_Analyzer.IsValid()`.
- The solid's total volume (via OCC `GProp`) is reasonable (>100L).

**Risks & Unknowns:**
- **Risk:** OCCT `ThruSections` may create degenerate topological anomalies at the pointed bow.
- **Validation:** BRepCheck validation is strictly enforced in the test suite to catch non-manifold edges early.

### Phase 2: Geometry Validation & Robustness

**Goal:** Ensure the lofting engine survives extreme optimizer variations without crashing or producing inverted solids.

**Scope:**
- Implement edge-case testing for the CAD loft.
- Tessellation logic for STL output.
- *Explicitly NOT included:* Optimizer wiring.

**Implementation Plan:**
- Create a test suite injecting bounded random variations into the `HullForm`.
- Implement `HullSolid.to_stl()` using `BRepMesh_IncrementalMesh` to ensure watertight triangulation at arbitrary resolutions.

**Deliverables:**
- Robust `HullSolid` definition (`hull_cad/solid.py`).
- Automated regression suite checking loft stability across extreme parameter domains.

**Success Criteria:**
- The lofter succeeds on 99%+ of geometrically valid parameter combinations.
- Generated STLs contain no holes and are accepted by cfMesh `surfaceFeatureEdges`.

**Risks & Unknowns:**
- **Risk:** High-curvature stations may cause the spline parameterization to twist the loft.
- **Mitigation:** Enforce identical starting points (e.g., keel center) for every section wire.

### Phase 3: Constraint Refactor

**Goal:** Implement a tiered constraint system to quickly discard bad parameters before invoking the CAD kernel.

**Scope:**
- Replace the monolithic parameter checks with tiered constraints.
- *Explicitly NOT included:* Replacing the `Cp` constraint check entirely (we will still use approximate Python integration for the fast-pruning tier).

**Implementation Plan:**
- Create `hull_core/constraints.py`.
- **Tier 0 (Params):** Instant checks (e.g., all drafts must be >= bow/transom drafts).
- **Tier 1 (Geometric proxy):** Fast Python PCHIP evaluation for rocker and beam reversal.
- **Tier 2 (Solid):** Checks executed only after `HullSolid` is built.

**Deliverables:**
- `HullConstraints` class.
- Unit tests verifying constraint bounds.

**Success Criteria:**
- The tiered constraints successfully replicate the pruning behavior of the current `check_constraints` function but execute in <50ms for Tiers 0/1.

**Risks & Unknowns:**
- **Risk:** Tier 1 approximations might aggressively prune shapes that would be valid in CAD.

### Phase 4: Hydrostatics Migration

**Goal:** Replace the 2D Python integration for displacement and Cp with 3D Boolean operations on the CAD solid.

**Scope:**
- Compute true displacement, waterline, and prismatic coefficient using OCCT.
- *Explicitly NOT included:* Updating the legacy math in `moth_designer`.

**Implementation Plan:**
- Create `hull_cad/hydrostatics.py`.
- Implement `find_waterline_z()` using binary search and `BRepAlgoAPI_Common` cut against a half-space.
- Compute volumetric properties using `BRepGProp`.

**Deliverables:**
- `HullSolid.hydrostatics()` returning a `HydrostaticResult` dataclass.

**Success Criteria:**
- CAD-based displacement matches the target displacement exactly (within tolerance).
- Calculated volume matches the legacy integration method within 1-2%.

**Risks & Unknowns:**
- **Risk:** Boolean operations in OCCT can be slow (~100ms per cut). Since the optimizer uses targeted displacement, binary searching the Z-height requires multiple Boolean cuts per trial.
- **Mitigation:** Start the binary search with a highly accurate initial guess based on the legacy Python integration to minimize iterations.

### Phase 5: CFD Integration Cleanup

**Goal:** Wire the CAD-generated STL into the complete OpenFOAM pipeline alongside the legacy generator.

**Scope:**
- Modify `hull_optimizer.py` to route through `HullSolid`.
- A/B test the CFD results.
- *Explicitly NOT included:* Deleting legacy code.

**Implementation Plan:**
- Create `hull_cad/export.py` to package the hull STL with the domain bounding box.
- Add an `--engine=cad` flag to the optimizer script.
- The optimizer loop evaluates Tiers 0 and 1, builds the CAD solid, evaluates Tier 2, writes the STL, and hands off to the existing `cfMesh` flow.

**Deliverables:**
- Feature-flagged execution path in the optimizer.

**Success Criteria:**
- The optimizer completes a 10-trial run using CAD geometry without crashing.
- Drag predictions correlate with the legacy path.

**Risks & Unknowns:**
- **Risk:** The CAD-tessellated STL will have different node topology than the legacy NumPy grid, which might slightly alter the cfMesh boundary layer structure and, consequently, the drag values.

### Phase 6: Full Architecture Migration

**Goal:** Demolish the legacy code and restructure the repository.

**Scope:**
- Remove old geometry code, update UI apps, lock repository structure.

**Implementation Plan:**
- Delete `hull_web/geometry.py` and rewire `streamlit_app.py` to use `hull_cad`.
- Rewire `moth_designer/app.py` to use `HullSolid.to_stl()` for preview.
- Extract the CFD orchestration from `hull_optimizer.py` into a clean `hull_cfd` package.
- Delete `moth_designer/geometry.py`.

**Deliverables:**
- The final repository structure matching the Target End State.

**Success Criteria:**
- The legacy geometry files are deleted, and all interactive and batch tooling functions flawlessly.

---

## 7. Integration with HPC / Slurm

The CAD capabilities will be heavily centralized to avoid dependency bloat on compute nodes:
- **Head Node / Login Node:** The Optuna optimizer runs here. It performs all geometry generation (CadQuery lofts), constraint checks, and STL tessellation.
- **Compute Nodes:** A Slurm job is dispatched *only after* the `geometry.stl` and `params.json` are written to disk. The compute node requires only OpenFOAM/cfMesh.
- **Performance:** A CadQuery loft + export takes ~3-10s, which is negligible compared to a ~40-minute CFD solve.

## 8. Testing Strategy

We will rely on a rigid test pyramid:
1. **Unit Tests (Fast):** Verify `HullForm` schema validation and Tier 0/1 constraint logic.
2. **CAD Tests (Medium):** Verify `HullSectionBuilder` generates valid `cq.Wire`s. Verify `HullLoftBuilder` generates `BRepCheck`-valid solids.
3. **Regression Tests (Slow):** Calculate displacement for known legacy configurations and assert that the new CAD Boolean method returns results within 2% of the original values.
4. **CFD A/B Validation:** Run the exact same `params.json` through the old and new geometry pipelines, execute OpenFOAM, and analyze the resulting drag correlation.

## 9. Rollback Strategy

The migration is additive. The legacy geometry (`moth_designer/geometry.py`) will not be modified or deleted until Phase 6. The optimizer will use a clean switch (e.g., `--engine=legacy` vs `--engine=cad`). If the CAD engine fails on extreme shapes or produces invalid cfMesh geometries, the optimizer can flip the flag back to the legacy engine to maintain productivity.

## 10. What We Are Deliberately NOT Doing Yet

- **Surrogate Modeling:** Machine learning surrogates are deferred until the CFD pipeline is stable.
- **Advanced Meshing (snappyHexMesh/GMSH):** We are retaining `cfMesh` to minimize variables during the geometry transition.
- **Fully Unified UI:** We are not rewriting the Qt or Streamlit UIs yet, merely routing their geometry backends.
- **Optuna Distributed Execution:** The optimizer will initially remain a local driver submitting Slurm jobs, rather than a distributed PostgreSQL-backed multi-node system.

## 11. First Implementation Steps (Actionable)

The immediate next steps focus exclusively on **Phase 1**:

1. **Step 1:** Create `hull_core/schema.py` and define the `HullForm` and `SectionParams` dataclasses.
2. **Step 2:** Write `tests/test_schema.py` to verify legacy parameter dictionaries can be deserialized into `HullForm` safely.
3. **Step 3:** Create `hull_cad/sections.py` and implement the B-spline generation logic (`HullSectionBuilder`) to create closed OCCT wires.
4. **Step 4:** Write `tests/test_sections.py` to confirm the generated wires pass `BRepCheck`.
5. **Step 5:** Create `hull_cad/builder.py` with the shell-lofter and basic transitive solid conversion.
