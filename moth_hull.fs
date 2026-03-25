FeatureScript 2126;
import(path : "onshape/std/geometry.fs", version : "2126.0");

/**
 * Moth Hull — split into 3 separate features so each can be tested independently:
 *
 *  1. Hull Section Plane  — creates ONE construction plane at a given x position
 *  2. Hull Section Sketch — draws ONE cross-section profile on a selected plane
 *  3. Hull Loft           — lofts through selected sketch profiles
 *
 * Test order: add feature 1, then 2 referencing it, then repeat for all stations,
 * then add feature 3 selecting all sketches.
 */

const N_BILGE = 32;

// ─── Shared: build closed polyline for one cross-section ─────────────
function sectionPolyline(hb is number,      depth   is number,
                         keel_w is number,  hb_z    is number,
                         deck_hb is number, deck_z  is number) returns array
{
    var pts = [];

    // Start at centreline top
    pts = append(pts, vector(0 * millimeter, deck_z * millimeter));

    // Deck line outward
    pts = append(pts, vector(deck_hb * millimeter, deck_z * millimeter));

    // Topsides down to max beam
    pts = append(pts, vector(hb * millimeter, hb_z * millimeter));

    // Quarter-ellipse bilge (max beam → keel shoulder)
    var ell_hb = hb - keel_w;
    for (var i = N_BILGE; i >= 0; i -= 1)
    {
        var t = (i / N_BILGE) * (PI / 2);
        var y = keel_w + ell_hb * sin(t);
        var z = hb_z + (-depth - hb_z) * cos(t);
        pts = append(pts, vector(y * millimeter, z * millimeter));
    }

    // Flat keel inward to centreline
    pts = append(pts, vector(0 * millimeter, -depth * millimeter));

    // Centreline back up to close
    pts = append(pts, vector(0 * millimeter, deck_z * millimeter));

    return pts;
}


// ════════════════════════════════════════════════════════════════════
// FEATURE 1 — Hull Section Plane
// Creates a single construction plane perpendicular to X at position x.
// Add one of these for each of the 7 stations.
// ════════════════════════════════════════════════════════════════════
annotation { "Feature Type Name" : "Hull Section Plane" }
export const hullSectionPlane = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "X position (mm)" }
        isLength(definition.xPos, NONNEGATIVE_LENGTH_BOUNDS);
    }
    {
        opPlane(context, id + "plane", {
            "plane" : plane(
                vector(definition.xPos, 0 * millimeter, 0 * millimeter),
                vector(1, 0, 0)
            )
        });
    });


// ════════════════════════════════════════════════════════════════════
// FEATURE 2 — Hull Section Sketch
// Draws one closed cross-section profile on a selected plane.
// Select the plane created by Feature 1 as the sketch plane.
// ════════════════════════════════════════════════════════════════════
annotation { "Feature Type Name" : "Hull Section Sketch" }
export const hullSectionSketch = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Sketch plane" }
        definition.sketchPlane is Query;

        annotation { "Name" : "Half beam (mm)" }
        isLength(definition.hb, NONNEGATIVE_LENGTH_BOUNDS);

        annotation { "Name" : "Draft (mm)" }
        isLength(definition.depth, NONNEGATIVE_LENGTH_BOUNDS);

        annotation { "Name" : "Keel width (mm)" }
        isLength(definition.keelW, NONNEGATIVE_LENGTH_BOUNDS);

        annotation { "Name" : "Beam height hz (mm)" }
        isLength(definition.hz, NONNEGATIVE_LENGTH_BOUNDS);

        annotation { "Name" : "Deck half beam (mm)" }
        isLength(definition.deckHb, NONNEGATIVE_LENGTH_BOUNDS);

        annotation { "Name" : "Deck height (mm)" }
        isLength(definition.deckZ, NONNEGATIVE_LENGTH_BOUNDS);
    }
    {
        var hb      = definition.hb     / millimeter;
        var depth   = definition.depth  / millimeter;
        var keelW   = definition.keelW  / millimeter;
        var hz      = definition.hz     / millimeter;
        var deckHb  = definition.deckHb / millimeter;
        var deckZ   = definition.deckZ  / millimeter;

        var sk = newSketch(context, id + "sk", {
            "sketchPlane" : definition.sketchPlane
        });

        skPolyline(sk, "profile", {
            "points" : sectionPolyline(hb, depth, keelW, hz, deckHb, deckZ)
        });

        skSolve(sk);
    });


// ════════════════════════════════════════════════════════════════════
// FEATURE 3 — Hull Loft
// Select all 7 section sketch regions to loft through.
// ════════════════════════════════════════════════════════════════════
annotation { "Feature Type Name" : "Hull Loft" }
export const hullLoft = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Section profiles", "Filter" : EntityType.FACE }
        definition.profiles is Query;
    }
    {
        opLoft(context, id + "loft", {
            "profileSubqueries" : evaluateQuery(context, definition.profiles),
            "bodyType"          : ToolBodyType.SOLID
        });
    });
