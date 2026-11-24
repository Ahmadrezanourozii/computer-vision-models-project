# CompVP Project Part 2: Mooney faces in the fly visual system model

**Team member:** Ahmadreza Nourozi (student no. 23726011, M.Sc. Artificial Intelligence), single-person team

## Contents

| Path | Content |
|---|---|
| `Original/` | the 10 original images selected in Part 1 (RGB) |
| `Mooney/` | the 10 corresponding Mooney images (Part 1 tool: blur sigma = 0.7 % of the image side, 10 % white) |
| `Results/SteadyState/` | 20 visualisations: each image as a static video, activity after 2 s (steady state) |
| `Results/RGBFirst/` | 20 visualisations: original for 2 s, then its Mooney |
| `Results/MooneyFirst/` | 20 visualisations: Mooney for 2 s, then its original |
| `Results/Comparisons/` | additional comparison figures and `summary.json` (see below) |
| `Nourozi_23726011_CompVP_Part2_findings.mp4` | 1-minute summary of the findings (H.264 + AAC) |

## Model and input

* flyvis (Lappalainen et al., Nature 634, 2024): connectome-constrained, task-optimised model of the fly
  visual system (45,669 neurons, 65 cell types). Pretrained ensemble `flow/0000`, best model `000`
  (as in the flyvis tutorials); robustness check with the 10 best models `000`-`009`.
* Originals converted to grey-scale (ITU-R BT.601 luminance), Mooneys are binary. Both rendered onto the
  721 hexagonal photoreceptor columns with `BoxEye` (extent 15, 13 px boxes; tutorial 7 defaults: the central
  391 px of the 512 px face crops, so the face fills the eye).
* Static video: the same frame is repeated; integration step 10 ms. Before the first image the network is
  in its steady state under uniform grey (0.5), the flyvis default initial state.

## Timing (chosen from experiments, `Results/Comparisons/comparison_1_timing.png`)

* Every image is shown for **2 s**. After the onset all 65 cell types are within 2 % of their final state by
  1.08 s (median 0.54 s); at 2 s the activity differs from a 10-s presentation by at most 0.08 %.
* The response to an onset or a switch peaks about **120 ms** after it.
* Read-out times:
  * `SteadyState`: 2.0 s after onset (steady state).
  * `RGBFirst` / `MooneyFirst`: every image 120 ms after it appeared, i.e. phase 1 after grey and
    phase 2 after the other image of the same pair (which had been shown for 2 s). This 2 x 2 design compares
    the same image right after grey and right after its partner.
  * 2 s after the switch the activity equals the `SteadyState` activity of the second image
    (difference at most 0.12 %), so the order of presentation only matters during the transient.

## File names

* `SteadyState/face_XXX_original.png`, `SteadyState/face_XXX_mooney.png`
* `RGBFirst/face_XXX_1_original_120ms.png`, `RGBFirst/face_XXX_2_mooney_120ms_after_switch.png`
* `MooneyFirst/face_XXX_1_mooney_120ms.png`, `MooneyFirst/face_XXX_2_original_120ms_after_switch.png`

Each visualisation follows Fig. 1g of the paper (flyvis Fig. 1 notebook, `WholeNetworkFigure`): input image,
its fly-eye rendering, then one hexagonal map per cell type (retina; lamina, medulla intrinsic cells, CT1;
T-shaped and transmedullary cells), black = low and white = high membrane voltage. The grey scale of each cell
type is shared by all 20 `SteadyState` figures, and by all 40 `RGBFirst` + `MooneyFirst` figures, so the
figures can be compared with each other.

## Findings (qualitative; best model 000, robustness in the 10 best models)

1. **To the fly eye a Mooney is almost the photo.** Inside the face the photoreceptor inputs of photo and
   Mooney correlate at r = 0.92 (the same two images at full resolution: 0.85). In the best model the responses
   stay about as similar (steady state: retina 0.92, lamina/medulla 0.92, T-shaped/transmedullary 0.90; most
   different: TmY18 0.76, Tm20 0.80, T4d 0.85). In the other models 0.7-0.9 (lower in model 004, which never settles).
2. **The Mooney drives the network harder.** Its steady-state response patterns are about 1.7 times stronger
   (median over the 65 cell types; 1.6-2.0 in all 10 models), following its higher black/white contrast.
3. **After a switch, ON and OFF cells signal what changed.** 120 ms after one image replaces the other,
   ON cells follow the brightness change (Mi1 +0.98, Tm3 +0.89, Mi4 +0.97, CT1(M10) +0.99) and OFF cells invert it
   (Mi9 -0.99, Tm2 -0.96, Tm9 -1.00); swapping the order flips the sign. In all 10 models Mi1, Tm3 and CT1(M10)
   are ON; the OFF types flip in 2-3 models (compare the Mi4/Mi9 clusters in Fig. 3 of the paper).
4. **Asymmetric masking.** The Mooney exaggerates the contrast of the photo, so the change photo -> Mooney looks
   like the Mooney itself: the Mooney after the photo is represented as after grey (median r = 0.99), while the
   photo after the Mooney is masked (median r = 0.75; below 0.5 in 18 of 65 cell types, e.g. Mi1 0.33, Tm3 0.30).
   Same asymmetry in all 10 models (0.91-0.99 vs. 0.41-0.75).
5. **Memory depends on the model.** In the best model the previous image is forgotten within one second (effect
   below 5 % for every cell type after 0.9 s; 2 s after the switch the activity equals the second image alone,
   difference at most 0.12 %), so it keeps no trace of the photo, unlike human observers, who keep seeing the
   face in a Mooney once they have seen the photo. Over 10-s presentations, models 001 and 005 also forget within
   1-2 s and 002 and 007 within 5-10 s, while in 004, 006, 008 and 009 part of the network keeps a trace for
   more than 10 s (004 does not settle under a static image at all). The optic-flow task (0.4-s training clips)
   does not constrain dynamics this slow. The timing above was chosen for the best model; the slower models would
   need presentations of 5-10 s.

## Comparison figures (`Results/Comparisons/`)

* `comparison_1_timing.png`: settling after onset, size of the onset/switch response, decay of the order effect
* `comparison_2_steady_state_similarity.png`: photo-vs-Mooney similarity and response strength per cell type
* `comparison_3_switch_transient_face_033.png`, `comparison_3_switch_polarity_by_cell_type.png`: ON/OFF
  responses to the switch
* `comparison_5_timecourses_face_033.png`: central-cell responses in all conditions (as in tutorial 7)
* `comparison_4_masking_by_cell_type.png`: masking of the new image right after the switch, per cell type
* `comparison_6_ensemble_robustness.png`: the same measures in the 10 best models, incl. 10-s presentations
* `per_pair/face_XXX.png`: selected cell types of every pair in all conditions
* `summary.json`: all numbers quoted above

Code: `part2_flyvis/` of the project repository (`simulate.py`, `memory_check.py`, `make_figures.py`,
`make_video.py`, `make_submission.py`).

AI assistance: as in Part 1, Claude (Anthropic) was used as a coding and writing assistant for this part
(simulation and figure scripts, video script).
