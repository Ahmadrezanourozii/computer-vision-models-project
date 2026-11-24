# Ideas for Project Part 3 (collected during Part 2)

Observations from Part 2 that raise follow-up questions:

1. **Light direction (link to Part 1).** Part 1 has 100 faces lit from the left, right, top and bottom. Humans
   (and Qwen3-VL) found bottom-lit Mooneys hardest. Does the fly model treat bottom-lit Mooneys differently, or
   only by their contrast? A light-from-above prior should be absent in a model trained on optic flow.
2. **Moving instead of static Mooneys.** Static images leave the motion detectors (T4/T5) weakly driven. Translating
   or rotating the face (photo vs. Mooney) would engage the trained motion pathway; does the Mooney's hard
   contrast give stronger or wrongly tuned direction-selective responses? The optic-flow decoder could be read out too.
3. **Masking time course.** The photo is masked right after the Mooney (not the other way round). Vary the
   presentation time of the first image and insert grey gaps of different length to map the masking window.
4. **Graded Mooneys.** Morph from photo to Mooney (blend, or the Part-1 tool with increasing blur / changing % white)
   and find where the network's representation departs from the photo; compare with where human ratings drop.
5. **Contrast polarity.** Inverted Mooneys (black <-> white) should swap ON and OFF pathways; test whether the
   asymmetry of finding 3 flips.
6. **Ensemble clusters.** Tutorial 5: models split into clusters (e.g. T4c up- vs. down-tuned). Do the clusters
   respond differently to Mooneys?
7. **Linear read-out.** Train a simple decoder on fly-model activity (face vs. scrambled face, photo vs. Mooney,
   identity) and compare with the Part-1 systems (SCRFD/ArcFace).
8. **Maximally excitatory stimuli** (tutorial 6) for the cell types that separate photo and Mooney most
   (TmY18, Tm20, T4d).
