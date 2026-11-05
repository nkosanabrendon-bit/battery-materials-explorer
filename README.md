# Battery Materials Explorer

Predicting the average voltage of lithium battery electrode materials from their crystal structure, using a graph neural network (GNN). Data comes from the Materials Project.

## Results (predicting voltage)

Test set: 389 materials whose formulas were never seen in training.

| Model | Test r2 | Test MAE (V) |
|---|---|---|
| Guess the average every time | 0.00 | 0.69 |
| GNN v1 (5 numbers per atom, no distances) | 0.31 | 0.548 |
| Random Forest (which elements are present) | 0.42 | 0.49 |
| GNN v2 (learned atom embeddings + atom distances) | 0.48 | 0.417 |

GNN v2 beat the Random Forest baseline, but the result is modest.

## Repeated on 3 different splits

Same data, same models, 3 different random splits by formula (seeds 1, 2, 3). Script: seeds_test.py.

| Model | Test r2 (mean, std over 3 splits) | Test MAE in V (mean) |
|---|---|---|
| Random Forest | 0.442 (std 0.074) | 0.530 |
| GNN v2 | 0.615 (std 0.099) | 0.424 |

The GNN had a higher r2 and a lower MAE on all 3 splits. Scores changed a lot between splits (GNN r2 from 0.48 to 0.70), so a single split can be misleading.

## How it works
1. get_batteries.py downloads 2,774 lithium insertion electrodes from the Materials Project.
2. clean.py removes rows that are not realistic batteries (voltage 1 to 5.5 V, capacity 10 to 700 mAh/g, energy 50 to 2000 Wh/kg). 2,514 rows are kept.
3. get_structures.py downloads the crystal structure of the discharged (lithium-full) material. 2,512 structures were found.
4. build_graphs.py turns each crystal into a graph: atoms are nodes, and each atom connects to its nearest neighbors (up to 12 within 5 angstroms).
5. baseline_voltage.py trains the Random Forest baseline.
6. train_voltage.py trains GNN v1. train_voltage2.py trains GNN v2 (CGConv layers that use distances).

The data is split by formula (70 percent train, 15 percent validation, 15 percent test), so the same material never appears in two piles.

## Honest caveats
- Only 3 splits were tested. That is better than 1, but still few, so treat the gap as a good sign, not proof.
- The values are computed by DFT, not measured in a lab. Some computed values are unrealistic, which is why the cleaning step exists. The cleaning ranges were chosen by hand.
- Only the discharged structure is used. The charged structure is not used.
- An r2 of 0.48 is not good enough to pick real battery materials.
- Large data files are not in this repo. Run the scripts in the order above to rebuild them.

## How to run it
1. Get a free API key from the Materials Project and store it in the MP_API_KEY environment variable. Never put the key in the code.
2. Make a virtual environment and install: mp-api pandas torch torch_geometric scikit-learn "numpy<2"
3. Run the scripts in the order listed under How it works.
