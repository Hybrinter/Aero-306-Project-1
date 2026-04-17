# Geometry Optimization Report -- 2026-04-17_heavy_v2

**Wall-clock**: 0h 18m 02s
**Origin**: CMA-ES seed 0, polished
**Feasible**: yes

## Objective
| Quantity              | Value         | Target / Limit  |
|-----------------------|--------------:|----------------:|
| Tip displacement      | 0.545294 mm | (minimised) |
| Stiffness K = F/|v|   | 27.508 N/mm | (maximised) |
| Baseline K            | 19.159 N/mm | -- |
| Improvement           | +43.6 % | -- |

## Constraints
| Quantity              | Value         | Limit           | Slack    |
|-----------------------|--------------:|----------------:|---------:|
| Max member |stress|   | 25.958 MPa | 72.000 MPa | +46.042 MPa |
| Max buckling ratio    | 1.0000 | < 1.0000 | +0.0000 |
| Min element length    | 5.0000 mm | >= 5.0000 mm | -0.0000 mm |

## Best design
| Node | x [mm] | y [mm] | Status |
|------|-------:|-------:|--------|
|  1   |   0.000 |   0.000 | frozen |
|  2   |   4.816 |  -1.343 | free   |
|  3   |  19.604 |  -0.159 | free   |
|  4   |   0.000 | -10.000 | frozen |
|  5   |  10.732 | -12.236 | free   |
|  6   |  21.788 |  -8.832 | free   |
|  7   |  11.316 |  -7.271 | free   |
|  8   |  17.745 |  -4.801 | free   |
|  9   |  25.000 |  -5.000 | frozen |

## Ensemble summary
| Algorithm | Seeds | Best K | Median K | Worst feasible K |
|-----------|------:|-------:|---------:|-----------------:|
| DE        |    16 | 27.472 |   26.889 |              n/a |
| CMA-ES    |    16 | 27.510 |   26.377 |              n/a |

Active constraints at optimum: stress members [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; buckling members [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; length members [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].
