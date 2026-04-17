# Geometry Optimization Report -- 2026-04-17_heavy_v1

**Wall-clock**: 0h 17m 38s
**Origin**: DE seed 15, polished
**Feasible**: yes

## Objective
| Quantity              | Value         | Target / Limit  |
|-----------------------|--------------:|----------------:|
| Tip displacement      | 0.546020 mm | (minimised) |
| Stiffness K = F/|v|   | 27.471 N/mm | (maximised) |
| Baseline K            | 19.159 N/mm | -- |
| Improvement           | +43.4 % | -- |

## Constraints
| Quantity              | Value         | Limit           | Slack    |
|-----------------------|--------------:|----------------:|---------:|
| Max member |stress|   | 22.139 MPa | 72.000 MPa | +49.861 MPa |
| Max buckling ratio    | 0.7938 | < 1.0000 | +0.2062 |
| Min element length    | 5.0000 mm | >= 5.0000 mm | +0.0000 mm |

## Best design
| Node | x [mm] | y [mm] | Status |
|------|-------:|-------:|--------|
|  1   |   0.000 |   0.000 | frozen |
|  2   |   9.895 |   0.211 | free   |
|  3   |  21.506 |  -1.423 | free   |
|  4   |   0.000 | -10.000 | frozen |
|  5   |   7.681 | -12.350 | free   |
|  6   |  21.013 | -10.104 | free   |
|  7   |   7.313 |  -7.363 | free   |
|  8   |  18.400 |  -5.842 | free   |
|  9   |  25.000 |  -5.000 | frozen |

## Ensemble summary
| Algorithm | Seeds | Best K | Median K | Worst feasible K |
|-----------|------:|-------:|---------:|-----------------:|
| DE        |    16 | 27.472 |   26.889 |              n/a |
| CMA-ES    |     0 |    n/a |      n/a |              n/a |

Active constraints at optimum: stress members [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; buckling members none; length members [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].
