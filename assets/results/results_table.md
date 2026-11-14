| Config | Episodes | Task success | Objects in target | Placed by skill itself | Pick success / attempt | Avg recoveries | Collision rate | Avg pick-place actions | Avg duration (sim s) | Avg wall time (s) | Self-check agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A  Oracle + scripted | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00 | 2.0% | 4.00 | 54.5 | 4.8 | 100.0% |
| B  Vision + scripted | 50 | 100.0% | 100.0% | 100.0% | 100.0% | 0.00 | 2.0% | 4.00 | 54.6 | 4.7 | 100.0% |
| C  Vision + BC policy | 50 | 98.0% | 99.5% | 38.0% | 95.2% | 2.60 | 6.0% | 4.18 | 89.4 | 8.2 | 96.0% |
| C2 BC, recovery OFF | 50 | 88.0% | 97.0% | 38.5% | 97.0% | 0.00 | 4.0% | 4.00 | 84.5 | 7.3 | 12.0% |
| D  Faults, recovery ON | 50 | 92.0% | 98.0% | 96.5% | 80.1% | 2.28 | 14.0% | 6.22 | 74.8 | 6.4 | 100.0% |
| E  Faults, recovery OFF | 50 | 8.0% | 46.5% | 46.0% | 72.5% | 0.00 | 2.0% | 4.00 | 45.6 | 3.9 | 100.0% |

Per-object placement success:

| Config | milk | bread | cereal | can |
|---|---|---|---|---|
| A  Oracle + scripted | 100% | 100% | 100% | 100% |
| B  Vision + scripted | 100% | 100% | 100% | 100% |
| C  Vision + BC policy | 100% | 98% | 100% | 100% |
| C2 BC, recovery OFF | 100% | 88% | 100% | 100% |
| D  Faults, recovery ON | 100% | 100% | 100% | 92% |
| E  Faults, recovery OFF | 40% | 60% | 48% | 38% |

Seeds 10000..10049 · commit(s) 046a7c8 · robosuite 1.5.2, MuJoCo 3.3.7, torch 2.15.0, Python 3.11.14
