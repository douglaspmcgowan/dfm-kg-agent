# 26-Frame Sample Stratification (IDETC §4.2 / Table 15)

Sample size: **26** frames    Corpus: **3879** frames    Threads: **7**    Distinct posts: **7**    Post-score range: **0 to 10**    Source-quote matches: **21/26**

## Distributions

| Stratum | Sample (n=26) | Corpus (n=3879) |
|---|---|---|
| Frame type | observation 12 (46.2%), heuristic 6 (23.1%), principle 4 (15.4%), risk 3 (11.5%), workaround 1 (3.8%) | heuristic 1358 (35.0%), observation 1327 (34.2%), risk 468 (12.1%), principle 410 (10.6%), workaround 154 (4.0%), comparison 147 (3.8%), case 15 (0.4%) |
| Scope | material 5 (19.2%), design_geometry 5 (19.2%), shop_operations 4 (15.4%), cost 3 (11.5%), tooling 3 (11.5%), machining_process 2 (7.7%), quality 2 (7.7%), machine_capability 2 (7.7%) | design_geometry 1177 (30.3%), machining_process 605 (15.6%), machine_capability 533 (13.7%), other 428 (11.0%), quality 252 (6.5%), programming 241 (6.2%), tooling 221 (5.7%), material 165 (4.3%), cost 131 (3.4%), shop_operations 126 (3.2%) |
| Stack Exchange site | engineering 18 (69.2%), electronics 8 (30.8%) | engineering 2452 (63.2%), electronics 723 (18.6%), robotics 342 (8.8%), woodworking 286 (7.4%), mechanics 76 (2.0%) |
| Epistemic stance | direct_neutral 19 (73.1%), tentative_subjective 7 (26.9%) | direct_neutral 2597 (67.0%), tentative_subjective 1028 (26.5%), hedged_emphatic 171 (4.4%), absolute_authoritative 69 (1.8%), narrative_implied 14 (0.4%) |
| Applicability | universal 14 (53.8%), context_dependent 12 (46.2%) | context_dependent 2530 (65.2%), universal 1310 (33.8%), situational 39 (1.0%) |

## Coverage check

- **frame_type**: sample covers 5 of 7 corpus values
  - Not represented in sample: ['case', 'comparison']
- **scope**: sample covers 8 of 10 corpus values
  - Not represented in sample: ['other', 'programming']
- **se_site**: sample covers 2 of 5 corpus values
  - Not represented in sample: ['mechanics', 'robotics', 'woodworking']

## Sample frame list

| # | Thread | Post score | frame_type | scope | site | matched |
|---|---|---|---|---|---|---|
| 1 | Is it possible to have rounded (fillet) edges on a | 10 | observation | cost | electronics | yes |
| 2 | Is it possible to have rounded (fillet) edges on a | 10 | observation | machining_process | electronics | yes |
| 3 | Is it possible to have rounded (fillet) edges on a | 10 | risk | shop_operations | electronics | yes |
| 4 | What's the best material to use for my project | 2 | heuristic | cost | engineering | yes |
| 5 | What's the best material to use for my project | 2 | heuristic | machining_process | engineering | yes |
| 6 | What's the best material to use for my project | 2 | observation | material | engineering | yes |
| 7 | What process is involved in freeform 3D tube bendi | 1 | heuristic | material | engineering | yes |
| 8 | What process is involved in freeform 3D tube bendi | 1 | principle | material | engineering | inferred |
| 9 | What process is involved in freeform 3D tube bendi | 1 | principle | material | engineering | inferred |
| 10 | What process is involved in freeform 3D tube bendi | 1 | observation | material | engineering | inferred |
| 11 | What process is involved in freeform 3D tube bendi | 1 | principle | design_geometry | engineering | inferred |
| 12 | What electrode material can I use for electrical d | 1 | observation | tooling | electronics | yes |
| 13 | What electrode material can I use for electrical d | 1 | observation | tooling | electronics | yes |
| 14 | What electrode material can I use for electrical d | 1 | observation | tooling | electronics | yes |
| 15 | Automating the Coolant Top-Up for CNC Machines | 0 | heuristic | shop_operations | engineering | inferred |
| 16 | Automating the Coolant Top-Up for CNC Machines | 0 | risk | shop_operations | engineering | yes |
| 17 | Automating the Coolant Top-Up for CNC Machines | 0 | observation | quality | engineering | yes |
| 18 | Automating the Coolant Top-Up for CNC Machines | 0 | principle | shop_operations | engineering | yes |
| 19 | Automating the Coolant Top-Up for CNC Machines | 0 | heuristic | quality | engineering | yes |
| 20 | Automating the Coolant Top-Up for CNC Machines | 0 | observation | cost | engineering | yes |
| 21 | USB communication keeps disconnecting because of p | 0 | risk | machine_capability | electronics | yes |
| 22 | USB communication keeps disconnecting because of p | 0 | workaround | machine_capability | electronics | yes |
| 23 | Trouble with Base circle and meshing with spur gea | 0 | heuristic | design_geometry | engineering | yes |
| 24 | Trouble with Base circle and meshing with spur gea | 0 | observation | design_geometry | engineering | yes |
| 25 | Trouble with Base circle and meshing with spur gea | 0 | observation | design_geometry | engineering | yes |
| 26 | Trouble with Base circle and meshing with spur gea | 0 | observation | design_geometry | engineering | yes |

