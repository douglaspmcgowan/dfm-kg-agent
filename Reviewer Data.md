# Reviewer 1 \-- Knowledge Frame Evaluation

## Background

This study evaluates an AI-powered knowledge extraction pipeline. The pipeline reads Q&A posts from engineering Stack Exchange sites (topics include CNC machining, materials, tooling, and shop practices) and uses a large language model to extract structured **knowledge frames** from each post.

A **knowledge frame** is one discrete, self-contained piece of knowledge extracted from a forum post. Each frame is distinct and is independent from every other frame. A frame captures a single claim, rule, observation, or practice described by the post's author. The extraction system reads the original post text, identifies passages that contain transferable knowledge, and produces a structured record for each one.

## What you are reviewing

Below you will find a set of extracted frames, grouped by the source post they came from. For each frame you will see:

- **Source Quote** (in a blockquote) \-- the verbatim text from the original forum post that the system identified as supporting this frame.\
- **Main Point** \-- a 2-3 sentence summary the system wrote to capture the transferable knowledge claim from the quote.\
- **Classified metadata** \-- labels the system assigned:
  - **Frame Type** \-- what kind of knowledge this is (see reference table below)\
  - **Post Role** \-- how the source post functions in the discussion (e.g., a direct answer, supplementary info, a narrative, etc.)\
  - **Scope** \-- the engineering domain the frame belongs to (e.g., machining_process, material, tooling, cost)\
  - **Applicability** \-- how broadly the claim holds: *universal* (true in virtually all cases), *context_dependent* (true under specific conditions), or *situational* (specific to a particular shop/machine)\
  - **Epistemic Stance** \-- the tone/certainty of the original quote: ranging from *absolute_authoritative* (strong, no hedges) to *narrative_implied* (inferred from a story rather than stated directly)

Each post group also shows contextual metadata from the source forum:

- **Thread** \-- the title of the Q&A thread the post appeared in.\
- **Post Role** \-- what role the post played in the thread.\
- **Post Score** \-- the community vote score on the original post (upvotes minus downvotes; higher = more community agreement).\
- **Tags** \-- topic tags assigned to the thread by the original poster.

## Frame type reference

| Frame Type | Meaning |
|:---|:---|
| risk | A condition that leads to a failure mode or operational limitation |
| heuristic | A personal practice, preference, or directed advice with rationale |
| principle | A causal explanation of WHY something happens |
| observation | An empirical, factual, or definitional claim (no action, no causal mechanism) |
| comparison | Explicitly weighs two or more options against each other |
| workaround | A non-standard fix done because of a stated limitation |
| case | A specific real-world job or incident narrative |
| other | Does not fit the above (should include an extraction rationale) |

## Rating instructions

For each frame, fill in every field in the response block below it. Type your rating directly after the colon on each line.

**Quality ratings (1-3 scale):**

| Rating | Extraction Accuracy | Tribal Knowledge | Clarity |
|:---|:---|:---|:---|
| 1 | Quote does not support the main point, or main point misrepresents the quote | Generic textbook knowledge \-- easily found in any reference | Poorly written, ambiguous, or hard to understand |
| 2 | Main point mostly captures the quote, but has minor omissions or overstatements | Moderately specialized \-- a practitioner might know this, but it is not trivial | Understandable but could be clearer or more precise |
| 3 | Main point accurately and completely captures the knowledge in the quote | Genuine tribal/experiential knowledge \-- unlikely to be found in textbooks | Clear, precise, and immediately actionable |

**Classification ratings (1-5 scale):**

For Frame Type, Post Role, Scope, Applicability, Epistemic Stance, and Main Point Quality, rate how appropriate the system's label is:

| Rating | Meaning                                            |
|:-------|:---------------------------------------------------|
| 1      | Completely wrong                                   |
| 2      | Poor \-- a different label would clearly be better |
| 3      | Acceptable \-- reasonable but not ideal            |
| 4      | Good \-- correct and appropriate                   |
| 5      | Excellent \-- could not be improved                |

**Overall judgment:**

- **Keep** \-- the frame is accurate, useful, and correctly classified\
- **Revise** \-- the frame captures real knowledge but needs corrections to labels or main point\
- **Remove** \-- the frame is wrong, trivial, or does not represent meaningful extracted knowledge

Use the **Comments** field for anything you want to note: suggested label corrections, concerns about the quote, or general observations.

**Total frames to review:** 26

------------------------------------------------------------------------

## Post Group 1

**Thread:** Is it possible to have rounded (fillet) edges on a PCB? **Post Role:** solution **Post Score:** 10 **Tags:** pcb, pcb-fabrication

### Frame 1

**Source Quote:**

It might not be perfectly rounded but a "beveled edge" should be common for most manufacturers and should not drive the cost up much.

**Main Point (system-generated summary):** Many PCB fabricators can provide a beveled board edge that approximates a rounded feel, and this feature typically adds little cost.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** PCB edge beveling availability and cost impact\
- **Example Context:** PCB fabrication houses offering edge beveling as an option

---REVIEW FRAME 1---

Extraction Accuracy (1-3): **2**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **4**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Overall makes an accurate enough assessment, only being slightly inaccurate in Frame Type, which points more to workaround**

------------------------------------------------------------------------

### 

### Frame 2

**Source Quote:**

A board separated by V-cut will also be quite rounded, but they don't look very nice, so certainty not the "premium" feel there.

**Main Point (system-generated summary):** V-cut panelization leaves PCB edges somewhat rounded after separation, but the resulting finish is not aesthetically premium.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** V-cut (V-score) panel separation edge aesthetics\
- **Example Context:** PCBs depanelized by V-cut/V-score

---REVIEW FRAME 2---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **1**

Post Role (1-5): **2**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **1**

Overall (Keep / Revise / Remove): **Revise**

Comments: **Post is not an observation, should be heuristic. Post role offers no solution to stated issue. Epistemic stance is not neutral**

------------------------------------------------------------------------

### 

### Frame 3

**Source Quote:**

And as others have said, if you are willing to put money on the table the manufacturer can do just about anything, but they will not even discuss it with you unless you have a profitable relationship with them.

**Main Point (system-generated summary):** Custom or unusual PCB edge features may be possible if you pay for them, but without a profitable ongoing relationship the manufacturer may refuse to discuss such requests at all.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** Manufacturer engagement for custom PCB features depends on profitability/relationship\
- **Example Context:** PCB manufacturers when asked for non-standard edge rounding or specialty finishing

---REVIEW FRAME 3---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **3**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Frame type may be heuristic rather than risk. Post role is partially a solution**

------------------------------------------------------------------------

## Post Group 2

**Thread:** What's the best material to use for my project **Post Role:** solution **Post Score:** 2 **Tags:** manufacturing-engineering, metals

### Frame 4

**Source Quote:**

If you wanted to do 50,000 units, the answer is almost certainly injection molding of some kind. You could make what you are describing easily for probably pennies per piece... but it would cost quite a bit to set up the mold. That's probably not worth the setup cost just for 500 pieces.

**Main Point (system-generated summary):** Injection molding delivers very low per-part cost at high volumes (on the order of tens of thousands), but the mold's upfront cost makes it uneconomical for small runs like \~500 units. Choose injection molding only when quantity is high enough to amortize the tooling cost.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** injection molding setup economics\
- **Example Value:** At \~50,000 units, parts can be made for pennies each; at \~500 units, the mold setup cost is not worth it.\
- **Example Context:** Process selection for plastic parts where injection molding is an option, comparing small (\~500) vs large (\~50,000) production volumes.

---REVIEW FRAME 4---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good**

------------------------------------------------------------------------

### Frame 5

**Source Quote:**

Buy some big sheets and have a machine shop slice them into 3" sections with a band saw.

**Main Point (system-generated summary):** For a few hundred simple flat parts, buy large plastic sheets and have a shop bandsaw them into 3-inch-wide blanks to reduce fabrication cost before drilling.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** bandsaw slicing of plastic sheet into blanks\
- **Example Value:** Cut 12"-long by 3"-wide strips from sheet as blanks for subsequent hole drilling.\
- **Example Context:** Low-volume (hundreds) production of simple plastic mounting plates from sheet stock.

---REVIEW FRAME 5---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **5**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, just some awkward wording in Main Point**

------------------------------------------------------------------------

### Frame 6

**Source Quote:**

Looks like you could get the raw material in acrylic for less than \$1 per piece. That should be easily enough to drill holes in and should be stiff enough.

**Main Point (system-generated summary):** Acrylic sheet is a very low-cost option (under about \$1 per piece for this application) that is stiff enough and easy to drill for simple mounting plates.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** acrylic sheet suitability and cost\
- **Example Value:** Acrylic blanks for the described plate can be sourced for under \$1 each.\
- **Example Context:** Simple 12" x 3" thin mounting plates requiring several drilled holes for PCB mounting hardware.

---REVIEW FRAME 6---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **5**

Scope (1-5): **4**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good**

------------------------------------------------------------------------

## Post Group 3

**Thread:** What process is involved in freeform 3D tube bending? **Post Role:** tutorial **Post Score:** 1 **Tags:** machine-design, cnc, bending, machine-elements

### Frame 7

**Source Quote:**

This is based on many simplifications, in reality it may be better to determine springback using tests.

**Main Point (system-generated summary):** Because analytical springback models rely on simplifying assumptions, a practical approach is to measure springback empirically with tests rather than relying solely on calculations.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** springback determination via testing

---REVIEW FRAME 7---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **2**

Frame Type (1-5): **5**

Post Role (1-5): **3**

Scope (1-5): **2**

Applicability (1-5): **3**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **3**

Overall (Keep / Revise / Remove): **Revise**

Comments: **Main Point appears correct, but post's purpose (scope, post role, applicability, etc.) seem off**

------------------------------------------------------------------------

### Frame 8

**Source Quote:**

In pure bending, axial strains have a linear distribution through section depending on coordinate $z$ and bend radius $R$ :

$$\\epsilon(z) \= \\frac{z}{R}$$

**Main Point (system-generated summary):** Under pure bending, axial strain varies linearly with distance from the neutral axis, equal to z divided by the bend radius R.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** axial strain distribution in pure bending

---REVIEW FRAME 8---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, textbook response**

------------------------------------------------------------------------

### Frame 9

**Source Quote:**

For bends with relatively small radii, we can assume, that the bending stress at applied loads is fully plastic at yield stress $R\_e$ :

**Main Point (system-generated summary):** When bend radii are small in elastoplastic materials, the section under load can be treated as fully plastic, with stress at approximately the yield stress across the thickness.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** fully plastic stress at small bend radii\
- **Example Context:** Small-radius bending of elastoplastic materials (e.g., tube bending where the section yields across most of its thickness).

---REVIEW FRAME 9---

Extraction Accuracy (1-3): **2**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **2**

Frame Type (1-5): **5**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **5**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, adds some additional info in main point**

------------------------------------------------------------------------

### Frame 10

**Source Quote:**

From this, you can integrate and end up with expression for $k$ (valid for thin-walled tubes):

$$k \= \\frac{4R\_e}{\\pi r E}$$

**Main Point (system-generated summary):** For thin-walled tube bending, the springback parameter k equals 4·Re divided by (π·r·E), relating yield stress, mean radius, and Young's modulus.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** springback curvature constant for thin-walled tubes\
- **Example Context:** Thin-walled tube bending with tube mean radius r, wall thinness assumption, yield stress Re, and modulus E.

---REVIEW FRAME 10---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, seems textbook**

------------------------------------------------------------------------

### Frame 11

**Source Quote:**

This constant represents difference between curvature under load and resulting target curvature:

$$k \= \\frac{1}{R\_{load}}-\\frac{1}{R\_{target}}$$

**Main Point (system-generated summary):** In springback modeling, k quantifies how much you must over-bend: it equals the difference between the curvature under load and the final (target) curvature, i.e., 1/R_load minus 1/R_target.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** design_geometry\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** curvature difference between loaded and target bend\
- **Example Context:** Tube bending springback calculations where k has been defined from material/geometry properties.

---REVIEW FRAME 11---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **2**

Frame Type (1-5): **5**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, seems textbook**

------------------------------------------------------------------------

## Post Group 4

**Thread:** What electrode material can I use for electrical discharge machining? **Post Role:** solution **Post Score:** 1 **Tags:** discharge, electric-machine, spark, electrode, die

### Frame 12

**Source Quote:**

The deciding factors for choice of electrode material for an electrical discharge machine (EDM - also known as a spark erosion machine) would be its conductivity and erosion resistance.

**Main Point (system-generated summary):** Selecting an EDM electrode material should be based primarily on its electrical conductivity and its resistance to erosion during sparking. These two properties drive electrode performance and life.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** EDM electrode material selection criteria

---REVIEW FRAME 12---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, neutral, almost textbook, but not trivial**

------------------------------------------------------------------------

### Frame 13

**Source Quote:**

Copper and graphite are the most preferred materials.

**Main Point (system-generated summary):** Copper and graphite are commonly preferred choices for EDM electrode materials due to their favorable properties for the process.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** preferred EDM electrode materials

---REVIEW FRAME 13---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **4**

Scope (1-5): **4**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, seems more heuristic than observation (says preferred, uses no criteria)**

------------------------------------------------------------------------

### Frame 14

**Source Quote:**

One wouldn't come across lead or solder for this application and their wear resistance wouldn't be known unless trials are carried out.

**Main Point (system-generated summary):** Lead or solder are not typically used as EDM electrode materials, and their wear resistance is uncertain without performing experimental trials.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** lead or solder as EDM electrode materials

---REVIEW FRAME 14---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **3**

Scope (1-5): **4**

Applicability (1-5): **4**

Main Point Quality (1-5): **4**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Revise**

Comments: **Appears more opinionated and context dependent than model predicts**

------------------------------------------------------------------------

## Post Group 5

**Thread:** Automating the Coolant Top-Up for CNC Machines **Post Role:** solution **Post Score:** 0 **Tags:** machining, cnc

### Frame 15

**Source Quote:**

While it's possible to have a tank of coolant concentrate and mix inline if you have so much work to do that automation is practical and manual checking is impractical then you should consider a coolant recycling system .

**Main Point (system-generated summary):** If the workload justifies automation and manual checking is impractical, choose a coolant recycling system rather than relying on inline mixing to manage coolant concentration.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** adopting a coolant recycling system instead of inline mixing

---REVIEW FRAME 15---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **3**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good advice, applicability depends more on context than universal**

------------------------------------------------------------------------

### Frame 16

**Source Quote:**

As you have noticed, using inline mixing is still going to require checking the machines; either manually or automatically.

**Main Point (system-generated summary):** Inline mixing does not eliminate oversight; machines still need to be checked for coolant condition, either by manual inspection or automated monitoring.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** inline mixing monitoring requirement

---REVIEW FRAME 16---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good**

------------------------------------------------------------------------

### Frame 17

**Source Quote:**

What you end up with is coolant with slowly deteriorating properties until you flush out periodically and occasionally have entirely fresh and properly mixed coolant.

**Main Point (system-generated summary):** Relying on periodic top-ups leads to coolant properties that gradually degrade over time, punctuated by periodic flushes that briefly restore proper mix.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** coolant property drift without recycling

---REVIEW FRAME 17---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **2**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Revise**

Comments: **Summary is correct, post isn't a solution**

------------------------------------------------------------------------

### Frame 18

**Source Quote:**

With a recycling system you can remove mechanically dispersed fines and tramp oils which wear and gum up the equipment.

**Main Point (system-generated summary):** Coolant recycling systems remove fines and tramp oils, contaminants that otherwise contribute to equipment wear and gumming.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** removal of fines and tramp oils via recycling

---REVIEW FRAME 18---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **5**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good**

------------------------------------------------------------------------

### Frame 19

**Source Quote:**

Using a refractometer the concentration of purified and properly mixed coolant can be maintained all the time,

**Main Point (system-generated summary):** Use a refractometer in a recycling setup to continuously maintain correct coolant concentration rather than allowing it to drift.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** refractometer-based coolant concentration control

---REVIEW FRAME 19---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **5**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good summary and evaluation**

------------------------------------------------------------------------

### 

### Frame 20

**Source Quote:**

Payback on equipment purchase is typically 6-9 months

**Main Point (system-generated summary):** Vendors of centralized coolant recycling systems report typical payback periods on equipment of around 6--9 months due to reduced disposal and new coolant costs.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** coolant recycling system payback period (vendor claim)\
- **Example Value:** Quoted claim for the PRAB Guardian Central Coolant Recycling System.\
- **Example Context:** PRAB Guardian Central Coolant Recycling System (vendor claim)

---REVIEW FRAME 20---

Extraction Accuracy (1-3): **2**

Tribal Knowledge (1-3): **2**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **4**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, frames claim well**

------------------------------------------------------------------------

## Post Group 6

**Thread:** USB communication keeps disconnecting because of power dip/spike **Post Role:** solution **Post Score:** 0 **Tags:** power-supply, capacitor, usb, usb-device, ups

### Frame 21

**Source Quote:**

I had similar issue, the desktop built-in USB port connection was failing when turning the spindle off (a Dremel via opto-isolated relay).

**Main Point (system-generated summary):** Motherboard USB ports can drop the CNC connection when the spindle is switched off, indicating susceptibility to disturbances during spindle power switching. This manifests as a USB link failure at the moment of turn-off.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** motherboard USB port susceptibility during spindle power switching\
- **Example Value:** On a desktop PC controlling a CNC, the built-in USB port would disconnect when a Dremel spindle (switched via an opto-isolated relay) was turned off.\
- **Example Context:** PC-based CNC controller connected over USB; spindle (Dremel) controlled via opto-isolated relay; disturbance occurs at spindle turn-off.

---REVIEW FRAME 21---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **4**

Post Role (1-5): **2**

Scope (1-5): **5**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Revise**

Comments: **Good framing, identifies risk not solution**

------------------------------------------------------------------------

### Frame 22

**Source Quote:**

I happened to have a PCI USB-3 card on the desktop, switched to one of those ports, and it seems to work, so far no disconnects.

**Main Point (system-generated summary):** Switching the CNC's USB connection from a motherboard port to a PCIe USB 3.0 add-in card can stop the disconnects, suggesting those ports are more tolerant of the disturbance from spindle power switching.

**System-assigned labels:**

- **Frame Type:** workaround\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** using PCIe USB 3.0 add-in card to improve USB connection robustness\
- **Example Value:** After repeated disconnects on the built-in USB, moving the cable to a PCI USB-3 card resulted in no disconnects so far.\
- **Example Context:** Desktop PC controlling a CNC over USB, experiencing disconnects when a Dremel spindle (via opto-isolated relay) is turned off; alternative USB provided by a PCIe USB 3.0 card.

---REVIEW FRAME 22---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **3**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **5**

Scope (1-5): **5**

Applicability (1-5): **4**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good all around**

------------------------------------------------------------------------

## Post Group 7

**Thread:** Trouble with Base circle and meshing with spur gears **Post Role:** solution **Post Score:** 0 **Tags:** mechanical-engineering, gears, meshing

### Frame 23

**Source Quote:**

For a pair of meshing gears the teeth on both need to be the same size ie you don't just scale a standard gear by some factor to get different diameters.

**Main Point (system-generated summary):** Meshing gears must have the same tooth size (module); you cannot obtain a compatible mate by simply scaling a standard gear to a different diameter. Change the tooth count while keeping the module consistent instead.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear tooth size compatibility (module consistency)

---REVIEW FRAME 23---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **3**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Simple knowledge, not really heuristic, but close**

------------------------------------------------------------------------

### Frame 24

**Source Quote:**

So as you have noticed, on a larger gear the pitch circle diameter and root diameter will be proportionally closer to each other than on a smaller gear.

**Main Point (system-generated summary):** As gear size increases, the pitch circle diameter and root diameter become proportionally closer to each other compared to smaller gears.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** relative spacing of pitch circle and root diameter vs gear size

---REVIEW FRAME 24---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **3**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, textbook**

------------------------------------------------------------------------

### Frame 25

**Source Quote:**

This is what the Module number is for, its a ratio between the diameter and number of teeth.

**Main Point (system-generated summary):** The module defines tooth size by specifying the ratio of gear diameter to the number of teeth, ensuring compatible meshing when the module is the same.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear module definition

---REVIEW FRAME 25---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, basic textbook**

------------------------------------------------------------------------

### Frame 26

**Source Quote:**

These are, conventionally the difference between the pitch circle diameter and the tip and root respectively ie addendum plus dedendum = the height of the tooth above the root circle.

**Main Point (system-generated summary):** By convention, addendum is the radial distance from the pitch circle to the tooth tip and dedendum from the pitch circle to the root; their sum equals the tooth height above the root circle.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** addendum and dedendum definitions and relationship

---REVIEW FRAME 26---

Extraction Accuracy (1-3): **3**

Tribal Knowledge (1-3): **1**

Clarity (1-3): **3**

Frame Type (1-5): **5**

Post Role (1-5): **4**

Scope (1-5): **5**

Applicability (1-5): **5**

Main Point Quality (1-5): **5**

Epistemic Stance (1-5): **5**

Overall (Keep / Revise / Remove): **Keep**

Comments: **Good, textbook**

------------------------------------------------------------------------

# Reviewer 2 \-- Knowledge Frame Evaluation

## Background

This study evaluates an AI-powered knowledge extraction pipeline. The pipeline reads Q&A posts from engineering Stack Exchange sites (topics include CNC machining, materials, tooling, and shop practices) and uses a large language model to extract structured **knowledge frames** from each post.

A **knowledge frame** is one discrete, self-contained piece of knowledge extracted from a forum post. Each frame is distinct and is independent from every other frame. A frame captures a single claim, rule, observation, or practice described by the post's author. The extraction system reads the original post text, identifies passages that contain transferable knowledge, and produces a structured record for each one.

## What you are reviewing

Below you will find a set of extracted frames, grouped by the source post they came from. For each frame you will see:

- **Source Quote** (in a blockquote) \-- the verbatim text from the original forum post that the system identified as supporting this frame.\
- **Main Point** \-- a 2-3 sentence summary the system wrote to capture the transferable knowledge claim from the quote.\
- **Classified metadata** \-- labels the system assigned:
  - **Frame Type** \-- what kind of knowledge this is (see reference table below)\
  - **Post Role** \-- how the source **post** functions in the discussion (e.g., a direct answer, supplementary info, a narrative, etc.)\
  - **Scope** \-- the engineering domain the frame belongs to (e.g., machining_process, material, tooling, cost)\
  - **Applicability** \-- how broadly the claim holds: *universal* (true in virtually all cases), *context_dependent* (true under specific conditions), or *situational* (specific to a particular shop/machine)\
  - **Epistemic Stance** \-- the tone/certainty of the original quote: ranging from *absolute_authoritative* (strong, no hedges) to *narrative_implied* (inferred from a story rather than stated directly)

Each post group also shows contextual metadata from the source forum:

- **Thread** \-- the title of the Q&A thread the post appeared in.\
- **Post Role** \-- what role the post played in the thread.\
- **Post Score** \-- the community vote score on the original post (upvotes minus downvotes; higher = more community agreement).\
- **Tags** \-- topic tags assigned to the thread by the original poster.

## Frame type reference

| Frame Type | Meaning |
|:---|:---|
| risk | A condition that leads to a failure mode or operational limitation |
| heuristic | A personal practice, preference, or directed advice with rationale |
| principle | A causal explanation of WHY something happens |
| observation | An empirical, factual, or definitional claim (no action, no causal mechanism) |
| comparison | Explicitly weighs two or more options against each other |
| workaround | A non-standard fix done because of a stated limitation |
| case | A specific real-world job or incident narrative |
| other | Does not fit the above (should include an extraction rationale) |

## Post Role Reference

*Defines the functional purpose of the post within the thread.*

- **Question**: The Original Poster (OP) seeking help or information.\
- **Solution**: A direct answer or fix for a specific problem posed.\
- **Addition**: Supplementary info, tips, or alternatives. Facts/techniques provided *without* a personal struggle narrative.\
- **General Advice**: Unprompted guidance not tied to a specific problem in the thread.\
- **Tutorial**: Teaching-oriented explanation of concepts or techniques.\
- **War Story**: A narrative of a **specific** past incident. Must have a concrete arc: *Situation → Problem → Resolution*.\
- **Debate**: Disagreement or back-and-forth between users.\
- **Reaction**: Acknowledgement, thanks, or endorsement without adding new substantive knowledge.\
- **Other**: Anything else (requires explanation in `extraction_rationale`).

## Epistemic Stance Classification List

• **`absolute_authoritative`**

**Tone:** Stated as objective fact using strong, definitive language.

**Keywords:** *must, always, never, cannot, only, required, critical.*

**Rule:** No hedges or "soft" language allowed.

• **`direct_neutral` (DEFAULT)**

**Tone:** Stated as fact with a technical, instructional, or tutorial tone.

**Keywords:** Professional, matter-of-fact phrasing.

**Rule:** Use this when there is no specific emphasis and no hedges.

• **`hedged_emphatic`**

**Tone:** A subjective view stated with conviction. Uses personal qualifiers but strong internal feelings.

**Keywords:** *I feel, in my experience, I find* combined with *hate, required, critical.*

• **`tentative_subjective`**

**Tone:** Weak, uncertain, or cautious language that allows for exceptions.

**Keywords:** *might, could, maybe, I think, seems, usually, typically, some, a few, most of the time.*

• **`narrative_implied`**

**Tone:** The claim is not explicitly stated as a rule but is clearly understood through the context of a story.

**Keywords:** Story-driven, example-based, or describing a specific outcome.

## 

## Rating instructions

For each frame, fill in every field in the response block below it. Type your rating directly after the colon on each line.

**Quality ratings (1-3 scale):**

| Rating | Extraction Accuracy | Tribal Knowledge | Clarity |
|:---|:---|:---|:---|
| 1 | Quote does not support the main point, or main point misrepresents the quote | Generic textbook knowledge \-- easily found in any reference | Poorly written, ambiguous, or hard to understand |
| 2 | Main point mostly captures the quote, but has minor omissions or overstatements | Moderately specialized \-- a practitioner might know this, but it is not trivial | Understandable but could be clearer or more precise |
| 3 | Main point accurately and completely captures the knowledge in the quote | Genuine tribal/experiential knowledge \-- unlikely to be found in textbooks | Clear, precise, and immediately actionable |

**Classification ratings (1-5 scale):**

For Frame Type, Post Role, Scope, Applicability, Epistemic Stance, and Main Point Quality, rate how appropriate the system's label is:

| Rating | Meaning                                            |
|:-------|:---------------------------------------------------|
| 1      | Completely wrong                                   |
| 2      | Poor \-- a different label would clearly be better |
| 3      | Acceptable \-- reasonable but not ideal            |
| 4      | Good \-- correct and appropriate                   |
| 5      | Excellent \-- could not be improved                |

**Overall judgment:**

- **Keep** \-- the frame is accurate, useful, and correctly classified\
- **Revise** \-- the frame captures real knowledge but needs corrections to labels or main point\
- **Remove** \-- the frame is wrong, trivial, or does not represent meaningful extracted knowledge

Use the **Comments** field for anything you want to note: suggested label corrections, concerns about the quote, or general observations.

**Note**: You're not going to have the full information/context around the post. Part of this research is to see how well the frames stand on their own, knowing they are pulling from external context. In a real-world application, a user won't have time to read the whole discussion before deciding how to use a knowledge graph, or frame-based insight.

**Total frames to review:** 26

------------------------------------------------------------------------

## Post Group 1

**Thread:** Is it possible to have rounded (fillet) edges on a PCB? **Post Role:** solution **Post Score:** 10 **Tags:** pcb, pcb-fabrication

### Frame 1

**Source Quote:**

It might not be perfectly rounded but a "beveled edge" should be common for most manufacturers and should not drive the cost up much.

**Main Point (system-generated summary):** Many PCB fabricators can provide a beveled board edge that approximates a rounded feel, and this feature typically adds little cost.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** PCB edge beveling availability and cost impact\
- **Example Context:** PCB fabrication houses offering edge beveling as an option

---REVIEW FRAME 1---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 2

**Source Quote:**

A board separated by V-cut will also be quite rounded, but they don't look very nice, so certainty not the "premium" feel there.

**Main Point (system-generated summary):** V-cut panelization leaves PCB edges somewhat rounded after separation, but the resulting finish is not aesthetically premium.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** V-cut (V-score) panel separation edge aesthetics\
- **Example Context:** PCBs depanelized by V-cut/V-score

---REVIEW FRAME 2---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 3

Scope (1-5): 5

Applicability (1-5): 3

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 3

**Source Quote:**

And as others have said, if you are willing to put money on the table the manufacturer can do just about anything, but they will not even discuss it with you unless you have a profitable relationship with them.

**Main Point (system-generated summary):** Custom or unusual PCB edge features may be possible if you pay for them, but without a profitable ongoing relationship the manufacturer may refuse to discuss such requests at all.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** Manufacturer engagement for custom PCB features depends on profitability/relationship\
- **Example Context:** PCB manufacturers when asked for non-standard edge rounding or specialty finishing

---REVIEW FRAME 3---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 3

Scope (1-5): 3

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Revise

Comments:

------------------------------------------------------------------------

## Post Group 2

**Thread:** What's the best material to use for my project **Post Role:** solution **Post Score:** 2 **Tags:** manufacturing-engineering, metals

### Frame 4

**Source Quote:**

If you wanted to do 50,000 units, the answer is almost certainly injection molding of some kind. You could make what you are describing easily for probably pennies per piece... but it would cost quite a bit to set up the mold. That's probably not worth the setup cost just for 500 pieces.

**Main Point (system-generated summary):** Injection molding delivers very low per-part cost at high volumes (on the order of tens of thousands), but the mold's upfront cost makes it uneconomical for small runs like \~500 units. Choose injection molding only when quantity is high enough to amortize the tooling cost.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** injection molding setup economics\
- **Example Value:** At \~50,000 units, parts can be made for pennies each; at \~500 units, the mold setup cost is not worth it.\
- **Example Context:** Process selection for plastic parts where injection molding is an option, comparing small (\~500) vs large (\~50,000) production volumes.

---REVIEW FRAME 4---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 4

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 4

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 5

**Source Quote:**

Buy some big sheets and have a machine shop slice them into 3" sections with a band saw.

**Main Point (system-generated summary):** For a few hundred simple flat parts, buy large plastic sheets and have a shop bandsaw them into 3-inch-wide blanks to reduce fabrication cost before drilling.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** bandsaw slicing of plastic sheet into blanks\
- **Example Value:** Cut 12"-long by 3"-wide strips from sheet as blanks for subsequent hole drilling.\
- **Example Context:** Low-volume (hundreds) production of simple plastic mounting plates from sheet stock.

---REVIEW FRAME 5---

Extraction Accuracy (1-3): 2

Tribal Knowledge (1-3): 2

Clarity (1-3): 2

Frame Type (1-5): 4

Post Role (1-5): 4

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 3

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Revise

Comments:

Not enough info to make some of the statements made

------------------------------------------------------------------------

### Frame 6

**Source Quote:**

Looks like you could get the raw material in acrylic for less than \$1 per piece. That should be easily enough to drill holes in and should be stiff enough.

**Main Point (system-generated summary):** Acrylic sheet is a very low-cost option (under about \$1 per piece for this application) that is stiff enough and easy to drill for simple mounting plates.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** acrylic sheet suitability and cost\
- **Example Value:** Acrylic blanks for the described plate can be sourced for under \$1 each.\
- **Example Context:** Simple 12" x 3" thin mounting plates requiring several drilled holes for PCB mounting hardware.

---REVIEW FRAME 6---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Revise

Comments:

Nothing was mentioned about mounting plates

------------------------------------------------------------------------

## Post Group 3

**Thread:** What process is involved in freeform 3D tube bending? **Post Role:** tutorial **Post Score:** 1 **Tags:** machine-design, cnc, bending, machine-elements

### Frame 7

**Source Quote:**

This is based on many simplifications, in reality it may be better to determine springback using tests.

**Main Point (system-generated summary):** Because analytical springback models rely on simplifying assumptions, a practical approach is to measure springback empirically with tests rather than relying solely on calculations.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** springback determination via testing

---REVIEW FRAME 7---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 8

**Source Quote:**

In pure bending, axial strains have a linear distribution through section depending on coordinate $z$ and bend radius $R$ :

$$\\epsilon(z) \= \\frac{z}{R}$$

**Main Point (system-generated summary):** Under pure bending, axial strain varies linearly with distance from the neutral axis, equal to z divided by the bend radius R.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** axial strain distribution in pure bending

---REVIEW FRAME 8---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 9

**Source Quote:**

For bends with relatively small radii, we can assume, that the bending stress at applied loads is fully plastic at yield stress $R\_e$ :

**Main Point (system-generated summary):** When bend radii are small in elastoplastic materials, the section under load can be treated as fully plastic, with stress at approximately the yield stress across the thickness.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** fully plastic stress at small bend radii\
- **Example Context:** Small-radius bending of elastoplastic materials (e.g., tube bending where the section yields across most of its thickness).

---REVIEW FRAME 9---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 2

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 10

**Source Quote:**

From this, you can integrate and end up with expression for $k$ (valid for thin-walled tubes):

$$k \= \\frac{4R\_e}{\\pi r E}$$

**Main Point (system-generated summary):** For thin-walled tube bending, the springback parameter k equals 4·Re divided by (π·r·E), relating yield stress, mean radius, and Young's modulus.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** springback curvature constant for thin-walled tubes\
- **Example Context:** Thin-walled tube bending with tube mean radius r, wall thinness assumption, yield stress Re, and modulus E.

---REVIEW FRAME 10---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 11

**Source Quote:**

This constant represents difference between curvature under load and resulting target curvature:

$$k \= \\frac{1}{R\_{load}}-\\frac{1}{R\_{target}}$$

**Main Point (system-generated summary):** In springback modeling, k quantifies how much you must over-bend: it equals the difference between the curvature under load and the final (target) curvature, i.e., 1/R_load minus 1/R_target.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** design_geometry\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** curvature difference between loaded and target bend\
- **Example Context:** Tube bending springback calculations where k has been defined from material/geometry properties.

---REVIEW FRAME 11---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 4

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 4

**Thread:** What electrode material can I use for electrical discharge machining? **Post Role:** solution **Post Score:** 1 **Tags:** discharge, electric-machine, spark, electrode, die

### Frame 12

**Source Quote:**

The deciding factors for choice of electrode material for an electrical discharge machine (EDM - also known as a spark erosion machine) would be its conductivity and erosion resistance.

**Main Point (system-generated summary):** Selecting an EDM electrode material should be based primarily on its electrical conductivity and its resistance to erosion during sparking. These two properties drive electrode performance and life.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** EDM electrode material selection criteria

---REVIEW FRAME 12---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 13

**Source Quote:**

Copper and graphite are the most preferred materials.

**Main Point (system-generated summary):** Copper and graphite are commonly preferred choices for EDM electrode materials due to their favorable properties for the process.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** preferred EDM electrode materials

---REVIEW FRAME 13---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 4

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 14

**Source Quote:**

One wouldn't come across lead or solder for this application and their wear resistance wouldn't be known unless trials are carried out.

**Main Point (system-generated summary):** Lead or solder are not typically used as EDM electrode materials, and their wear resistance is uncertain without performing experimental trials.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** lead or solder as EDM electrode materials

---REVIEW FRAME 14---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 5

**Thread:** Automating the Coolant Top-Up for CNC Machines **Post Role:** solution **Post Score:** 0 **Tags:** machining, cnc

### Frame 15

**Source Quote:**

While it's possible to have a tank of coolant concentrate and mix inline if you have so much work to do that automation is practical and manual checking is impractical then you should consider a coolant recycling system .

**Main Point (system-generated summary):** If the workload justifies automation and manual checking is impractical, choose a coolant recycling system rather than relying on inline mixing to manage coolant concentration.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** adopting a coolant recycling system instead of inline mixing

---REVIEW FRAME 15---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 16

**Source Quote:**

As you have noticed, using inline mixing is still going to require checking the machines; either manually or automatically.

**Main Point (system-generated summary):** Inline mixing does not eliminate oversight; machines still need to be checked for coolant condition, either by manual inspection or automated monitoring.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** inline mixing monitoring requirement

---REVIEW FRAME 16---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 17

**Source Quote:**

What you end up with is coolant with slowly deteriorating properties until you flush out periodically and occasionally have entirely fresh and properly mixed coolant.

**Main Point (system-generated summary):** Relying on periodic top-ups leads to coolant properties that gradually degrade over time, punctuated by periodic flushes that briefly restore proper mix.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** coolant property drift without recycling

---REVIEW FRAME 17---

Extraction Accuracy (1-3):

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 18

**Source Quote:**

With a recycling system you can remove mechanically dispersed fines and tramp oils which wear and gum up the equipment.

**Main Point (system-generated summary):** Coolant recycling systems remove fines and tramp oils, contaminants that otherwise contribute to equipment wear and gumming.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** removal of fines and tramp oils via recycling

---REVIEW FRAME 18---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 19

**Source Quote:**

Using a refractometer the concentration of purified and properly mixed coolant can be maintained all the time,

**Main Point (system-generated summary):** Use a refractometer in a recycling setup to continuously maintain correct coolant concentration rather than allowing it to drift.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** refractometer-based coolant concentration control

---REVIEW FRAME 19---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 20

**Source Quote:**

Payback on equipment purchase is typically 6-9 months

**Main Point (system-generated summary):** Vendors of centralized coolant recycling systems report typical payback periods on equipment of around 6--9 months due to reduced disposal and new coolant costs.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** coolant recycling system payback period (vendor claim)\
- **Example Value:** Quoted claim for the PRAB Guardian Central Coolant Recycling System.\
- **Example Context:** PRAB Guardian Central Coolant Recycling System (vendor claim)

---REVIEW FRAME 20---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 4

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 3

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 3

Overall (Keep / Revise / Remove): Revise

Comments:

The context can be inferred

------------------------------------------------------------------------

## Post Group 6

**Thread:** USB communication keeps disconnecting because of power dip/spike **Post Role:** solution **Post Score:** 0 **Tags:** power-supply, capacitor, usb, usb-device, ups

### Frame 21

**Source Quote:**

I had similar issue, the desktop built-in USB port connection was failing when turning the spindle off (a Dremel via opto-isolated relay).

**Main Point (system-generated summary):** Motherboard USB ports can drop the CNC connection when the spindle is switched off, indicating susceptibility to disturbances during spindle power switching. This manifests as a USB link failure at the moment of turn-off.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** motherboard USB port susceptibility during spindle power switching\
- **Example Value:** On a desktop PC controlling a CNC, the built-in USB port would disconnect when a Dremel spindle (switched via an opto-isolated relay) was turned off.\
- **Example Context:** PC-based CNC controller connected over USB; spindle (Dremel) controlled via opto-isolated relay; disturbance occurs at spindle turn-off.

---REVIEW FRAME 21---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 3

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 22

**Source Quote:**

I happened to have a PCI USB-3 card on the desktop, switched to one of those ports, and it seems to work, so far no disconnects.

**Main Point (system-generated summary):** Switching the CNC's USB connection from a motherboard port to a PCIe USB 3.0 add-in card can stop the disconnects, suggesting those ports are more tolerant of the disturbance from spindle power switching.

**System-assigned labels:**

- **Frame Type:** workaround\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** using PCIe USB 3.0 add-in card to improve USB connection robustness\
- **Example Value:** After repeated disconnects on the built-in USB, moving the cable to a PCI USB-3 card resulted in no disconnects so far.\
- **Example Context:** Desktop PC controlling a CNC over USB, experiencing disconnects when a Dremel spindle (via opto-isolated relay) is turned off; alternative USB provided by a PCIe USB 3.0 card.

---REVIEW FRAME 22---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 7

**Thread:** Trouble with Base circle and meshing with spur gears **Post Role:** solution **Post Score:** 0 **Tags:** mechanical-engineering, gears, meshing

### Frame 23

**Source Quote:**

For a pair of meshing gears the teeth on both need to be the same size ie you don't just scale a standard gear by some factor to get different diameters.

**Main Point (system-generated summary):** Meshing gears must have the same tooth size (module); you cannot obtain a compatible mate by simply scaling a standard gear to a different diameter. Change the tooth count while keeping the module consistent instead.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear tooth size compatibility (module consistency)

---REVIEW FRAME 23---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 24

**Source Quote:**

So as you have noticed, on a larger gear the pitch circle diameter and root diameter will be proportionally closer to each other than on a smaller gear.

**Main Point (system-generated summary):** As gear size increases, the pitch circle diameter and root diameter become proportionally closer to each other compared to smaller gears.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** relative spacing of pitch circle and root diameter vs gear size

---REVIEW FRAME 24---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 25

**Source Quote:**

This is what the Module number is for, its a ratio between the diameter and number of teeth.

**Main Point (system-generated summary):** The module defines tooth size by specifying the ratio of gear diameter to the number of teeth, ensuring compatible meshing when the module is the same.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear module definition

---REVIEW FRAME 25---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 26

**Source Quote:**

These are, conventionally the difference between the pitch circle diameter and the tip and root respectively ie addendum plus dedendum = the height of the tooth above the root circle.

**Main Point (system-generated summary):** By convention, addendum is the radial distance from the pitch circle to the tooth tip and dedendum from the pitch circle to the root; their sum equals the tooth height above the root circle.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** addendum and dedendum definitions and relationship

---REVIEW FRAME 26---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

# Reviewer 4 \-- Knowledge Frame Evaluation

## Background

This study evaluates an AI-powered knowledge extraction pipeline. The pipeline reads Q&A posts from engineering Stack Exchange sites (topics include CNC machining, materials, tooling, and shop practices) and uses a large language model to extract structured **knowledge frames** from each post.

A **knowledge frame** is one discrete, self-contained piece of knowledge extracted from a forum post. Each frame is distinct and is independent from every other frame. A frame captures a single claim, rule, observation, or practice described by the post's author. The extraction system reads the original post text, identifies passages that contain transferable knowledge, and produces a structured record for each one.

## What you are reviewing

Below you will find a set of extracted frames, grouped by the source post they came from. For each frame you will see:

- **Source Quote** (in a blockquote) \-- the verbatim text from the original forum post that the system identified as supporting this frame.\
- **Main Point** \-- a 2-3 sentence summary the system wrote to capture the transferable knowledge claim from the quote.\
- **Classified metadata** \-- labels the system assigned:
  - **Frame Type** \-- what kind of knowledge this is (see reference table below)\
  - **Post Role** \-- how the source post functions in the discussion (e.g., a direct answer, supplementary info, a narrative, etc.)\
  - **Scope** \-- the engineering domain the frame belongs to (e.g., machining_process, material, tooling, cost)\
  - **Applicability** \-- how broadly the claim holds: *universal* (true in virtually all cases), *context_dependent* (true under specific conditions), or *situational* (specific to a particular shop/machine)\
  - **Epistemic Stance** \-- the tone/certainty of the original quote: ranging from *absolute_authoritative* (strong, no hedges) to *narrative_implied* (inferred from a story rather than stated directly)

Each post group also shows contextual metadata from the source forum:

- **Thread** \-- the title of the Q&A thread the post appeared in.\
- **Post Role** \-- what role the post played in the thread.\
- **Post Score** \-- the community vote score on the original post (upvotes minus downvotes; higher = more community agreement).\
- **Tags** \-- topic tags assigned to the thread by the original poster.

## Frame type reference

| Frame Type | Meaning |
|:---|:---|
| risk | A condition that leads to a failure mode or operational limitation |
| heuristic | A personal practice, preference, or directed advice with rationale |
| principle | A causal explanation of WHY something happens |
| observation | An empirical, factual, or definitional claim (no action, no causal mechanism) |
| comparison | Explicitly weighs two or more options against each other |
| workaround | A non-standard fix done because of a stated limitation |
| case | A specific real-world job or incident narrative |
| other | Does not fit the above (should include an extraction rationale) |

## Post Role Reference

*Defines the functional purpose of the post within the thread.*

- **Question**: The Original Poster (OP) seeking help or information.\
- **Solution**: A direct answer or fix for a specific problem posed.\
- **Addition**: Supplementary info, tips, or alternatives. Facts/techniques provided *without* a personal struggle narrative.\
- **General Advice**: Unprompted guidance not tied to a specific problem in the thread.\
- **Tutorial**: Teaching-oriented explanation of concepts or techniques.\
- **War Story**: A narrative of a **specific** past incident. Must have a concrete arc: *Situation → Problem → Resolution*.\
- **Debate**: Disagreement or back-and-forth between users.\
- **Reaction**: Acknowledgement, thanks, or endorsement without adding new substantive knowledge.\
- **Other**: Anything else (requires explanation in `extraction_rationale`).

## Epistemic Stance Classification List

• **`absolute_authoritative`**

**Tone:** Stated as objective fact using strong, definitive language.

**Keywords:** *must, always, never, cannot, only, required, critical.*

**Rule:** No hedges or "soft" language allowed.

• **`direct_neutral` (DEFAULT)**

**Tone:** Stated as fact with a technical, instructional, or tutorial tone.

**Keywords:** Professional, matter-of-fact phrasing.

**Rule:** Use this when there is no specific emphasis and no hedges.

• **`hedged_emphatic`**

**Tone:** A subjective view stated with conviction. Uses personal qualifiers but strong internal feelings.

**Keywords:** *I feel, in my experience, I find* combined with *hate, required, critical.*

• **`tentative_subjective`**

**Tone:** Weak, uncertain, or cautious language that allows for exceptions.

**Keywords:** *might, could, maybe, I think, seems, usually, typically, some, a few, most of the time.*

• **`narrative_implied`**

**Tone:** The claim is not explicitly stated as a rule but is clearly understood through the context of a story.

**Keywords:** Story-driven, example-based, or describing a specific outcome.

## Rating instructions

For each frame, fill in every field in the response block below it. Type your rating directly after the colon on each line.

**Quality ratings (1-3 scale):**

| Rating | Extraction Accuracy | Tribal Knowledge | Clarity |
|:---|:---|:---|:---|
| 1 | Quote does not support the main point, or main point misrepresents the quote | Generic textbook knowledge \-- easily found in any reference | Poorly written, ambiguous, or hard to understand |
| 2 | Main point mostly captures the quote, but has minor omissions or overstatements | Moderately specialized \-- a practitioner might know this, but it is not trivial | Understandable but could be clearer or more precise |
| 3 | Main point accurately and completely captures the knowledge in the quote | Genuine tribal/experiential knowledge \-- unlikely to be found in textbooks | Clear, precise, and immediately actionable |

**Classification ratings (1-5 scale):**

For Frame Type, Post Role, Scope, Applicability, Epistemic Stance, and Main Point Quality, rate how appropriate the system's label is:

| Rating | Meaning                                            |
|:-------|:---------------------------------------------------|
| 1      | Completely wrong                                   |
| 2      | Poor \-- a different label would clearly be better |
| 3      | Acceptable \-- reasonable but not ideal            |
| 4      | Good \-- correct and appropriate                   |
| 5      | Excellent \-- could not be improved                |

**Overall judgment:**

- **Keep** \-- the frame is accurate, useful, and correctly classified\
- **Revise** \-- the frame captures real knowledge but needs corrections to labels or main point\
- **Remove** \-- the frame is wrong, trivial, or does not represent meaningful extracted knowledge

Use the **Comments** field for anything you want to note: suggested label corrections, concerns about the quote, or general observations.

**Total frames to review:** 26

------------------------------------------------------------------------

## Post Group 1

**Thread:** Is it possible to have rounded (fillet) edges on a PCB? **Post Role:** solution **Post Score:** 10 **Tags:** pcb, pcb-fabrication

### Frame 1

**Source Quote:**

It might not be perfectly rounded but a "beveled edge" should be common for most manufacturers and should not drive the cost up much.

**Main Point (system-generated summary):** Many PCB fabricators can provide a beveled board edge that approximates a rounded feel, and this feature typically adds little cost.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** PCB edge beveling availability and cost impact\
- **Example Context:** PCB fabrication houses offering edge beveling as an option

---REVIEW FRAME 1---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 2

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 2

**Source Quote:**

A board separated by V-cut will also be quite rounded, but they don't look very nice, so certainty not the "premium" feel there.

**Main Point (system-generated summary):** V-cut panelization leaves PCB edges somewhat rounded after separation, but the resulting finish is not aesthetically premium.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** V-cut (V-score) panel separation edge aesthetics\
- **Example Context:** PCBs depanelized by V-cut/V-score

---REVIEW FRAME 2---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 3

**Source Quote:**

And as others have said, if you are willing to put money on the table the manufacturer can do just about anything, but they will not even discuss it with you unless you have a profitable relationship with them.

**Main Point (system-generated summary):** Custom or unusual PCB edge features may be possible if you pay for them, but without a profitable ongoing relationship the manufacturer may refuse to discuss such requests at all.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** Manufacturer engagement for custom PCB features depends on profitability/relationship\
- **Example Context:** PCB manufacturers when asked for non-standard edge rounding or specialty finishing

---REVIEW FRAME 3---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 2

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 3

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 2

**Thread:** What's the best material to use for my project **Post Role:** solution **Post Score:** 2 **Tags:** manufacturing-engineering, metals

### Frame 4

**Source Quote:**

If you wanted to do 50,000 units, the answer is almost certainly injection molding of some kind. You could make what you are describing easily for probably pennies per piece... but it would cost quite a bit to set up the mold. That's probably not worth the setup cost just for 500 pieces.

**Main Point (system-generated summary):** Injection molding delivers very low per-part cost at high volumes (on the order of tens of thousands), but the mold's upfront cost makes it uneconomical for small runs like \~500 units. Choose injection molding only when quantity is high enough to amortize the tooling cost.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** injection molding setup economics\
- **Example Value:** At \~50,000 units, parts can be made for pennies each; at \~500 units, the mold setup cost is not worth it.\
- **Example Context:** Process selection for plastic parts where injection molding is an option, comparing small (\~500) vs large (\~50,000) production volumes.

---REVIEW FRAME 4---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 2

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 1

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 2

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 5

**Source Quote:**

Buy some big sheets and have a machine shop slice them into 3" sections with a band saw.

**Main Point (system-generated summary):** For a few hundred simple flat parts, buy large plastic sheets and have a shop bandsaw them into 3-inch-wide blanks to reduce fabrication cost before drilling.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** machining_process\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** bandsaw slicing of plastic sheet into blanks\
- **Example Value:** Cut 12"-long by 3"-wide strips from sheet as blanks for subsequent hole drilling.\
- **Example Context:** Low-volume (hundreds) production of simple plastic mounting plates from sheet stock.

---REVIEW FRAME 5---

Extraction Accuracy (1-3): 2

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 4

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 4

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Revise (see comments)

Comments: Not clear to me if the system hallucinated the *plastic* and *drilling* components of the summary or if they came from a part of the post I am not seeing.

------------------------------------------------------------------------

### Frame 6

**Source Quote:**

Looks like you could get the raw material in acrylic for less than \$1 per piece. That should be easily enough to drill holes in and should be stiff enough.

**Main Point (system-generated summary):** Acrylic sheet is a very low-cost option (under about \$1 per piece for this application) that is stiff enough and easy to drill for simple mounting plates.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** acrylic sheet suitability and cost\
- **Example Value:** Acrylic blanks for the described plate can be sourced for under \$1 each.\
- **Example Context:** Simple 12" x 3" thin mounting plates requiring several drilled holes for PCB mounting hardware.

---REVIEW FRAME 6---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 3

Overall (Keep / Revise / Remove): Revise (see comments)

Comments: Again, not sure if the *mounting plates* part of the summary was hallucinated or comes from some part I am not seeing.

------------------------------------------------------------------------

## Post Group 3

**Thread:** What process is involved in freeform 3D tube bending? **Post Role:** tutorial **Post Score:** 1 **Tags:** machine-design, cnc, bending, machine-elements

### Frame 7

**Source Quote:**

This is based on many simplifications, in reality it may be better to determine springback using tests.

**Main Point (system-generated summary):** Because analytical springback models rely on simplifying assumptions, a practical approach is to measure springback empirically with tests rather than relying solely on calculations.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** springback determination via testing

---REVIEW FRAME 7---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 4

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 4

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 8

**Source Quote:**

In pure bending, axial strains have a linear distribution through section depending on coordinate $z$ and bend radius $R$ :

$$\\epsilon(z) \= \\frac{z}{R}$$

**Main Point (system-generated summary):** Under pure bending, axial strain varies linearly with distance from the neutral axis, equal to z divided by the bend radius R.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** axial strain distribution in pure bending

---REVIEW FRAME 8---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 9

**Source Quote:**

For bends with relatively small radii, we can assume, that the bending stress at applied loads is fully plastic at yield stress $R\_e$ :

**Main Point (system-generated summary):** When bend radii are small in elastoplastic materials, the section under load can be treated as fully plastic, with stress at approximately the yield stress across the thickness.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** fully plastic stress at small bend radii\
- **Example Context:** Small-radius bending of elastoplastic materials (e.g., tube bending where the section yields across most of its thickness).

---REVIEW FRAME 9---

Extraction Accuracy (1-3): 2

Tribal Knowledge (1-3): 1

Clarity (1-3): 1

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 3

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 2

Overall (Keep / Revise / Remove): Revise

Comments: The wording is confusing, and I'm not completely sure it's accurately representing the source

------------------------------------------------------------------------

### Frame 10

**Source Quote:**

From this, you can integrate and end up with expression for $k$ (valid for thin-walled tubes):

$$k \= \\frac{4R\_e}{\\pi r E}$$

**Main Point (system-generated summary):** For thin-walled tube bending, the springback parameter k equals 4·Re divided by (π·r·E), relating yield stress, mean radius, and Young's modulus.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** tutorial\
- **Scope:** material\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** springback curvature constant for thin-walled tubes\
- **Example Context:** Thin-walled tube bending with tube mean radius r, wall thinness assumption, yield stress Re, and modulus E.

---REVIEW FRAME 10---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 2

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 2

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 11

**Source Quote:**

This constant represents difference between curvature under load and resulting target curvature:

$$k \= \\frac{1}{R\_{load}}-\\frac{1}{R\_{target}}$$

**Main Point (system-generated summary):** In springback modeling, k quantifies how much you must over-bend: it equals the difference between the curvature under load and the final (target) curvature, i.e., 1/R_load minus 1/R_target.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** tutorial\
- **Scope:** design_geometry\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** curvature difference between loaded and target bend\
- **Example Context:** Tube bending springback calculations where k has been defined from material/geometry properties.

---REVIEW FRAME 11---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 2

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 4

**Thread:** What electrode material can I use for electrical discharge machining? **Post Role:** solution **Post Score:** 1 **Tags:** discharge, electric-machine, spark, electrode, die

### Frame 12

**Source Quote:**

The deciding factors for choice of electrode material for an electrical discharge machine (EDM - also known as a spark erosion machine) would be its conductivity and erosion resistance.

**Main Point (system-generated summary):** Selecting an EDM electrode material should be based primarily on its electrical conductivity and its resistance to erosion during sparking. These two properties drive electrode performance and life.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** EDM electrode material selection criteria

---REVIEW FRAME 12---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Revise (see comment)

Comments: Again, not sure if the *"These two properties drive electrode performance and life."* part of the summary was hallucinated, as it doesn't appear in the source quote. I think it is correct, but I'm not sure if you're measuring just direct summarization, or if the LLM can add this explanatory information

------------------------------------------------------------------------

### Frame 13

**Source Quote:**

Copper and graphite are the most preferred materials.

**Main Point (system-generated summary):** Copper and graphite are commonly preferred choices for EDM electrode materials due to their favorable properties for the process.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** preferred EDM electrode materials

---REVIEW FRAME 13---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 4

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 14

**Source Quote:**

One wouldn't come across lead or solder for this application and their wear resistance wouldn't be known unless trials are carried out.

**Main Point (system-generated summary):** Lead or solder are not typically used as EDM electrode materials, and their wear resistance is uncertain without performing experimental trials.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** tooling\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** lead or solder as EDM electrode materials

---REVIEW FRAME 14---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 4

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 5

**Thread:** Automating the Coolant Top-Up for CNC Machines **Post Role:** solution **Post Score:** 0 **Tags:** machining, cnc

### Frame 15

**Source Quote:**

While it's possible to have a tank of coolant concentrate and mix inline if you have so much work to do that automation is practical and manual checking is impractical then you should consider a coolant recycling system .

**Main Point (system-generated summary):** If the workload justifies automation and manual checking is impractical, choose a coolant recycling system rather than relying on inline mixing to manage coolant concentration.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** adopting a coolant recycling system instead of inline mixing

---REVIEW FRAME 15---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 16

**Source Quote:**

As you have noticed, using inline mixing is still going to require checking the machines; either manually or automatically.

**Main Point (system-generated summary):** Inline mixing does not eliminate oversight; machines still need to be checked for coolant condition, either by manual inspection or automated monitoring.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** inline mixing monitoring requirement

---REVIEW FRAME 16---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 17

**Source Quote:**

What you end up with is coolant with slowly deteriorating properties until you flush out periodically and occasionally have entirely fresh and properly mixed coolant.

**Main Point (system-generated summary):** Relying on periodic top-ups leads to coolant properties that gradually degrade over time, punctuated by periodic flushes that briefly restore proper mix.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** coolant property drift without recycling

---REVIEW FRAME 17---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 4

Post Role (1-5): 5

Scope (1-5): 3

Applicability (1-5): 3

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 18

**Source Quote:**

With a recycling system you can remove mechanically dispersed fines and tramp oils which wear and gum up the equipment.

**Main Point (system-generated summary):** Coolant recycling systems remove fines and tramp oils, contaminants that otherwise contribute to equipment wear and gumming.

**System-assigned labels:**

- **Frame Type:** principle\
- **Post Role:** solution\
- **Scope:** shop_operations\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** removal of fines and tramp oils via recycling

---REVIEW FRAME 18---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 4

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 19

**Source Quote:**

Using a refractometer the concentration of purified and properly mixed coolant can be maintained all the time,

**Main Point (system-generated summary):** Use a refractometer in a recycling setup to continuously maintain correct coolant concentration rather than allowing it to drift.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** quality\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** refractometer-based coolant concentration control

---REVIEW FRAME 19---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 20

**Source Quote:**

Payback on equipment purchase is typically 6-9 months

**Main Point (system-generated summary):** Vendors of centralized coolant recycling systems report typical payback periods on equipment of around 6--9 months due to reduced disposal and new coolant costs.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** cost\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** coolant recycling system payback period (vendor claim)\
- **Example Value:** Quoted claim for the PRAB Guardian Central Coolant Recycling System.\
- **Example Context:** PRAB Guardian Central Coolant Recycling System (vendor claim)

---REVIEW FRAME 20---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 2

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 4

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 1

Overall (Keep / Revise / Remove): Keep

Comments: Again, not sure if the *"due to reduced disposal and new coolant costs."* part of the summary was hallucinated, as it doesn't appear in the source quote.

------------------------------------------------------------------------

## Post Group 6

**Thread:** USB communication keeps disconnecting because of power dip/spike **Post Role:** solution **Post Score:** 0 **Tags:** power-supply, capacitor, usb, usb-device, ups

### Frame 21

**Source Quote:**

I had similar issue, the desktop built-in USB port connection was failing when turning the spindle off (a Dremel via opto-isolated relay).

**Main Point (system-generated summary):** Motherboard USB ports can drop the CNC connection when the spindle is switched off, indicating susceptibility to disturbances during spindle power switching. This manifests as a USB link failure at the moment of turn-off.

**System-assigned labels:**

- **Frame Type:** risk\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** direct_neutral\
- **Subject:** motherboard USB port susceptibility during spindle power switching\
- **Example Value:** On a desktop PC controlling a CNC, the built-in USB port would disconnect when a Dremel spindle (switched via an opto-isolated relay) was turned off.\
- **Example Context:** PC-based CNC controller connected over USB; spindle (Dremel) controlled via opto-isolated relay; disturbance occurs at spindle turn-off.

---REVIEW FRAME 21---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 4

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 22

**Source Quote:**

I happened to have a PCI USB-3 card on the desktop, switched to one of those ports, and it seems to work, so far no disconnects.

**Main Point (system-generated summary):** Switching the CNC's USB connection from a motherboard port to a PCIe USB 3.0 add-in card can stop the disconnects, suggesting those ports are more tolerant of the disturbance from spindle power switching.

**System-assigned labels:**

- **Frame Type:** workaround\
- **Post Role:** solution\
- **Scope:** machine_capability\
- **Applicability:** context_dependent\
- **Epistemic Stance:** tentative_subjective\
- **Subject:** using PCIe USB 3.0 add-in card to improve USB connection robustness\
- **Example Value:** After repeated disconnects on the built-in USB, moving the cable to a PCI USB-3 card resulted in no disconnects so far.\
- **Example Context:** Desktop PC controlling a CNC over USB, experiencing disconnects when a Dremel spindle (via opto-isolated relay) is turned off; alternative USB provided by a PCIe USB 3.0 card.

---REVIEW FRAME 22---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 3

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 4

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 4

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

## Post Group 7

**Thread:** Trouble with Base circle and meshing with spur gears **Post Role:** solution **Post Score:** 0 **Tags:** mechanical-engineering, gears, meshing

### Frame 23

**Source Quote:**

For a pair of meshing gears the teeth on both need to be the same size ie you don't just scale a standard gear by some factor to get different diameters.

**Main Point (system-generated summary):** Meshing gears must have the same tooth size (module); you cannot obtain a compatible mate by simply scaling a standard gear to a different diameter. Change the tooth count while keeping the module consistent instead.

**System-assigned labels:**

- **Frame Type:** heuristic\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear tooth size compatibility (module consistency)

---REVIEW FRAME 23---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 3

Post Role (1-5): 5

Scope (1-5): 4

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 3

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 24

**Source Quote:**

So as you have noticed, on a larger gear the pitch circle diameter and root diameter will be proportionally closer to each other than on a smaller gear.

**Main Point (system-generated summary):** As gear size increases, the pitch circle diameter and root diameter become proportionally closer to each other compared to smaller gears.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** relative spacing of pitch circle and root diameter vs gear size

---REVIEW FRAME 24---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 25

**Source Quote:**

This is what the Module number is for, its a ratio between the diameter and number of teeth.

**Main Point (system-generated summary):** The module defines tooth size by specifying the ratio of gear diameter to the number of teeth, ensuring compatible meshing when the module is the same.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** gear module definition

---REVIEW FRAME 25---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------

### Frame 26

**Source Quote:**

These are, conventionally the difference between the pitch circle diameter and the tip and root respectively ie addendum plus dedendum = the height of the tooth above the root circle.

**Main Point (system-generated summary):** By convention, addendum is the radial distance from the pitch circle to the tooth tip and dedendum from the pitch circle to the root; their sum equals the tooth height above the root circle.

**System-assigned labels:**

- **Frame Type:** observation\
- **Post Role:** solution\
- **Scope:** design_geometry\
- **Applicability:** universal\
- **Epistemic Stance:** direct_neutral\
- **Subject:** addendum and dedendum definitions and relationship

---REVIEW FRAME 26---

Extraction Accuracy (1-3): 3

Tribal Knowledge (1-3): 1

Clarity (1-3): 3

Frame Type (1-5): 5

Post Role (1-5): 5

Scope (1-5): 5

Applicability (1-5): 5

Main Point Quality (1-5): 5

Epistemic Stance (1-5): 5

Overall (Keep / Revise / Remove): Keep

Comments:

------------------------------------------------------------------------
