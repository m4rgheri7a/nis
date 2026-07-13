# Ghostwriter external case

This case is excluded from the main development dataset. Two incident records
from Lithuanian NCSC reports form the historical reference profile. Three
technical-annex cases disclosed by Mandiant in April 2021 are held-out queries.

The split is retrospective. `date_basis=published` means that an underlying
operation date was not asserted and the Mandiant public disclosure date is used
for ordering. Therefore this experiment demonstrates external, source-separated
case applicability, not a live prospective attribution benchmark.

Only indicators printed in the cited public reports are included. Defanged
values such as `88.99.132[.]118` are stored in normalized form for matching.

# Frozen multi-campaign generalisation pilot

`generalization_protocol.yaml` freezes a four-class closed-set pilot before
holdout execution. Each class has two reference events and five holdout events:

- Ghostwriter/UNC1151: NCSC/Mandiant 2020 references; Mandiant 2021 holdouts.
- Doppelganger/RRN: Meta 2022 references; VIGINUM 2023 holdouts.
- Spamouflage/Dragonbridge: Mandiant 2022 references; GTIG 2024 holdouts.
- Storm-1516/Neva Flood: Microsoft 2024 references; VIGINUM 2025 holdouts.

The ranker receives labels only for the eight reference events. Holdout labels
are opened at evaluation time, and holdout events cannot become references for
one another. All IOCs are public-report indicators and none are synthetic.

Source separation is based on evidence-report IDs. It does not always imply an
independent publisher: both Dragonbridge source sets come from the Google/
Mandiant reporting lineage. Event time is used when published; otherwise
`date_basis=published` records the report date. Consequently, the automated
chronology check is not proof of strict operation-time separation for every
record.

The GLASSBRIDGE commercial firms and their full site networks are not labelled
as Dragonbridge actors. The holdouts represent content-placement cases that the
GTIG report explicitly links to Dragonbridge material. Because some report text
describes the surrounding distribution ecosystem rather than a single post,
this class has greater label-scope uncertainty than the other three classes.

This benchmark evaluates ranking among four known campaign hypotheses. It does
not evaluate discovery of an unseen actor, state attribution, identity proof,
or legal attribution.
