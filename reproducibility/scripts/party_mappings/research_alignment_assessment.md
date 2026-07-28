# Research Assessment: Parliamentary Speeches vs. MPDS Manifestos

## Question

Can the parliamentary speech data and MPDS manifesto data be combined to estimate party-manifesto alignment?

Short answer:

- `EE` and `LV`: yes, with good prospects
- `CZ`: yes, with manageable caveats
- `GB`: usable, but only after excluding non-party and transient labels
- `UA`: possible only in a restricted, high-curation design
- `LT`: not assessable yet because the extracted parliamentary speech file is missing

This assessment is based on the standardized mapping files in this folder and the LLM verification file [llm_verification_summary.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/llm_verification_summary.csv).

## Main Finding

The two datasets are usable for manifesto-alignment research only when the unit of comparison is restricted to:

1. speech-side labels that correspond to actual parliamentary parties
2. labels that can be linked to a single MPDS party with high confidence
3. cases where coalition labels, independents, factions, and parliamentary groups are excluded or modeled separately

The central problem is not missing speeches. The central problem is entity harmonization:

- parliamentary corpora often use short local labels
- some labels refer to factions, groups, coalitions, or independent status rather than parties
- MPDS is party-manifesto based, so speech-side parliamentary formations may exist without a manifesto record
- in some countries, especially Ukraine and partly Great Britain, party lineage and label instability are substantial

## Country Summary

| Country | Speech labels | Mapped | Unmapped parties | Unmapped non-party labels | LLM research verdict |
|---|---:|---:|---:|---:|---|
| `EE` | 6 | 6 | 0 | 0 | Strong |
| `LV` | 9 | 9 | 0 | 0 | Strong |
| `CZ` | 12 | 10 | 0 | 2 | Strong after coalition exclusion |
| `GB` | 24 | 11 | 5 | 8 | Moderate, selective use |
| `UA` | 111 | 54 | 23 | 34 | Weak unless heavily restricted |
| `LT` | - | - | - | - | Blocked by missing input |

### Estonia

Files:

- [EE_party_mapping_speech_to_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/EE_party_mapping_speech_to_mpds.csv)

Assessment:

- All observed speech-party labels map cleanly.
- The corrections are substantively plausible and stable.
- This is the cleanest case for manifesto-alignment estimation.

Research use:

- Very good candidate for direct party-level alignment analysis.

### Latvia

Files:

- [LV_party_mapping_speech_to_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/LV_party_mapping_speech_to_mpds.csv)

Assessment:

- All observed speech-party labels map cleanly after manual reconciliation of local Latvian labels.
- The remaining ambiguity is low.
- This is also a strong case for party-level alignment analysis.

Research use:

- Very good candidate for direct party-level alignment analysis.

### Czech Republic

Files:

- [CZ_party_mapping_speech_to_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/CZ_party_mapping_speech_to_mpds.csv)
- [CZ_parliamentary_labels_not_in_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/CZ_parliamentary_labels_not_in_mpds.csv)

Assessment:

- Core party labels map well.
- The two unresolved labels are `SPOLU` and `PirSTAN`, which are coalition labels, not single parties.
- This is not a fatal problem, but it means the speech data and manifesto data are not always aligned at the same organizational level.

Research use:

- Strong if coalition labels are excluded.
- Also viable if coalitions are analyzed separately as parliamentary formations rather than manifesto parties.

### Great Britain

Files:

- [GB_party_mapping_speech_to_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/GB_party_mapping_speech_to_mpds.csv)
- [GB_real_parties_not_in_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/GB_real_parties_not_in_mpds.csv)
- [GB_parliamentary_labels_not_in_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/GB_parliamentary_labels_not_in_mpds.csv)

Assessment:

- The major Westminster parties map well.
- Several speech labels are clearly not manifesto parties: `CB`, `BI`, `I`, `IL`, `LI`, `L8TA`, `QMZZ`, `ZKPW`.
- Some real speech-side parties are present but absent from MPDS, especially `64RT` and `IGC`.
- `PAUB` is likely a real party missing from MPDS.
- `LJ95` and `0UBS` appear to be unstable or transitional labels and are higher-risk for inference.

Research use:

- Moderate.
- Suitable if the analysis is restricted to the cleanly mapped main parties.
- Not suitable for full-party-system inference unless transient and speech-only parties are explicitly handled.

Main risk:

- Selection bias if you compare only mapped parties, because smaller or short-lived parties disappear from the manifesto side.

### Ukraine

Files:

- [UA_party_mapping_speech_to_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/UA_party_mapping_speech_to_mpds.csv)
- [UA_real_parties_not_in_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/UA_real_parties_not_in_mpds.csv)
- [UA_parliamentary_labels_not_in_mpds.csv](/Users/anna/projs/BA/BaThesis/scripts/party_mappings/UA_parliamentary_labels_not_in_mpds.csv)

Assessment:

- Ukraine is substantially more heterogeneous than the other cases.
- The corpus contains not only parties, but also blocs, factions, and parliamentary groups.
- Many labels can be linked only through manual political judgment.
- A nontrivial number of speech-side parties remain unmatched in MPDS.
- Even among matched cases, organizational continuity is sometimes interpretive rather than direct.

Research use:

- Weak for a broad, automated cross-country design.
- Defensible only if restricted to high-confidence direct party mappings.
- High risk if factions, groups, or blocs are mixed with manifesto parties without additional rules.

Main risk:

- Linkage error may be large enough to contaminate substantive estimates of alignment.

### Lithuania

Assessment:

- Cannot be evaluated in this workspace because [ParlaMint-LT_extracted.csv](/Users/anna/projs/BA/BaThesis/data/parlam/ParlaMint-LT_extracted.csv) is missing.

## Implications for Manifesto-Alignment Research

If your goal is to estimate whether parliamentary speech content aligns with party manifestos, the merged dataset is promising, but only under a strict design.

### Good design

- Use only labels with high-confidence one-to-one party mappings.
- Exclude non-party parliamentary labels.
- Exclude coalition labels unless you have a principled coalition-manifesto strategy.
- Treat speech-side parties missing from MPDS as genuine missingness, not failed matching.
- Match manifesto timing to the relevant parliamentary period as closely as possible.

### Bad design

- Treat every value in the speech `party` column as a manifesto party.
- Pool parties, factions, blocs, coalitions, and independents into one comparable unit.
- Ignore speech-side parties that are in parliament but missing from MPDS.
- Assume cross-country comparability is uniform.

## Recommended Analytic Strategy

### For a cleaner comparative design

Use:

- `EE`
- `LV`
- `CZ` after excluding `SPOLU` and `PirSTAN`
- `GB` only for the clearly mapped main parties

Avoid or heavily restrict:

- `UA` unless you manually curate a narrower party subset

### Inclusion rule

For the main analysis, include only rows where:

- `mapping_status == "mapped"`
- `speech_label_type == "party"`

For sensitivity analysis:

- add coalition cases separately
- report speech-side parties that are parliamentary but missing from MPDS
- compare results with and without countries with heavy manual reconciliation

## Overall Verdict

The speech and manifesto datasets are good enough for manifesto-alignment research, but not as a single fully automated harmonized panel.

Best use:

- a restricted comparative design centered on `EE`, `LV`, `CZ`, and a curated subset of `GB`

High-risk use:

- a full pooled design that includes all available Ukrainian labels as if they were directly comparable manifesto parties

The merged data are therefore:

- strong for targeted, high-confidence alignment estimation
- weak for unfiltered full-party-system comparison across all countries in the workspace
