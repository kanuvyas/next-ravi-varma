# Visual vocabulary
Base prompt: `rrvarma style, two figures, {phrase}`, seeds 1-2, scale 0.8. Result = single scorer (no second scorer available) (>=1.5 works, >=0.5 partly, else no).

| # | Phrase | Scorer 1 | Scorer 2 | Result | Example image |
|---|---|---|---|---|---|
| 0 | wearing a saffron dhoti and draped shawl | 1 | - | partly | vocab/ph00_s1.png |
| 1 | in a silk sari with a gold border | 2 | - | works | vocab/ph01_s1.png |
| 2 | adorned with pearl necklaces and gold jewelry | 1 | - | partly | vocab/ph02_s1.png |
| 3 | wearing a crown and royal regalia | 2 | - | works | vocab/ph03_s1.png |
| 4 | an ash-smeared sage with matted hair | 0 | - | no | vocab/ph04_s1.png |
| 5 | a warrior in ornate armor holding a bow | 1 | - | partly | vocab/ph05_s1.png |
| 6 | beside a lotus pond | 1 | - | partly | vocab/ph06_s1.png |
| 7 | in a palace hall with carved pillars | 1 | - | partly | vocab/ph07_s1.png |
| 8 | in a forest clearing at dusk | 2 | - | works | vocab/ph08_s1.png |
| 9 | on a battlefield with chariots | 1 | - | partly | vocab/ph09_s1.png |
| 10 | lit by an oil lamp | 1 | - | partly | vocab/ph10_s1.png |
| 11 | under a dramatic cloudy sky | 1 | - | partly | vocab/ph11_s1.png |

M3's prompt expander should prefer phrases that work.
