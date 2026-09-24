# Numeric artifact schema

The TF files retain only pseudonymous question/scene identifiers, branch acceptance, original/mapped seven-token mean log likelihoods, and their directional gap. Question and scene pseudonyms preserve the original sort order, so the frozen RNG's scene-bootstrap draw order is identical. No images, prompts, free-form completions or personal annotation records are included.

The generation files retain parsed A–D answers and max-length indicators only, with original/mapped option labels and per-branch draw counts. Invalid answers remain in the denominator. The subset file encodes the exact 107 aligned questions and seven supported-axis TF questions using the same pseudonyms; it contains no individual annotator responses.

Bridge data retain one row per scene/arm with the mean mapped boundary probability over that arm's valid prefixes, suffix hit/total counts, complete-generation counts and number of valid prefixes. Scene-equal suffix rates average each scene's hit fraction; they do not divide pooled hits by pooled totals.

The numerical script reproduces eight TF comparisons (four per seed), two matched-generation comparisons, all branch absolute candidate differences and sign transitions, and nine local bridge contrasts. The disjoint-scene bridge results are supplied as aggregate values in results/table_plotted_values.json; raw disjoint-generation text is excluded. Reproducing figures from fresh training requires collecting outputs using the original instruments, with the same frozen metadata and processing conventions.
