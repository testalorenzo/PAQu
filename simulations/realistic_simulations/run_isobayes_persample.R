#
# Helper script: run IsoBayes separately on each of the n samples of a
# simulated replicate, producing an n x q matrix of per-isoform Abundance
# estimates (one IsoBayes fit per sample).
#
# Usage:
#   Rscript run_isobayes_persample.R <peptides_csv> <tpm_csv> <output_csv>
#
# <peptides_csv> must have columns: sample, Y, EC  (n*r rows)
# <tpm_csv> must have columns: sample, isoname, tpm  (n*q rows)
# <output_csv> will contain rbind'ed 'isoform_results' (Isoform, Abundance,
# Pi, sample) for each sample
#

suppressMessages(library(IsoBayes))

args = commandArgs(trailingOnly = TRUE)
peptides_csv = args[1]
tpm_csv = args[2]
output_csv = args[3]

peptides_all = read.csv(peptides_csv, stringsAsFactors = FALSE)
tpm_all = read.csv(tpm_csv, stringsAsFactors = FALSE)

samples = sort(unique(peptides_all$sample))

results_list = list()
for (k in samples) {
  df = peptides_all[peptides_all$sample == k, c("Y", "EC")]
  tpm_df = tpm_all[tpm_all$sample == k, c("isoname", "tpm")]

  SE = generate_SE(path_to_peptides_psm = df,
                    abundance_type = "psm",
                    input_type = "other",
                    PEP = FALSE)

  data_loaded = input_data(SE, path_to_tpm = tpm_df)

  set.seed(169612)
  results = suppressMessages(inference(data_loaded, n_cores = 1, K = 2000, burn_in = 1000))

  res = results$isoform_results[, c("Isoform", "Abundance", "Pi")]
  res$sample = k
  results_list[[length(results_list) + 1]] = res
}

out = do.call(rbind, results_list)
write.csv(out, output_csv, row.names = FALSE)
