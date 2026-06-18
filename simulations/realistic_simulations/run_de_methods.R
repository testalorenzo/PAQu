#
# Helper script: run edgeR (QL F-test), DESeq2 (Wald test) and limma-voom on
# a counts matrix (isoforms x samples), comparing two groups (group == 0 vs
# group == 1), as suggested by the IsoBayes authors as a downstream DE step.
#
# Usage:
#   Rscript run_de_methods.R <counts_csv> <group_csv> <output_csv>
#
# <counts_csv> must have a column 'isoform' (row id) and one column per
# sample with non-negative integer counts.
# <group_csv> must have a single column 'group' with 0/1 entries, one row
# per sample, in the same order as the sample columns of <counts_csv>.
# <output_csv> will contain columns: isoform, pval_edger, pval_deseq2,
# pval_deseq2_sf1, pval_limma, pval_aldex2. pval_deseq2_sf1 is DESeq2 with
# size factors fixed to 1 (no internal normalization), appropriate when the
# input counts are already on a common per-sample scale. pval_aldex2 is
# ALDEx2's Welch's t-test p-value (aldex.clr + aldex.ttest), appropriate for
# compositional data.
#

suppressMessages({
  library(edgeR)
  library(DESeq2)
  library(limma)
  library(ALDEx2)
})

args = commandArgs(trailingOnly = TRUE)
counts_csv = args[1]
group_csv = args[2]
output_csv = args[3]

counts = read.csv(counts_csv, row.names = 1, check.names = FALSE)
counts = as.matrix(counts)
group = factor(read.csv(group_csv)$group)

design = model.matrix(~group)

# edgeR (quasi-likelihood F-test)
dge = DGEList(counts = counts, group = group)
dge = calcNormFactors(dge)
dge = estimateDisp(dge, design)
fit = glmQLFit(dge, design)
qlf = glmQLFTest(fit, coef = 2)
pval_edger = qlf$table$PValue

# DESeq2 (Wald test)
coldata = data.frame(group = group)
dds = DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ group)
dds = suppressMessages(DESeq(dds, quiet = TRUE))
res = results(dds)
pval_deseq2 = res$pvalue

# DESeq2 with size factors fixed to 1, skipping DESeq2's own
# normalization -- appropriate when the input "counts" are already on a
# common per-sample scale (e.g. closure-normalized abundance estimates),
# so that DESeq2's median-ratio size factors don't distort the comparison.
dds_sf1 = DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ group)
sizeFactors(dds_sf1) = rep(1, ncol(counts))
dds_sf1 = suppressMessages(DESeq(dds_sf1, quiet = TRUE))
res_sf1 = results(dds_sf1)
pval_deseq2_sf1 = res_sf1$pvalue

# limma-voom
dge2 = DGEList(counts = counts, group = group)
dge2 = calcNormFactors(dge2)
v = voom(dge2, design)
fit2 = lmFit(v, design)
fit2 = eBayes(fit2)
pval_limma = fit2$p.value[, 2]

# ALDEx2: compositional-data-aware DE testing. aldex.clr() draws Monte Carlo
# samples from the Dirichlet posterior of each sample's composition and
# applies a centered log-ratio transform; aldex.ttest() runs Welch's t-test
# (and Wilcoxon) per feature across those MC instances. Appropriate here
# because the input counts (rounded IsoBayes abundances) are themselves a
# closure-normalized composition, which is what ALDEx2 is designed for.
clr = aldex.clr(counts, conds = as.character(group), mc.samples = 128, verbose = FALSE)
aldex_tt = aldex.ttest(clr, verbose = FALSE)
# ALDEx2 silently drops all-zero-count features; reindex back to the full
# isoform set, filling NA for any dropped isoforms.
pval_aldex2 = aldex_tt[rownames(counts), "we.ep"]

out = data.frame(
  isoform = rownames(counts),
  pval_edger = pval_edger,
  pval_deseq2 = pval_deseq2,
  pval_deseq2_sf1 = pval_deseq2_sf1,
  pval_limma = pval_limma,
  pval_aldex2 = pval_aldex2
)
write.csv(out, output_csv, row.names = FALSE)
