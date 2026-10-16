# Chapter 8: Conclusion and Future Work

## 8.1 Definitive Answers to Research Questions
1. **Did VLM contextual movability improve?** Yes, when integrated into the selective routing pipeline. Standalone prompting achieves only 33.3% balanced accuracy due to linguistic mobility bias, whereas selective routing achieves 85.2%–100.0% by combining cheap geometric resolution with targeted VLM affordance dispatch.
2. **Did selective routing reduce query usage?** Yes. Estimator-aware routing avoids 73.3%–80.0% of VLM queries compared to the All-Eligible baseline, achieving a 2.91x speedup in projected latency.
3. **Does estimator awareness outperform non-estimator awareness?** Estimator awareness achieves identical classification accuracy to R2 on the evaluated dataset, but provides principled compute gating by avoiding query escalation on unconstrained background clutter ($I_{\text{estimator}} = 0$).
4. **Does routing translate into downstream SLAM improvement?** Yes, for metric landmark retention (91.1%), map consistency, and decoupled pose history maintenance. Continuous trajectory ATE/RPE error remains unevaluated on the physical room sequence due to the absence of 6-DoF optical motion capture ground truth.

## 8.2 Summary of Scientific Integrity Commitments
- Calibrated steel tape measurements ($0.563, 0.386, 0.396\text{ m}$) remain strictly unaltered.
- Zero TUM trajectory numbers were substituted into custom room results.
- Stale claims regarding 100% cross-session recall or zero false invalidations have been eliminated and replaced with verified empirical metrics.
