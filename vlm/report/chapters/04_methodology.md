# Chapter 4: Methodology

## 4.1 Coordinate Frames & Rigid SE(2) Registration
Let $\mathcal{F}_1$ and $\mathcal{F}_2$ denote the metric coordinate reference frames of Session 1 and Session 2. Under the assumption of planar ground movement, the coordinate transformation between sessions is modeled as a special Euclidean transformation $\text{SE}(2)$:
$$p_1 = R_{12} p_2 + t_{12}$$
where $R_{12} \in \text{SO}(2)$ represents yaw rotation by angle $\theta$, and $t_{12} = [t_x, t_y]^T \in \mathbb{R}^2$ represents planar horizontal translation.
Parameters are estimated using consensus RANSAC over stationary rigid anchors, achieving an inlier residual RMSE of $0.1198\text{ m}$ across 7 consensus anchors.

## 4.2 Spatial Uncertainty Propagation & Relocation Gating
Let $c_1$ and $c_2$ denote the estimated 3D centroids of an entity observed in Session 1 and Session 2, with spatial position covariances $\Sigma_1$ and $\Sigma_2$. The estimated displacement vector is:
$$\Delta p = c_1 - (R_{12} c_2 + t_{12})$$
The total displacement uncertainty standard deviation $\sigma_{\text{disp}}$ combines observation dispersion and registration residual:
$$\sigma_{\text{disp}} = \sqrt{\sigma_{c1}^2 + \sigma_{c2}^2 + \sigma_{\text{reg}}^2}$$
Under the estimator-aware policy (R3), an entity is classified as physically displaced if and only if:
$$\|\Delta p\| \ge d_{\text{min}} \quad \text{and} \quad \|\Delta p\| \ge 3.0 \cdot \sigma_{\text{disp}}$$
where $d_{\text{min}} = 0.25\text{ m}$.

## 4.3 Estimator-Aware Query Utility Formulation
The decision to escalate an observation to the Vision-Language Model is governed by the expected utility:
$$U(q) = \Delta L_{\text{SLAM}}(q) - \lambda_{\text{cost}} C_{\text{VLM}}$$
where:
- $\Delta L_{\text{SLAM}}(q) = I_{\text{estimator}} \cdot A(q)$
- $I_{\text{estimator}} \in [0.0, 1.0]$ represents the landmark's relevance to the state estimator (observation count, inlier ratio, and spatial consistency).
- $A(q)$ represents affordance ambiguity (unresolved displacement, ambiguous mounting class, or high semantic entropy).
- $C_{\text{VLM}}$ represents the inference cost (normalized latency and token consumption).

## 4.4 Decoupled Policy Definitions (R0 through R4)
- **R0 (Semantic Prior Only)**: Static dictionary lookup mapping class names to prototype mobility categories; zero VLM queries.
- **R1 (Geometry & Motion Only)**: Threshold-based motion gating without semantic reasoning; zero VLM queries.
- **R2 (Non-Estimator-Aware Selective Router)**: Escalates on affordance ambiguity regardless of estimator relevance.
- **R3 (Estimator-Aware Selective Router - Proposed)**: Escalates only when both estimator relevance and query utility are positive.
- **R4 (All-Eligible VLM Reference)**: Dispatches VLM inference on every image-eligible instance.
