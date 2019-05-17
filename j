@article{Huybrechs2009,
abstract = {Newton-Cotes quadrature rules are based on polynomial interpolation in a set of equidistant points. They are very useful in applications where sampled function values are only available on a regular grid. Yet, these rules rapidly become unstable for high orders. In this paper we review two techniques to construct stable high-order quadrature rules using equidistant quadrature points. The stability follows from the fact that all coefficients are positive. This result can be achieved by allowing the number of quadrature points to be larger than the polynomial order of accuracy. The computed approximations then implicitly correspond to the integral of a least squares approximation of the integrand. We show how the underlying discrete least squares approximation can be optimised for the purpose of numerical integration. {\textcopyright} 2009 Elsevier B.V. All rights reserved.},
author = {Huybrechs, Daan},
doi = {10.1016/j.cam.2009.05.018},
issn = {03770427},
journal = {Journal of Computational and Applied Mathematics},
keywords = {Discrete orthogonal polynomials,Least squares approximation,Numerical integration},
title = {{Stable high-order quadrature rules with equidistant points}},
year = {2009}
}
@inproceedings{Menegaz2011,
abstract = {In this work we propose a new set of sigma points for the Unscented Transform that uses the minimum number of points. We than compare this new set with the symmetric set, the reduced set, and the spherical set. Simulations comparing this sets are done to verify the properties of this set and to verify their transforms. Lastly, we simulate each of these sets in a recursive filter for SLAM. The results show that our set is a better choice for a non symmetric prior distribution and still a good alternative for symmetric prior distributions.},
author = {Menegaz, Henrique M. and Ishihara, Joao Y. and Borges, Geovany A.},
booktitle = {Proceedings of the IEEE Conference on Decision and Control},
doi = {10.1109/CDC.2011.6161480},
isbn = {9781612848006},
issn = {01912216},
title = {{A new smallest sigma set for the Unscented Transform and its applications on SLAM}},
year = {2011}
}
@article{Julier1997,
abstract = {The Kalman Filter(KF) is one of the most widely used methods for tracking$\backslash$nand estimation due to its simplicity, optimality, tractability and$\backslash$nrobustness. However, the application of the KF to nonlinear systems$\backslash$ncan be diffcult. The most common approach is to use the Extended$\backslash$nKalman Filter (EKF) which simply linearises all nonlinear models$\backslash$nso that the traditional linear Kalman Filter can be applied. Although$\backslash$nthe EKF (in its many forms) is a widely used filtering strategy,$\backslash$nover thirty years of experience with it has led to a general consensus$\backslash$nwithin the tracking and control community that it is diffcult to$\backslash$nimplement, diffcult to tune, and only reliable for systems which$\backslash$nare almost linear on the time scale of the update intervals. In this$\backslash$npaper a new linear estimator is developed and demonstrated. Using$\backslash$nthe principle that a set of discretely sampled points can be used$\backslash$nto parameterise mean and covariance, the estimator yields performance$\backslash$nequivalent to the KF for linear systems yet generalises elegantly$\backslash$nto nonlinear systems without the linearisation steps required by$\backslash$nthe EKF. We show analytically that the expected performance of the$\backslash$nnew approach is superior to that of the EKF and, in fact, is directly$\backslash$ncomparable to that of the second order Gauss Filter. The method is$\backslash$nnot restricted to assuming that the distributions of noise sources$\backslash$nare Gaussian. We argue that the ease of implementation and more accurate$\backslash$nestimation features of the new lter recommend its use over the EKF$\backslash$nin virtually all applications.},
archivePrefix = {arXiv},
arxivId = {arXiv:1011.1669v3},
author = {Julier, Simon J and Uhlmann, Jeffrey K},
doi = {10.1117/12.280797},
eprint = {arXiv:1011.1669v3},
isbn = {0-7803-6293-4},
issn = {0277786X},
journal = {Int Symp AerospaceDefense Sensing Simul and Controls},
keywords = {estimation,kalman filtering,navigation,non-linear systems,sampling},
pmid = {5098},
title = {{New extension of the Kalman filter to nonlinear systems}},
year = {1997}
}
@article{Sarkka2013,
abstract = {Filtering and smoothing methods are used to produce an accurate estimate of the state of a time-varying system based on multiple observational inputs (data). Interest in these methods has exploded in recent years, with numerous applications emerging in fields such as navigation, aerospace engineering, telecommunications and medicine. This compact, informal introduction for graduate students and advanced undergraduates presents the current state-of-the-art filtering and smoothing methods in a unified Bayesian framework. Readers learn what non-linear Kalman filters and particle filters are, how they are related, and their relative advantages and disadvantages. They also discover how state-of-the-art Bayesian parameter estimation methods can be combined with state-of-the-art filtering and smoothing algorithms. The book's practical and algorithmic approach assumes only modest mathematical prerequisites. Examples include MATLAB computations, and the numerous end-of-chapter exercises include computational assignments. MATLAB/GNU Octave source code is available for download at www.cambridge.org/sarkka, promoting hands-on work with the methods.},
author = {Sarkka, Simo},
doi = {10.1017/CBO9781139344203},
isbn = {9781139344203},
journal = {Cambridge University Press},
pmid = {17963234},
title = {{Bayesian Filtering and Smoothing}},
year = {2013}
}
@article{Zhao-Ming2017,
abstract = {With more satellites launched into orbits during recent years,
monitoring and cataloging of satellites play an important role in
improving the utilization rate of space resource and alleviating the
pressure of orbit resource. Ground-based radar, a kind of sensor in
space surveillance system, does not consider the influences of the
weather and other special circumstances. And it is a key technology in
space target tracking by using the measurement data for real-time orbit
determination. Due to the influence of orbital perturbation, the
satellite orbital dynamic model is a nonlinear system. The optimal
estimation of the orbital state can be achieved by means of nonlinear
filtering based on the measured ranging, velocity and angle data with
measurement noise, which is the essence of real time orbit determination
and has important research value. The extended Kalman filter (EKF) and
unscented Kalman filter (UKF) are most widely used nonlinear Kalman
filters. However, the first-order Taylor expansion of nonlinear function
in EKF degrades the filtering accuracy. And the weight value in UKF
might be negative for the high-dimensional system, which may directly
affect the filtering stability. As an important method in nonlinear
filtering, cubature Kalman filter (CKF) has better accuracy and
stability than UKF. However, CKF only has third-degree filtering
accuracy. In order to improve the filtering accuracy further, some
fifth-degree cubature Kalman filters are proposed, mainly including the
fifth-degree cubature Kalman filter and the fifth-degree spherical
simplex-radial cubature Kalman filter. The optimality of the radial
integral cannot be guaranteed by using the moment matching method in
these fifth-degree filters, so a high-degree cubature quadrature Kalman
filter (HDCQKF) is proposed. The radial integral is calculated using the
high-degree Gauss-Laguerre formula in HDCQKF. However, the
aforementioned filtering algorithm leads to an increase in the number of
cubature points, thereby improving the accuracy, and the number of
cubature points increases polynomially with the increase of system
dimension. Once the algorithm is applied to a high-dimensional system,
or the processor has a relatively poor performance, it may impose a
heavier computing burden, thus the real-time performance decreases.
Therefore, it is necessary to study how to reduce the computational
complexity of the fifth-degree filtering algorithm. In order to improve
the real-time performance of orbit determination on condition that the
accuracy of orbit determination is kept, a novel fifth-degree cubature
Kalman filter for orbit determination is proposed at the lower bound
approaching to the number of cubature points. The key problem in the
nonlinear Kalman filter is to calculate the multidimensional integral in
the form of ``nonlinear function x Gaussian probability density
function{\{}''{\}}, and the integral is approximated using a fifth-degree
numerical cubature rule, in which the number of cubature points required
is only one more than the theoretical lower bound. The abovementioned
cubature rule is embedded into the nonlinear Kalman filtering framework,
from which the update steps of the novel cubature Kalman filter are
derived. Then, the equations of state and measurement for real-time
orbit determination are obtained. The J(2) perturbation and atmospheric
drag perturbation are taken into account in the state equation, and the
coordinate transformation is used to derive the nonlinear relationship
between the orbital state and measurement element.
The simulation results show that the proposed fifth-degree cubature
Kalman filter can achieve a higher filtering accuracy than the CKF and
the same accuracy as the existing fifth-degree filters, but has the
fewest cubature points and the best real-time performance, which proves
the effectiveness of the proposed algorithm.},
author = {Zhao-Ming, Li and Wen-Ge, Yang and Dan, Ding and Yu-Rong, Liao},
doi = {10.7498/aps.66.158401},
issn = {1000-3290},
journal = {ACTA PHYSICA SINICA},
title = {{A novel algorithm of fifth-degree cubature Kalman filter for orbit determination at the lower bound approaching to the number of cubature points}},
year = {2017}
}
@article{Peng2017,
abstract = {A new sparse Gauss-Hermite cubature rule is designed to avoid dimension explosion caused by the traditional full tensor-product based Gauss-Hermite cubature rule. Although Smolyak's quadrature rule can successfully generate sparse cubature points for high dimensional integral, it has a potential drawback that some cubature points generated by Smolyak's rule have negative weights, which may result in instability for the computation. A relative-weight-ratio criterion based sparse Gauss-Hermite rule is presented in this paper, in which cubature points are kept symmetric in the input space and corresponding weights are guaranteed to be positive. The generation of the new sparse cubature points set is simple and meaningful for practice. The difference between our new sparse Gauss-Hermite cubature rule and other cubature rules is analysed. Simulation results show that, compared with Kalman filter with those types of full tensor-product based Gauss-Hermite rules, our new sparse Gauss-Hermite cubature rule based Kalman filter can lead to a substantially reduced number of cubature points, more stable computation capability, and maintaining the accuracy of integration at the same time.},
author = {Peng, Lijun and Duan, Xiaojun and Zhu, Jubo},
doi = {10.1155/2017/2783781},
issn = {1687-5591},
journal = {Modelling and Simulation in Engineering},
title = {{A New Sparse Gauss-Hermite Cubature Rule Based on Relative-Weight-Ratios for Bearing-Ranging Target Tracking}},
year = {2017}
}
@article{Moireau2011,
abstract = {We propose a general reduced-order filtering strategy adapted to Unscented Kalman Filtering for any choice of sampling points distribution. This provides tractable filtering algorithms which can be used with large-dimensional systems when the uncertainty space is of reduced size, and these algorithms only invoke the original dynamical and observation operators, namely, they do not require tangent operator computations, which of course is of considerable benefit when nonlinear operators are considered. The algorithms are derived in discrete time as in the classical UKF formalism – well-adapted to time discretized dynamical equations – and then extended into consistent continuous-time versions. This reduced-order filtering approach can be used in particular for the estimation of parameters in large dynamical systems arising from the discretization of partial differential equations, when state estimation can be handled by an adequate Luenberger observer inspired from feedback control. In this case, we give an analysis of the joint state-parameter estimation procedure based on linearized error, and we illustrate the effectiveness of the approach using a test problem inspired from cardiac biomechanics.},
author = {Moireau, Philippe and Chapelle, Dominique},
doi = {10.1051/cocv/2011001},
file = {:home/jake/Downloads/simplex.pdf:pdf},
issn = {1292-8119},
journal = {ESAIM: Control, Optimisation and Calculus of Variations},
keywords = {and phrases,data assimilation,filtering,identification in pdes,state and parameter estimation},
number = {2},
pages = {406--409},
title = {{Erratum of article “Reduced-order Unscented Kalman Filtering with application to parameter identification in large-dimensional systems”}},
volume = {17},
year = {2011}
}
@article{Adurthi2018,
author = {Adurthi, Nagavenkat and Singla, Puneet and Singh, Tarunraj},
doi = {10.1115/1.4037783},
file = {:home/jake/Downloads/CUT.pdf:pdf},
number = {March},
pages = {1--22},
title = {{Conjugate Unscented Transformation : Applications to Estimation and Control}},
volume = {140},
year = {2018}
}
@article{VanderMerwe2004,
abstract = {Probabilistic inference is the problem of estimating the hidden variables (states or parameters) of a system in an optimal and consistent fashion as a set of noisy or incomplete observations of the system becomes available online. The optimal solution to this problem is given by the recursive Bayesian estimation algorithm which recursively updates the posterior density of the system state as new observations arrive. This posterior density constitutes the complete solution to the probabilistic inference problem, and allows us to calculate any "optimal" estimate of the state. Unfortunately, for most real-world problems, the optimal Bayesian recursion is intractable and approximate solutions must be used. Within the space of approximate solutions, the extended Kalman filter (EKF) has become one of the most widely used algorithms with applications in state, parameter and dual estimation. Unfortunately, the EKF is based on a sub-optimal implementation of the recursive Bayesian estimation framework applied to Gaussian random variables. This can seriously affect the accuracy or even lead to divergence of any inference system that is based on the EKF or that uses the EKF as a component part. Recently a number of related novel, more accurate and theoretically better motivated algorithmic alternatives to the EKF have surfaced in the literature, with specific application to state estimation for automatic control. We have extended these algorithms, all based on derivativeless deterministic sampling based approximations of the relevant Gaussian statistics, to a family of algorithms called Sigma-Point Kalman Filters (SPKF). Furthermore, we successfully expanded the use of this group of algorithms (SPKFs) within the general field of probabilistic inference and machine learning, both as stand-alone filters and as subcomponents of more powerful sequential Monte Carlo methods (particle filters). We have consistently shown that there are large performance benefits to be gained by applying Sigma-Point Kalman filters to areas where EKFs have been used as the de facto standard in the past, as well as in new areas where the use of the EKF is impossible.},
author = {{Van der Merwe}, R.},
doi = {10.6083/M4Z60KZ5},
isbn = {3129163},
issn = {3129163},
journal = {PhD thesis},
pmid = {1201778},
title = {{Sigma-point Kalman filters for probabilistic inference in dynamic state-space models}},
year = {2004}
}
