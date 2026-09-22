%% HARIS: Expert Systems With Applications Submission
%% elsarticle LaTeX template — preprint (review) mode
%%
%% Journal: Expert Systems With Applications (ESWA)
%% ISSN: 0957-4174  |  Impact Factor: 7.5 (2024), 10.48 (2025 est.)
%% Publisher: Elsevier
%% Submission type: Full Research Article

\documentclass[preprint,12pt]{elsarticle}

%% ── Packages ─────────────────────────────────────────────────────────────────
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{listings}
\usepackage{verbatim}
\usepackage{microtype}

%% FIX: you use enumerate options [label=(\alph*)] → requires enumitem
\usepackage{enumitem}

%% FIX: better URL line breaking (reduces overfull hboxes from long URLs)
\usepackage{xurl}

%% ── Hyperref setup ───────────────────────────────────────────────────────────
\hypersetup{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black
}

%% ── Algorithm style ──────────────────────────────────────────────────────────
\algrenewcommand\algorithmicrequire{\textbf{Input:}}
\algrenewcommand\algorithmicensure{\textbf{Output:}}
\algnewcommand{\LineComment}[1]{\State \(\triangleright\) \textit{#1}}

%% ── Math shortcuts ───────────────────────────────────────────────────────────
\newcommand{\B}{\mathbf{B}}
\newcommand{\P}{\mathbf{P}}
\newcommand{\R}{\mathbf{R}}
\newcommand{\Om}{\boldsymbol{\Omega}}
\newcommand{\calK}{\mathcal{K}}
\newcommand{\calA}{\mathcal{A}}
\newcommand{\calT}{\mathcal{T}}
\newcommand{\calH}{\mathcal{H}}
\newcommand{\calF}{\mathcal{F}}
\newcommand{\kap}{\kappa}
\newcommand{\nuu}{\nu}

%% FIX: reduce overfull/underfull boxes globally (safe for review preprint)
\setlength{\emergencystretch}{2em}

%% FIX (natbib + elsarticle): keep numeric citations but allow \citet author extraction
\biboptions{numbers,sort&compress}

%% ─────────────────────────────────────────────────────────────────────────────
\begin{document}

\begin{frontmatter}

%% ── Title ────────────────────────────────────────────────────────────────────
\title{HARIS: A Hybrid Adaptive Research Intelligence System\\
       for Autonomous Startup Ecosystem Analysis}

%% ── Authors (use \fnref for footnotes) ───────────────────────────────────────
\author[inst1]{[Author~1]\corref{cor1}}
\ead{author1@university.edu}

\author[inst1]{[Author~2]}
\ead{author2@university.edu}

\author[inst2]{[Author~3]}
\ead{author3@university.edu}

\cortext[cor1]{Corresponding author}

\address[inst1]{[Department], [University 1], [City, Country]}
\address[inst2]{[Department], [University 2], [City, Country]}

%% ── Highlights (ESWA requirement: 3–5 bullets, max 85 chars each) ────────────
\begin{highlights}
\item Proposes HARIS, a hybrid multi-agent RAG system for startup ecosystem intelligence
\item Hybrid planner routes 49.7\%/50.3\% catalog/LLM with semantically validated decisions
\item 100\% plan validity on 145 real company descriptions across 10 industry domains
\item Local agents achieve sub-10\,ms evidence retrieval with 100\% success rate
\item Six-level graceful degradation ensures non-empty reports under 30\% failure injection
\end{highlights}

%% ── Abstract ─────────────────────────────────────────────────────────────────
\begin{abstract}
Startup ecosystem intelligence requires synthesizing heterogeneous data
streams---venture funding records, regulatory filings, organic growth signals,
and live market intelligence---a task both labor-intensive and demanding of
multi-domain expertise most early-stage founders lack. We present \textbf{HARIS}
(Hybrid Adaptive Research Intelligence System), a multi-agent framework that
autonomously decomposes a natural-language idea brief into a structured research
plan, dispatches specialized agents to curated data sources, and synthesizes a
grounded business intelligence report via retrieval-augmented generation (RAG).
The central technical contribution is a \emph{hybrid planning module} that
switches between a deterministic keyword-catalog planner and an LLM-driven
adaptive planner based on domain-coverage signal $\kappa$, achieving both
reliability for well-understood sectors and adaptability for emergent domains.
We build and characterize a multi-source evaluation corpus of 83,389$+$ records
spanning three publicly available datasets: 401~US SaaS companies from Crunchbase
(80,902 investment records; 17,152 unique investors), 994 high-growth global
companies from Growjo (median YoY growth: 41.0\%; Kruskal-Wallis inter-industry
significance: $H{=}38.59$, $p{<}0.0001$), and 1,492 company profiles from
Simplify.jobs. Experiments on 145 real Crunchbase descriptions (10 industry
categories, zero-shot) with an expanded 118-term corpus-derived catalog and
recalibrated threshold $\kappa < 0.10$ show both planners achieve 100\% plan
validity; the hybrid routes 49.7\% to the catalog (fast, deterministic) and
50.3\% to the LLM (adaptive), with semantically validated routing based on
domain-signal density. Cross-source generalizability experiments on 100
independent Simplify profiles confirm 100\% plan validity; local agents maintain
100\% success and consistent evidence yield across both sources. A domain-adaptive
SEC query constructor eliminates the majority of false-positive non-operating
entity results. A three-rater pilot evaluation ($n{=}10$ reports;
Krippendorff's $\alpha{=}0.68$) shows HARIS consistently outperforms a
template-only baseline by 1.1--1.8 Likert points across four quality
dimensions. A failure-injection study across 50 sessions ($p_{\text{fail}}{=}0.30$)
confirms the six-level graceful-degradation hierarchy holds unconditionally:
non-empty reports were produced in all 50/50 (100\%) tested sessions.
\end{abstract}

%% ── Keywords ─────────────────────────────────────────────────────────────────
\begin{keyword}
Multi-agent systems \sep Retrieval-augmented generation \sep Business intelligence
\sep Hybrid planning \sep Startup ecosystem \sep Large language models
\sep Evidence synthesis \sep Venture capital analytics \sep SEC filings
\end{keyword}

\end{frontmatter}

%% ─────────────────────────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}
%% ─────────────────────────────────────────────────────────────────────────────

Global venture capital investment reached \$285~billion in 2023, with over
50,000 companies receiving first-time institutional funding~\cite{kpmg2024}.
Yet for most early-stage founders, assessing a new business idea remains a
manual, fragmented, and weeks-long process: pulling comparable funding histories
from Crunchbase~\cite{crunchbase2025}, checking organic revenue-growth signals
from growth-tracker platforms~\cite{growjo2025}, reviewing SEC Form~D filings to
find comparable private placements~\cite{sec2025}, scanning recent news, and
synthesizing these signals into an actionable investment thesis. This gap between
data availability and analytical capacity is the central motivation for HARIS.

Large Language Models (LLMs) have shown strong general reasoning
capabilities~\cite{brown2020,wei2022} and multi-agent frameworks wrapping LLMs
with external tools have tackled complex multi-step tasks~\cite{wang2024survey,
xi2023rise}. However, prior agentic systems---including
AutoGPT~\cite{autogpt2023}, BabyAGI~\cite{babyagi2023}, and
WebGPT~\cite{nakano2021}---fall short for startup intelligence in three ways:
\begin{enumerate}[label=(\alph*)]
  \item they lack \emph{structured integration} of curated financial databases
        alongside live web search;
  \item they offer no \emph{domain-adaptive planning} that degrades gracefully
        when an idea falls outside known category templates; and
  \item they ignore \emph{regulatory filing intelligence} (SEC Form~D, EDGAR),
        which captures early-stage equity raises before they appear in commercial
        databases.
\end{enumerate}

We address all three gaps with HARIS, making the following specific
contributions:

\begin{description}
  \item[\textbf{C1. Hybrid Planning Mechanism}] (Section~\ref{sec:planner}):
    A two-mode planner combining deterministic catalog-based task generation with
    LLM-driven adaptive planning, switching on domain-coverage threshold
    $\kappa < 0.10$ against a 118-term corpus-derived catalog~$\calK$. Both modes
    achieve 100\% plan validity on 145 real company descriptions; the hybrid
    achieves a near-equal 49.7\%\,/\,50.3\% catalog/LLM split that is
    semantically validated. The catalog provides a guaranteed fallback with
    ${<}1$\,ms latency when the LLM path fails.

  \item[\textbf{C2. Multi-Modal Evidence Pipeline}]
    (Section~\ref{sec:agents}): Five specialized agents integrating structured
    Parquet databases (Crunchbase, Growjo), a FAISS semantic vector index
    (SentenceTransformer \texttt{all-MiniLM-L6-v2}), live web search (Brave API),
    geo-targeted news aggregation (SerpAPI), and SEC EDGAR Full-Text Search into
    a unified typed evidence schema.

  \item[\textbf{C3. Domain-Adaptive SEC Query Constructor}]
    (Section~\ref{sec:sec}): An industry-sensitive Boolean query builder for SEC
    Form~D filings that combines industry-keyword-driven Boolean queries with a
    12-pattern exclusion post-filter to eliminate non-operating entity noise.
    Manual inspection of 15 pre-exhaustion queries confirms cleaner retrieval
    versus na\"ive queries.

  \item[\textbf{C4. Layered Graceful Degradation}]
    (Section~\ref{sec:degradation}): A multi-tier fallback hierarchy guaranteeing
    non-empty, human-readable report generation regardless of LLM or API
    availability.

  \item[\textbf{C5. Empirical Corpus and Evaluation}]
    (Section~\ref{sec:eval}): A characterized multi-source corpus of 83,389$+$
    records with statistical analysis, planner benchmarking on 145 real company
    descriptions across 10 domains (zero-shot), cross-source generalizability
    experiments on 100 independent Simplify descriptions, local agent benchmarking,
    and a synthesis quality pilot with inter-rater reliability.
\end{description}

\paragraph{Illustrative example.}
Consider the brief: \emph{``AI-powered diagnostic assistant for rural healthcare
clinics in Southeast Asia.''} Domain tokens include \{diagnostic, healthcare,
clinics\}; $\kappa = 0.43 \ge 0.10$, triggering the catalog plan. For the novel
brief \emph{``Mycelium-based packaging certification platform for food
exporters,''} only ``platform'' and ``food'' match $\calK$; $\kappa = 0.07 <
0.10$, triggering the LLM planner, which generates a custom task list targeting
SEC Form~D filings with materials/biotech terms and Brave Search with
ESG-packaging queries---tasks the catalog cannot generate.

The remainder of the paper is organised as follows. Section~\ref{sec:related}
reviews related work. Section~\ref{sec:problem} formalises the problem.
Section~\ref{sec:arch} describes the architecture. Section~\ref{sec:eval}
presents the evaluation. Section~\ref{sec:discussion} discusses findings and
limitations. Section~\ref{sec:conclusion} concludes.

%% ─────────────────────────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}
%% ─────────────────────────────────────────────────────────────────────────────

\subsection{Multi-Agent LLM Systems}

The coordination of multiple agents to handle complex tasks has roots in
classical distributed AI~\cite{minsky1986} and has been revisited with LLMs.
\citet{yao2023react} introduced \emph{ReAct}, interleaving chain-of-thought
reasoning with tool-action traces for grounded retrieval. \citet{park2023generative}
demonstrated generative agents with emergent social behaviours.
\citet{shen2023hugging} proposed \emph{HuggingGPT}, using ChatGPT as a
meta-controller dispatching to specialised models. \citet{schick2023toolformer}
showed LLMs can autonomously learn API usage through \emph{Toolformer}.
\citet{wang2024survey} and \citet{xi2023rise} survey the rapidly growing
landscape of LLM-based autonomous agents. HARIS differs from these in its
domain-specific architecture: it combines zero-shot LLM planning with
pre-specified agent catalogs optimised for the startup intelligence domain,
without requiring fine-tuning or reinforcement learning.

\subsection{Retrieval-Augmented Generation}

\citet{lewis2020rag} formalised RAG, combining a non-parametric retrieval
component with a parametric generator, demonstrating superior performance on
knowledge-intensive NLP tasks. \citet{karpukhin2020dpr} introduced Dense Passage
Retrieval using dual-encoder BERT models for open-domain QA.
\citet{izacard2021leveraging} showed that fusing multiple retrieved passages via
cross-attention improves generation. HARIS extends RAG to a \emph{multi-source,
multi-modal} setting: evidence is retrieved from structured relational tables,
FAISS vector indices over semantic embeddings, and live APIs, then unified via a
typed schema before synthesis. Unlike standard RAG which retrieves from a
homogeneous document corpus, HARIS navigates five heterogeneous evidence channels
with distinct retrieval strategies per channel.

\subsection{Business Intelligence and Startup Analytics}

Traditional BI systems rely on predefined dashboards over structured data
warehouses~\cite{chaudhuri2011overview}. ML approaches to startup outcome
prediction~\cite{gastaud2019varying,lussier2010three} use Crunchbase features to
predict acquisition or IPO outcomes. \citet{sharchilev2018web} trained gradient-boosted
classifiers on investment event sequences; \citet{arroyo2022analysing} applied
network analysis to co-investment graphs. These systems output predictive scores
rather than natural-language intelligence reports. \citet{balaguer2023automating}
use LLMs to summarise competitive landscapes from web sources alone. HARIS
uniquely integrates structured funding databases, vector-indexed Q\&A, regulatory
filings, and live search into a single orchestrated pipeline generating actionable
natural-language reports grounded in multi-source evidence.

\subsection{Financial and Regulatory Document Analysis}

SEC documents have been analysed for financial signal
extraction~\cite{kogan2009predicting,loughran2011liability}.
\citet{loukas2021edgar} introduced EDGAR-CORPUS for large-scale financial NLP.
Traditional approaches use keyword filters and rule-based extraction. Our
Domain-Adaptive SEC Query Constructor (Section~\ref{sec:sec}) contributes a
dynamic, industry-sensitive Boolean query generation approach with empirically
validated exclusion filtering---advancing beyond static keyword approaches.

\subsection{Hybrid Planning in Autonomous Systems}

Classical AI planning formalised action sequences via STRIPS~\cite{fikes1971strips}
and PDDL. LLM-based planning---exemplified by Chain-of-Thought~\cite{wei2022},
Tree-of-Thought~\cite{yao2023tot}, and LLM+P~\cite{liu2023llmp}---generates
action plans in natural language. \citet{wu2023hybrid} demonstrated that hybrid
(learned + symbolic) planners outperform either component alone on multi-step
reasoning. Our hybrid planner follows this design principle: the deterministic
catalog provides formal validity guarantees, while the LLM component provides
adaptability---with an explicit fallback ensuring system-level reliability.
Unlike LLM+P~\cite{liu2023llmp}, which converts LLM outputs to PDDL for symbolic
solvers, HARIS operates end-to-end in natural language while maintaining
structural validity through post-generation agent-name validation.

%% ─────────────────────────────────────────────────────────────────────────────
\section{Problem Formulation}
\label{sec:problem}
%% ─────────────────────────────────────────────────────────────────────────────

\subsection{Idea Brief}

An \emph{idea brief} $\B = \langle r, p, a, g, m, G, Q \rangle$ where:
$r \in \Sigma^*$ is the raw natural-language startup description;
$p, a, g \in \Sigma^* \cup \{\emptyset\}$ are the problem statement, target
audience, and geographic region; $m \in \{\text{concept, mvp, pre-seed, seed,
growth}\} \cup \{\emptyset\}$ is the maturity stage; $G \subseteq \Sigma^*$ is a
set of goals; and $Q \subseteq \Sigma^*$ is a set of domain assumptions.

\subsection{Agent Registry and Task Plan}

Let $\calA = \{\alpha_1, \ldots, \alpha_k\}$ be the registered agent set. A
\emph{research plan} $\P = (\tau_1, \ldots, \tau_n)$ is an ordered sequence where
each task $\tau_i = (id_i, \alpha_i, ins_i, inp_i, pri_i)$: $id_i \in
\text{UUID}$ is the task identifier; $\alpha_i \in \calA$ is the assigned agent;
$ins_i \in \Sigma^*$ is the natural-language instruction; $inp_i$ is a
\texttt{Dict[str, Any]} of typed agent inputs; $pri_i \in \mathbb{N}$ is the
execution priority (ascending). A plan $\P$ is \emph{valid} if $\forall i$:
$\alpha_i \in \calA$ and $inp_i$ contains all required inputs for $\alpha_i$.

\subsection{Evidence, Synthesis, and Objective}

Each agent $\alpha_i$ executing $\tau_i$ returns $\R_i = (summary, conf,
\varepsilon_i, citations)$ where $\varepsilon_i = \{e_1, \ldots, e_k\}$ is a set
of evidence items $e = (title, content, source, metadata, score)$. The synthesis
function $S$ produces:
\begin{equation}
  \Om = S\!\left(\B,\, \{\R_i\}_{i=1}^{n}\right)
  \label{eq:synthesis}
\end{equation}

\noindent\textbf{System objective:} Produce $\Om \ne \emptyset$ for all valid
$\B$, maximising relevance (evidence grounded in $\B$), completeness (coverage of
funding, competitor, market, regulatory dimensions), and actionability (concrete,
evidence-cited recommendations).

%% ─────────────────────────────────────────────────────────────────────────────
\section{HARIS Architecture}
\label{sec:arch}
%% ─────────────────────────────────────────────────────────────────────────────

Figure~\ref{fig:arch} shows the five-layer HARIS architecture. Processing flows
sequentially from Intake through Planning, Orchestration, Agent Execution, and
Synthesis.

%% FIX: the ASCII diagram was causing "Float too large" sometimes.
%% Put it in a minipage and slightly smaller font.
\begin{figure}[!t]
\centering
\footnotesize
\begin{minipage}{0.97\linewidth}
\begin{verbatim}
+------------------------------------------------------------+
|               User Input B  (REST / CLI)                   |
+----------------------------+-------------------------------+
                             |
             +---------------v--------------+
             |  Layer 1: Intake             |
             |  IdeaInterpreter             |
             |  Normalise --> IdeaBrief(B)  |
             +---------------+--------------+
                             |
             +---------------v--------------+
             |  Layer 2: Hybrid Planner     |
             |  +------------+ +---------+  |
             |  | LLM Planner|<>| Catalog |  |
             |  | (adaptive) | | Planner |  |
             |  +------------+ +---------+  |
             +---------------+--------------+
                             | P=(t1,...,tn)
             +---------------v--------------+
             |  Layer 3: Orchestrator        |
             |  Sequential dispatch + wrap   |
             +--+------+------+------+-------+
                |      |      |      |
         +------v+ +---v--+ +-v--+ +-v---+ +----+
         |Funding| |Comp. | |Web | |Serp | |SEC | Layer 4
         |Agent  | |Agent | |Agt | |Agt  | |Agt | Agents
         +---+---+ +--+---+ +-+--+ +--+--+ +-+--+
             |        |       |       |       |
         Parquet  Parquet  Brave  SerpAPI  SEC API
         +FAISS    DB      API            +XML
                             |
             +---------------v--------------+
             |  Layer 5: Synthesis          |
             |  Curation->LLM->Report Omega |
             |  (Template fallback)         |
             +------------------------------+
\end{verbatim}
\end{minipage}
\caption{HARIS five-layer pipeline. Arrows indicate data flow; $\leftrightarrow$
  indicates the hybrid planner's decision switch.}
\label{fig:arch}
\end{figure}

%% (rest of your manuscript remains unchanged)
%% ...
%% ─────────────────────────────────────────────────────────────────────────────
%% References
%% ─────────────────────────────────────────────────────────────────────────────

%% FIX: use elsarticle-num-names so natbib can extract authors for \citet
\bibliographystyle{elsarticle-num-names}
\bibliography{haris_eswa}

\appendix

\section{Corpus Statistical Summary}

%% FIX: avoid [h] only; let LaTeX place better to avoid float warnings
\begin{table}[!htbp]
\centering
\caption{Growjo high-growth corpus statistics ($n{=}994$).}
\label{tab:growjo_stats}
\begin{tabular}{lrrrr}
\toprule
\textbf{Statistic} & \textbf{Growth \%} & \textbf{Revenue} & \textbf{Valuation} & \textbf{Employees} \\
\midrule
Count (non-null) & 994     & 994       & 181       & 994 \\
Mean             & 129.5\% & \$317.3M  & \$2.68B   & 273 \\
Median           & 41.0\%  & \$52.0M   & \$1.10B   & 208 \\
Std Dev          & 179.8\% & ---       & ---       & --- \\
Q1               & 19.0\%  & ---       & ---       & --- \\
Q3               & 165.0\% & ---       & ---       & --- \\
Maximum          & 1065\%  & \$220.2B  & ---       & --- \\
$>$100\% growth  & 29.8\%  & ---       & ---       & --- \\
$>$200\% growth  & 23.7\%  & ---       & ---       & --- \\
$>$500\% growth  & 5.5\%   & ---       & ---       & --- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htbp]
\centering
\caption{Crunchbase US SaaS corpus statistics ($n{=}401$).}
\label{tab:crunchbase_stats}
\begin{tabular}{lrrrr}
\toprule
\textbf{Statistic} & \textbf{Funding} & \textbf{Rounds} & \textbf{Investors} & \textbf{Revenue Est.} \\
\midrule
Count  & 395      & 399  & 383  & 275 \\
Mean   & \$75.3M  & 4.31 & 8.96 & \$116.7M \\
Median & \$26.8M  & 4    & 7    & \$5.5M \\
Std    & \$158.5M & 2.38 & 8.64 & \$580.9M \\
Max    & \$1.41B  & 12   & 104  & \$5.50B \\
\bottomrule
\end{tabular}
\end{table}

%% (keep rest of appendices as you had them)
\end{document}