"""
Experiment 1: Planner Reliability and Novel-Domain Coverage
Runs the hybrid/catalog/LLM planners on 150 synthetic idea briefs across 10 industries.
Measures: plan validity %, novel-domain task accuracy %, mean tasks/plan, LLM invocations %.
"""
import sys, time, json, statistics
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.schemas import IdeaBrief, MaturityStage
from backend.planning.research_planner import ResearchPlanner
from backend.agents.orchestrator import DEFAULT_AGENT_CONFIGS

AVAILABLE_AGENTS = {cfg.name for cfg in DEFAULT_AGENT_CONFIGS}

# Ground-truth: for each industry category, which agents are appropriate
GROUND_TRUTH_AGENTS = {
    "tech":        {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "fintech":     {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "healthcare":  {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "cleantech":   {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "edtech":      {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "logistics":   {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "retail":      {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "biotech":     {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "legaltech":   {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SerpSearchAgent","SecFilingsAgent"},
    "novel":       {"FundingIntelligenceAgent","CompetitorGrowthAgent","WebResearchAgent","SecFilingsAgent"},
}

BRIEFS = {
"tech": [
    ("AI-powered code review platform for enterprise dev teams",
     "Developers waste 30% of review time on style/formatting issues",
     "senior engineers at Fortune 500 firms", "United States", MaturityStage.seed),
    ("No-code API integration builder for SMBs",
     "SMBs cannot afford developers to integrate SaaS tools",
     "small business owners", "United States", MaturityStage.mvp),
    ("Real-time database migration tool for cloud-native apps",
     "Zero-downtime schema migrations are painful at scale",
     "DevOps engineers", "United States", MaturityStage.pre_seed),
    ("AI-assisted incident response platform for SRE teams",
     "Mean time to resolve outages is too high",
     "site reliability engineers", "United States", MaturityStage.seed),
    ("Serverless cost optimization SaaS for AWS workloads",
     "Cloud spend is unpredictable and over-provisioned",
     "cloud architects", "United States", MaturityStage.mvp),
    ("Open-source observability platform with AI anomaly detection",
     "Existing APM tools are expensive and vendor-locked",
     "DevOps teams", "United States", MaturityStage.growth),
    ("Developer productivity analytics for remote engineering teams",
     "Engineering managers lack objective velocity metrics",
     "VPs of Engineering", "United States", MaturityStage.seed),
    ("Automated technical documentation generator from code diffs",
     "Docs drift out of sync with code changes",
     "software architects", "United States", MaturityStage.mvp),
    ("Privacy-first AI data pipeline builder",
     "Data teams struggle with GDPR-compliant ML pipelines",
     "ML engineers", "United States", MaturityStage.pre_seed),
    ("Browser-native IDE for pair programming with AI",
     "Remote pair programming requires clunky screen sharing",
     "freelance developers", "United States", MaturityStage.concept),
    ("Platform for selling internal developer tools as SaaS",
     "Companies build tools internally that have commercial potential",
     "startup CTOs", "United States", MaturityStage.pre_seed),
    ("Automated API versioning and deprecation management",
     "API breaking changes cause downstream failures",
     "API platform teams", "United States", MaturityStage.mvp),
    ("AI code migration assistant from legacy Java to modern stack",
     "Legacy codebases block digital transformation",
     "enterprise IT directors", "United States", MaturityStage.seed),
    ("Cross-cloud database replication with conflict resolution",
     "Multi-cloud data consistency is unsolved",
     "cloud DBAs", "United States", MaturityStage.pre_seed),
    ("Instant test environment provisioning for CI/CD",
     "QA bottleneck slows release velocity",
     "QA engineers", "United States", MaturityStage.mvp),
    ("AI-powered software architecture review platform",
     "Architecture anti-patterns are caught too late",
     "software architects", "United States", MaturityStage.seed),
    ("Developer-first compliance automation for SOC 2",
     "SOC 2 certification takes 18 months manually",
     "startup CTOs", "United States", MaturityStage.growth),
    ("Real-time collaborative SQL editor for data teams",
     "Data analysts duplicate query work without collaboration",
     "data analysts", "United States", MaturityStage.mvp),
    ("Automated dependency vulnerability remediation",
     "Open-source CVEs are not patched quickly enough",
     "security engineers", "United States", MaturityStage.seed),
    ("LLM-powered codebase onboarding assistant",
     "New engineers take 3 months to become productive",
     "engineering managers", "United States", MaturityStage.mvp),
],
"fintech": [
    ("B2B embedded lending platform for SaaS companies",
     "SaaS platforms lose customers who can't afford upfront fees",
     "SaaS CFOs and SMB owners", "United States", MaturityStage.seed),
    ("Real-time treasury management for mid-market companies",
     "CFOs lack visibility into global cash positions",
     "corporate treasurers", "United States", MaturityStage.growth),
    ("Crypto payroll infrastructure for remote-first companies",
     "Cross-border payroll is slow and expensive",
     "HR teams at distributed companies", "United States", MaturityStage.mvp),
    ("AI-powered credit scoring for thin-file borrowers",
     "Traditional FICO excludes 40M credit-invisible Americans",
     "community banks", "United States", MaturityStage.seed),
    ("Automated accounts payable using OCR and AI",
     "AP teams manually process 500+ invoices per week",
     "mid-market finance teams", "United States", MaturityStage.growth),
    ("Subscription revenue financing for SaaS startups",
     "Recurring revenue is underused as collateral",
     "SaaS founders", "United States", MaturityStage.seed),
    ("Open banking data aggregation API for EU fintech",
     "PSD2 compliance is complex for new entrants",
     "European fintech developers", "Germany", MaturityStage.pre_seed),
    ("Insurance claims automation using computer vision",
     "Claims processing takes 30+ days and is fraud-prone",
     "P&C insurance carriers", "United States", MaturityStage.seed),
    ("Fractional real estate investment platform",
     "Retail investors cannot access commercial real estate",
     "retail investors aged 25-45", "United States", MaturityStage.growth),
    ("KYC/AML automation for crypto exchanges",
     "Compliance burden is prohibitive for new crypto platforms",
     "crypto exchange operators", "United States", MaturityStage.seed),
    ("Cross-border payment infrastructure for African SMBs",
     "African trade payments are 5x more expensive than global average",
     "African exporters/importers", "Nigeria", MaturityStage.seed),
    ("Revenue-based financing marketplace for e-commerce brands",
     "D2C brands need non-dilutive growth capital",
     "e-commerce founders", "United States", MaturityStage.growth),
    ("Real-time financial data API for investment research",
     "Quant researchers need normalized multi-source data",
     "quantitative analysts", "United States", MaturityStage.mvp),
    ("Automated tax optimization engine for freelancers",
     "Self-employed workers overpay by average $3,400/year",
     "freelancers and gig workers", "United States", MaturityStage.mvp),
    ("ESG investment data platform for asset managers",
     "ESG data is inconsistent across rating agencies",
     "institutional asset managers", "United States", MaturityStage.seed),
],
"healthcare": [
    ("AI diagnostic assistant for rural primary care",
     "Rural clinics lack specialist access for differential diagnosis",
     "rural family physicians", "United States", MaturityStage.seed),
    ("Patient intake automation reducing ER wait times",
     "Hospital EDs average 4-hour wait times",
     "hospital administrators", "United States", MaturityStage.growth),
    ("Remote patient monitoring platform for chronic disease",
     "85M Americans with chronic conditions need continuous monitoring",
     "health systems and payers", "United States", MaturityStage.growth),
    ("AI-powered prior authorization automation",
     "PA denials delay care and cost $35B annually",
     "health system revenue cycle teams", "United States", MaturityStage.seed),
    ("Clinical trial patient matching platform",
     "75% of clinical trials fail to recruit enough patients on time",
     "pharmaceutical clinical operations teams", "United States", MaturityStage.seed),
    ("Mental health platform for employer-sponsored benefits",
     "Only 43% of employees with mental illness receive treatment",
     "HR benefits managers at enterprise companies", "United States", MaturityStage.growth),
    ("Surgical robotics data analytics platform",
     "Surgeons lack outcome analytics to improve technique",
     "hospital surgery departments", "United States", MaturityStage.seed),
    ("AI-powered radiology second opinion platform",
     "Radiology error rates remain at 3-5%",
     "radiologists and hospital networks", "United States", MaturityStage.mvp),
    ("Medication adherence platform for elderly patients",
     "Non-adherence causes 125,000 preventable deaths annually",
     "home health agencies and pharmacies", "United States", MaturityStage.mvp),
    ("Value-based care analytics for ACOs",
     "ACOs struggle to identify high-risk patients proactively",
     "accountable care organization directors", "United States", MaturityStage.growth),
    ("Digital therapeutics for Type 2 diabetes management",
     "Lifestyle intervention programs have poor adherence",
     "endocrinologists and payers", "United States", MaturityStage.seed),
    ("Real-time surgical supply chain visibility",
     "OR supply shortages delay surgeries costing $3,000/minute",
     "hospital supply chain managers", "United States", MaturityStage.mvp),
    ("AI-powered nursing documentation reduction tool",
     "Nurses spend 30% of shift time on documentation",
     "hospital nursing directors", "United States", MaturityStage.seed),
    ("Genetic counseling platform for consumer genomics",
     "Consumer DNA tests lack actionable clinical interpretation",
     "primary care physicians and consumers", "United States", MaturityStage.mvp),
    ("Healthcare revenue cycle AI for independent practices",
     "Independent practices lose 15% revenue to billing errors",
     "practice managers at independent physician groups", "United States", MaturityStage.growth),
],
"cleantech": [
    ("Community solar subscription platform for renters",
     "65% of US households cannot install rooftop solar",
     "residential renters in deregulated markets", "United States", MaturityStage.growth),
    ("Carbon credit marketplace for voluntary offsets",
     "Voluntary carbon markets lack price transparency",
     "sustainability officers at enterprises", "United States", MaturityStage.seed),
    ("EV fleet management software for last-mile logistics",
     "Fleets transitioning to EVs lack range optimization tools",
     "logistics fleet managers", "United States", MaturityStage.growth),
    ("Industrial heat electrification platform for manufacturers",
     "Industrial heating accounts for 20% of global CO2",
     "manufacturing plant engineers", "United States", MaturityStage.seed),
    ("Battery-as-a-Service for commercial real estate",
     "Building owners want storage but not capex",
     "commercial real estate operators", "United States", MaturityStage.pre_seed),
    ("Water recycling automation for semiconductor fabs",
     "Chip fabs use 10M gallons/day and face water scarcity risk",
     "semiconductor fab operations managers", "United States", MaturityStage.seed),
    ("Methane leak detection via satellite analytics",
     "Oil and gas methane leaks are systematically underreported",
     "oil and gas ESG compliance teams", "United States", MaturityStage.mvp),
    ("Green hydrogen infrastructure optimization platform",
     "Hydrogen electrolyzer economics are poorly understood at scale",
     "energy infrastructure developers", "United States", MaturityStage.pre_seed),
    ("AI-powered energy procurement for corporate buyers",
     "Large companies overpay for energy due to poor contract timing",
     "corporate energy managers", "United States", MaturityStage.growth),
    ("Biodiversity credit measurement and trading platform",
     "Nature-positive investing has no standardized metric",
     "conservation finance investors", "United States", MaturityStage.pre_seed),
    ("Circular economy materials marketplace for manufacturing",
     "Industrial scrap with residual value goes to landfill",
     "procurement managers at manufacturers", "United States", MaturityStage.mvp),
    ("Offshore wind project permitting AI assistant",
     "Offshore wind permitting takes 7-10 years",
     "offshore wind developers", "United States", MaturityStage.seed),
    ("Demand flexibility aggregation for commercial buildings",
     "Grid operators need demand response but lack aggregated assets",
     "energy managers at commercial property firms", "United States", MaturityStage.seed),
    ("Climate risk analytics for real estate lenders",
     "Physical climate risk is mispriced in mortgage portfolios",
     "risk managers at regional banks", "United States", MaturityStage.mvp),
    ("Sustainable packaging certification platform for CPG",
     "CPG companies lack credible packaging sustainability scoring",
     "sustainability teams at consumer goods firms", "United States", MaturityStage.seed),
],
"edtech": [
    ("AI tutoring platform for K-12 math personalization",
     "Math learning gaps widen after COVID disruption",
     "K-12 schools and parents", "United States", MaturityStage.growth),
    ("Upskilling platform for manufacturing workers facing automation",
     "2 million US manufacturing jobs will be automated by 2030",
     "HR teams at industrial companies", "United States", MaturityStage.seed),
    ("Coding bootcamp outcomes guarantee model",
     "Bootcamp graduates carry debt risk without job placement",
     "career changers aged 25-40", "United States", MaturityStage.growth),
    ("AI-powered language learning for business professionals",
     "Standard language apps fail adult learners with busy schedules",
     "international business professionals", "United States", MaturityStage.growth),
    ("School mental health early warning system",
     "Student mental health crises are identified too late",
     "school counselors and administrators", "United States", MaturityStage.mvp),
    ("Corporate learning management system with AI pathways",
     "LMS completion rates average only 15%",
     "L&D managers at enterprises", "United States", MaturityStage.growth),
    ("Micro-credential marketplace for community college students",
     "Community college students need faster pathways to employment",
     "community college administrators", "United States", MaturityStage.mvp),
    ("Adaptive practice platform for medical licensing exams",
     "USMLE Step 1 failure rates are rising",
     "medical students", "United States", MaturityStage.seed),
    ("Teacher professional development platform using lesson analytics",
     "Teachers lack data-driven feedback on classroom performance",
     "school district administrators", "United States", MaturityStage.seed),
    ("Trade skills training simulator using AR",
     "Electricians and plumbers train on outdated equipment",
     "vocational schools and union apprenticeship programs", "United States", MaturityStage.pre_seed),
    ("AI essay feedback for university writing centers",
     "University writing centers are underfunded and overwhelmed",
     "university writing center directors", "United States", MaturityStage.mvp),
    ("Parent engagement platform bridging home-school communication",
     "Low parental engagement predicts academic underperformance",
     "K-12 school principals", "United States", MaturityStage.mvp),
    ("Workforce readiness analytics for higher education",
     "Colleges lack employer-aligned curriculum signals",
     "university career services offices", "United States", MaturityStage.seed),
    ("International student credential verification SaaS",
     "Foreign credential recognition takes 6-12 months",
     "university admissions offices", "United States", MaturityStage.mvp),
    ("Live interactive coding challenges for CS education",
     "CS assessments fail to capture real-world problem solving",
     "CS instructors at universities", "United States", MaturityStage.mvp),
],
"logistics": [
    ("AI freight forwarding platform for cross-border SME shipments",
     "SMEs pay 40% premiums on international freight",
     "small and mid-size importers/exporters", "United States", MaturityStage.growth),
    ("Last-mile delivery orchestration for rural e-commerce",
     "Rural delivery failure rates exceed 20%",
     "e-commerce retailers and 3PLs", "United States", MaturityStage.seed),
    ("Cold chain IoT monitoring for pharmaceutical logistics",
     "Temperature excursion events waste $35B in pharma product annually",
     "pharmaceutical logistics managers", "United States", MaturityStage.seed),
    ("Autonomous warehouse robotics-as-a-service",
     "Warehouse labor shortage drives 30% labor cost inflation",
     "mid-size 3PL warehouse operators", "United States", MaturityStage.growth),
    ("Dynamic truck load matching marketplace",
     "40% of truck miles are driven empty in the US",
     "trucking carriers and freight brokers", "United States", MaturityStage.growth),
    ("Port drayage scheduling optimization platform",
     "Container drayage scheduling is manual and inefficient",
     "drayage carriers and port terminal operators", "United States", MaturityStage.seed),
    ("Returns management automation for DTC brands",
     "Returns cost DTC brands 20% of revenue",
     "DTC brand operations teams", "United States", MaturityStage.mvp),
    ("Supply chain risk intelligence platform",
     "Geopolitical disruption is invisible until it hits",
     "supply chain directors at manufacturers", "United States", MaturityStage.seed),
    ("Sustainable last-mile delivery with cargo bikes",
     "Urban delivery emissions face imminent municipal restrictions",
     "urban delivery operators and retailers", "United States", MaturityStage.mvp),
    ("AI customs classification and compliance for importers",
     "HS code misclassification triggers audits and penalties",
     "import compliance managers", "United States", MaturityStage.mvp),
],
"retail": [
    ("AI-powered inventory optimization for specialty retailers",
     "Specialty retailers carry 20% excess inventory on average",
     "retail operations managers", "United States", MaturityStage.growth),
    ("Unified commerce platform for omnichannel brands",
     "Multi-channel inventory visibility is fragmented",
     "e-commerce directors at mid-market brands", "United States", MaturityStage.growth),
    ("Conversational commerce AI for WhatsApp retail",
     "Emerging market retailers need mobile-first sales channels",
     "small retailers in Latin America", "Brazil", MaturityStage.seed),
    ("Predictive store traffic analytics using computer vision",
     "Brick-and-mortar retailers lack granular footfall intelligence",
     "retail real estate managers", "United States", MaturityStage.seed),
    ("Social commerce aggregator for Gen Z creators",
     "Creator-commerce is fragmented across platforms",
     "Gen Z micro-influencers and their audiences", "United States", MaturityStage.mvp),
    ("Dynamic pricing engine for fashion clearance",
     "Fashion retailers destroy $50B of unsold inventory annually",
     "merchandise planning teams at apparel brands", "United States", MaturityStage.seed),
    ("Grocery waste reduction platform using demand sensing",
     "US grocery retailers lose 8% of revenue to food waste",
     "grocery category managers", "United States", MaturityStage.seed),
    ("B2B wholesale marketplace for independent retailers",
     "Independent retailers lack access to wholesale price discovery",
     "independent retail store owners", "United States", MaturityStage.growth),
    ("AR try-before-you-buy for furniture e-commerce",
     "Furniture return rates of 30% destroy margins",
     "furniture e-commerce brands", "United States", MaturityStage.mvp),
    ("Loyalty program consolidation platform for local merchants",
     "Consumers ignore fragmented single-merchant loyalty apps",
     "local business district associations", "United States", MaturityStage.mvp),
],
"biotech": [
    ("AI-powered protein structure prediction for drug discovery",
     "Target identification is the slowest step in drug development",
     "pharmaceutical R&D directors", "United States", MaturityStage.seed),
    ("CRISPR gene editing workflow automation platform",
     "CRISPR protocol design takes weeks of manual work",
     "academic and commercial genomics labs", "United States", MaturityStage.pre_seed),
    ("Cell line authentication SaaS for biopharma",
     "Cell line misidentification causes 20% of failed experiments",
     "biopharma quality assurance managers", "United States", MaturityStage.mvp),
    ("Lab notebook automation using AI for wet labs",
     "Manual ELN entry loses 3 hours of researcher time per day",
     "research scientists at biotech companies", "United States", MaturityStage.mvp),
    ("Biomarker discovery platform using federated learning",
     "Clinical data siloes slow biomarker research",
     "clinical research organizations", "United States", MaturityStage.seed),
    ("Synthetic biology parts registry and characterization platform",
     "Synthetic biology components lack standardized performance data",
     "synthetic biology researchers", "United States", MaturityStage.pre_seed),
    ("AI-driven formulation optimization for drug delivery",
     "Oral drug formulation development takes 18 months",
     "pharmaceutical formulation scientists", "United States", MaturityStage.seed),
    ("Rapid diagnostics platform for AMR pathogen detection",
     "Antibiotic-resistant infections kill 700,000 people annually",
     "hospital infection control teams", "United States", MaturityStage.pre_seed),
    ("NGS data analysis pipeline for clinical genomics labs",
     "Clinical genomics labs lack scalable bioinformatics",
     "clinical genomics laboratory directors", "United States", MaturityStage.seed),
    ("Regulatory submission automation for IND applications",
     "IND application preparation takes 18 months and $2M",
     "biotech regulatory affairs teams", "United States", MaturityStage.mvp),
],
"legaltech": [
    ("AI contract review platform for M&A due diligence",
     "M&A due diligence review takes 6-8 weeks manually",
     "M&A associates at mid-size law firms", "United States", MaturityStage.growth),
    ("Automated lease abstraction for commercial real estate",
     "RE firms manually review thousands of leases per transaction",
     "commercial real estate legal teams", "United States", MaturityStage.growth),
    ("IP portfolio analytics and maintenance automation",
     "Patent maintenance fees are poorly tracked by SMBs",
     "IP counsel at technology companies", "United States", MaturityStage.mvp),
    ("Immigration case management SaaS for law firms",
     "Immigration firms manage cases on spreadsheets",
     "immigration law firms", "United States", MaturityStage.growth),
    ("E-discovery AI for reducing document review cost",
     "E-discovery accounts for 70% of litigation cost",
     "litigation partners at large law firms", "United States", MaturityStage.growth),
    ("Regulatory change management platform for compliance teams",
     "Compliance teams manually track thousands of regulatory updates",
     "compliance officers at financial institutions", "United States", MaturityStage.seed),
    ("Legal billing analytics and pricing intelligence",
     "Law firms underprice complex matters by 25% on average",
     "law firm managing partners", "United States", MaturityStage.mvp),
    ("AI-powered trademark monitoring and enforcement",
     "Brand owners miss 60% of trademark infringement online",
     "brand managers and IP counsel", "United States", MaturityStage.mvp),
    ("Contract lifecycle management for mid-market companies",
     "Mid-market legal teams lose contracts in email threads",
     "general counsel at mid-market companies", "United States", MaturityStage.growth),
    ("Automated court filing and deadline management",
     "Missed court deadlines result in malpractice claims",
     "litigation paralegals and attorneys", "United States", MaturityStage.mvp),
],
"novel": [
    # Genuinely novel -- tokens outside catalog K
    ("Mycelium-based packaging certification platform for food exporters",
     "No standardized sustainability rating exists for bio-packaging",
     "sustainable packaging procurement teams", "United States", MaturityStage.pre_seed),
    ("Permaculture farm design AI for regenerative agriculture",
     "Regenerative farmers lack design optimization tools",
     "small-scale regenerative farmers", "United States", MaturityStage.concept),
    ("Neuromorphic chip benchmarking platform for edge AI",
     "Neuromorphic hardware lacks standardized performance comparison",
     "edge AI hardware engineers", "United States", MaturityStage.pre_seed),
    ("Psychedelic-assisted therapy session analytics",
     "Psychedelic therapy outcomes lack data infrastructure",
     "licensed psychedelic therapy clinics", "United States", MaturityStage.concept),
    ("Atmospheric water harvesting optimization for arid regions",
     "Water scarcity affects 2 billion people but AWG yield is unpredictable",
     "humanitarian NGOs and municipal water utilities", "Kenya", MaturityStage.concept),
    ("Seaweed aquaculture yield prediction platform",
     "Seaweed farming is expanding but yield modeling is primitive",
     "seaweed aquaculture operators", "Norway", MaturityStage.pre_seed),
    ("Vertical farm crop scheduling using digital twins",
     "Vertical farms overproduce low-margin crops due to poor forecasting",
     "vertical farm operations managers", "United States", MaturityStage.seed),
    ("Coral reef restoration project monitoring using AI",
     "Coral restoration efforts lack standardized outcome measurement",
     "marine conservation NGOs", "Australia", MaturityStage.pre_seed),
    ("Halal supply chain certification blockchain platform",
     "Halal certification fraud costs the industry $1.4B annually",
     "halal food importers and Islamic finance institutions", "Malaysia", MaturityStage.seed),
    ("Space debris remediation mission planning software",
     "LEO debris density will make launches uninsurable by 2035",
     "commercial space operators and national space agencies", "United States", MaturityStage.concept),
    ("Quantum-safe cryptography migration tool for enterprises",
     "NIST PQC standards require cryptographic infrastructure overhaul",
     "CISOs at financial institutions and defense contractors", "United States", MaturityStage.seed),
    ("Insect protein feed certification platform for aquaculture",
     "Aquaculture industry lacks standardized insect-feed safety certification",
     "aquaculture feed manufacturers", "Netherlands", MaturityStage.pre_seed),
    ("Biochar soil amendment marketplace for carbon farming",
     "Biochar carbon credits lack verified measurement methodology",
     "regenerative farmers and carbon credit buyers", "United States", MaturityStage.pre_seed),
    ("AR-guided traditional craft preservation platform",
     "Indigenous craft techniques are dying without digital preservation",
     "indigenous cultural organizations and artisan cooperatives", "Mexico", MaturityStage.concept),
    ("Soilless nitrogen-fixing crop cultivation system",
     "Synthetic fertilizer produces 1.5% of global CO2 and has no bio alternative",
     "precision agriculture startups and food security researchers", "United States", MaturityStage.concept),
    ("Exoskeleton rental-as-a-service for construction workers",
     "Musculoskeletal injuries cost construction industry $20B/year",
     "general contractors and labor unions", "United States", MaturityStage.seed),
    ("Synthetic emotional memory chip for elderly care robotics",
     "Companion robots fail to build emotional rapport with dementia patients",
     "elder care facility operators and robotics manufacturers", "Japan", MaturityStage.concept),
    ("Precision fermentation platform for dairy alternatives",
     "Plant-based dairy lacks native casein/whey protein structures",
     "food tech R&D directors", "United States", MaturityStage.seed),
    ("Dynamic zoning compliance AI for urban infill developers",
     "Zoning code interpretation delays urban housing projects by 18 months",
     "urban real estate developers and city planners", "United States", MaturityStage.mvp),
    ("Bioluminescence-based pathogen detection strips for water testing",
     "Point-of-care water quality testing requires lab-grade equipment",
     "environmental health agencies and rural water utilities", "India", MaturityStage.pre_seed),
    ("Real-time wildfire behavior prediction for prescribed burn planning",
     "Prescribed burns are cancelled due to inability to predict fire spread",
     "state forestry agencies and fire management teams", "United States", MaturityStage.seed),
    ("Piezoelectric kinetic energy harvesting for smart building floors",
     "Building sensor networks require constant battery replacement",
     "smart building developers and facility managers", "United States", MaturityStage.concept),
    ("Autonomous drone-based crop pollination service",
     "Honeybee colony collapse threatens $15B in US crop production",
     "almond and berry farmers in California", "United States", MaturityStage.pre_seed),
    ("Tactile internet haptic feedback platform for remote surgery",
     "Telesurgery lacks haptic feedback causing surgical errors",
     "neurosurgeons and robotic surgery device manufacturers", "United States", MaturityStage.concept),
    ("Cultural heritage NFT royalty management for indigenous artists",
     "Indigenous digital art is exploited without fair royalty structures",
     "indigenous artist cooperatives and digital rights organizations", "Canada", MaturityStage.pre_seed),
    ("Geothermal heat pump leasing for suburban residential",
     "Geothermal upfront cost ($20K) prevents mass adoption",
     "homeowners in cold climates and HVAC installers", "United States", MaturityStage.seed),
    ("Precision livestock genetics marketplace for small farms",
     "Small livestock farms lack access to elite genetics programs",
     "independent beef and dairy farmers", "United States", MaturityStage.mvp),
    ("Acoustic ecology monitoring for urban biodiversity planning",
     "City planners lack biodiversity baseline data for green corridor design",
     "urban planning departments and environmental consultants", "United Kingdom", MaturityStage.pre_seed),
    ("Predictive maintenance AI for desalination plant membranes",
     "Desalination membrane failures halt water production with no warning",
     "municipal desalination plant operators", "Saudi Arabia", MaturityStage.seed),
    ("Longevity biomarker tracking platform for functional medicine",
     "Functional medicine practitioners lack longitudinal biomarker benchmarks",
     "longevity clinics and biohacking individuals", "United States", MaturityStage.mvp),
],
}


def make_brief(raw, problem, audience, region, maturity):
    return IdeaBrief(
        idea_id=str(uuid4()),
        raw_input=raw,
        problem=problem,
        audience=audience,
        region=region,
        maturity_stage=maturity,
    )


def is_valid_plan(tasks):
    """Check all tasks have valid agent names."""
    if not tasks:
        return False
    for t in tasks:
        if t.agent not in AVAILABLE_AGENTS:
            return False
        if not t.inputs:
            return False
    return True


def task_accuracy(tasks, expected_agents):
    """Fraction of expected agents covered by generated tasks."""
    generated_agents = {t.agent for t in tasks}
    if not expected_agents:
        return 1.0
    return len(generated_agents & expected_agents) / len(expected_agents)


def run_experiment(mode="hybrid"):
    """mode: 'hybrid' | 'catalog' | 'llm'"""
    planner = ResearchPlanner(available_agents=AVAILABLE_AGENTS)

    if mode == "catalog":
        original_should_use = planner._should_use_llm
        planner._should_use_llm = lambda x: False
    elif mode == "llm":
        original_create = planner.create_plan
        def llm_only_plan(brief):
            tasks = planner._llm_plan(brief)
            return tasks if tasks else []
        planner.create_plan = llm_only_plan

    results = []
    llm_invocations = 0
    total = 0

    for industry, briefs in BRIEFS.items():
        gt_agents = GROUND_TRUTH_AGENTS[industry]
        is_novel = (industry == "novel")

        for (raw, prob, aud, reg, mat) in briefs:
            brief = make_brief(raw, prob, aud, reg, mat)
            total += 1

            t0 = time.perf_counter()
            try:
                tasks = planner.create_plan(brief)
            except Exception as e:
                tasks = []

            elapsed = time.perf_counter() - t0

            valid = is_valid_plan(tasks)
            n_tasks = len(tasks)
            accuracy = task_accuracy(tasks, gt_agents) if tasks else 0.0

            # Detect LLM invocation: if mode hybrid, check coverage
            if mode == "hybrid":
                tokens = [t.lower() for t in (brief.problem or brief.raw_input or "").split() if len(t) > 3]
                known = {"analytics","software","platform","technology","healthcare","fintech","saas","cloud",
                         "digital","ai","retail","payment","medical","biotech","telehealth","investment",
                         "hardware","manufacturing","logistics","mobility","energy","supply","agriculture","education"}
                coverage = sum(1 for t in tokens if t in known) / max(len(tokens), 1)
                novel_count = sum(1 for t in tokens if t not in known)
                used_llm = (coverage < 0.30 or novel_count >= 3) and planner.llm_client is not None
                if used_llm:
                    llm_invocations += 1

            results.append({
                "industry": industry,
                "novel": is_novel,
                "valid": valid,
                "n_tasks": n_tasks,
                "accuracy": accuracy,
                "elapsed": elapsed,
            })

    valid_pct = sum(r["valid"] for r in results) / total * 100
    novel_results = [r for r in results if r["novel"]]
    novel_acc = statistics.mean(r["accuracy"] for r in novel_results) * 100 if novel_results else 0.0
    mean_tasks = statistics.mean(r["n_tasks"] for r in results)
    std_tasks = statistics.stdev(r["n_tasks"] for r in results) if len(results) > 1 else 0.0
    llm_pct = llm_invocations / total * 100 if mode == "hybrid" else (100.0 if mode == "llm" else 0.0)

    # Per-industry breakdown
    per_industry = {}
    for ind in BRIEFS:
        ind_r = [r for r in results if r["industry"] == ind]
        per_industry[ind] = {
            "n": len(ind_r),
            "valid_pct": sum(r["valid"] for r in ind_r) / len(ind_r) * 100,
            "accuracy": statistics.mean(r["accuracy"] for r in ind_r) * 100,
        }

    return {
        "mode": mode,
        "total": total,
        "valid_pct": valid_pct,
        "novel_accuracy": novel_acc,
        "mean_tasks": mean_tasks,
        "std_tasks": std_tasks,
        "llm_pct": llm_pct,
        "per_industry": per_industry,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 1: PLANNER RELIABILITY BENCHMARK")
    print("=" * 60)

    all_results = {}
    for mode in ["catalog", "hybrid"]:
        print(f"\n--- Mode: {mode.upper()} ---")
        r = run_experiment(mode)
        all_results[mode] = r
        print(f"  Total briefs:          {r['total']}")
        print(f"  Valid plans:           {r['valid_pct']:.1f}%")
        print(f"  Novel-domain accuracy: {r['novel_accuracy']:.1f}%")
        print(f"  Mean tasks/plan:       {r['mean_tasks']:.2f} ± {r['std_tasks']:.2f}")
        print(f"  LLM invocations:       {r['llm_pct']:.1f}%")
        print(f"\n  Per-industry breakdown:")
        for ind, stats in r["per_industry"].items():
            print(f"    {ind:12s}: n={stats['n']:2d}  valid={stats['valid_pct']:.0f}%  accuracy={stats['accuracy']:.1f}%")

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "processed" / "qa" / "benchmark_planner_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
